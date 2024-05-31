import torch
from torch.testing import assert_close
import cupy as cp
import ml_dtypes
import time
import numpy as np


def bench(func, num_iterations, output=None):
    # warm up
    ret = func()
    if output != None:
        ret = output.cpu()
    start = time.time()
    for _ in range(num_iterations):
        func()
    stop = time.time()
    return ret, start, stop


def count_mismatches(tensor1, tensor2, atol=1e-8, rtol=1e-5):
    """
    Counts the number of mismatched elements between two tensors
    within the specified absolute (atol) and relative (rtol) tolerances.

    Args:
        tensor1 (torch.Tensor): The first tensor for comparison.
        tensor2 (torch.Tensor): The second tensor for comparison.
        atol (float): The absolute tolerance.
        rtol (float): The relative tolerance.

    Returns:
        int: The number of mismatched elements.
    """
    mismatches = ~torch.isclose(tensor1, tensor2, atol=atol, rtol=rtol)
    mismatch_count = torch.sum(mismatches).item()
    total_elements = tensor1.numel()  # Total number of elements in the tensor
    fraction_mismatch = mismatch_count / total_elements
    return mismatch_count, fraction_mismatch


def ceildiv(a, b):
    return -(a // -b)


# Define the path to the PTX file containing the matmul_kernel
ptx_code_path = "./matmul_split_k2_f16.ptx.save"

# Create a RawModule object to load the PTX code
raw_module = cp.RawModule(path=ptx_code_path)

# Retrieve the matmul kernel function from the PTX code
matmul_kernel = raw_module.get_function("scaled_gemm_split_k_kernel")

# Define the dimensions of the matrices
m_n_k_tuple = (
    # (5, 4096, 4096),
    # (5, 28672, 4096),
    # (5, 4096, 14336),
    (5, 6144, 4096),
)
for M, N, K in m_n_k_tuple:
    block_m = 64
    block_n = 64
    block_k = 128
    num_warps = 4
    split_k = 2

    # Create random matrices A and B and an output matrix C
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    scale_a = torch.from_numpy(np.array([1])).to(device="cuda", dtype=torch.float32)
    scale_b = torch.from_numpy(np.array([1])).to(device="cuda", dtype=torch.float32)
    # A = A.to(torch.float8_e4m3fn)
    # B = B.to(torch.float8_e4m3fn)

    # A = torch.full((M, K), 1.0, dtype=torch.float8_e4m3fn, device="cuda")
    # B = torch.full((K, N), 1.0, dtype=torch.float8_e4m3fn, device="cuda")

    C = torch.zeros((M, N), device=A.device, dtype=torch.float16)

    A_f32 = A.to(torch.float32)
    B_f32 = B.to(torch.float32)
    golden = torch.matmul(A_f32, B_f32.T)

    # Set the strides for each matrix
    stride_am, _ = K, 1
    _, stride_bk = 1, K
    stride_zm, _ = N, 1

    # Set grid and block sizes for the kernel launch
    total_blocks_m = ceildiv(M, block_m)
    total_blocks_n = ceildiv(N, block_n)
    total_programs_mn = total_blocks_m * total_blocks_n
    total_programs_k = split_k

    grid = (total_programs_mn, total_programs_k, 1)

    block = (num_warps * 32, 1, 1)
    print(grid, block)
    shared_memory_size = 98304

    # Launch the kernel
    num_iterations = 1000
    result, start, stop = bench(
        lambda: matmul_kernel(
            grid,
            block,
            (
                A.data_ptr(),
                B.data_ptr(),
                C.data_ptr(),
                scale_a.data_ptr(),
                scale_b.data_ptr(),
                stride_am,
                stride_bk,
                stride_zm,
                M,
                N,
                K,
            ),
            shared_mem=shared_memory_size,
        ),
        num_iterations,
        output=C,
    )

    # Print the output matrix C
    print("Output Matrix C:")
    print(result)

    print("Ref: ")
    print(golden)

    mismatch_count, fraction_mismatch = count_mismatches(
        result.to(torch.float32), golden.cpu(), rtol=1e-1, atol=1e-3
    )
    print(f"Percent mismatch: {fraction_mismatch}")
    assert (
        fraction_mismatch < 0.01
    ), "Greater than 1% of elements mismatched with float32 results"
