import torch
import triton
import triton.language as tl
import time
import os
from torch.testing import assert_close
import numpy as np

os.environ["ENABLE_TMA"] = "1"


@triton.jit
def grouped_launch(
    pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr
):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.jit
def col_major(pid, m, n, block_m: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)

    pid_m = pid % grid_m
    pid_n = pid // grid_m

    return pid_m, pid_n


@triton.jit
def gemm_split_k_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scale_a_ptr,
    scale_b_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    m,
    n,
    k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    split_k: tl.constexpr,
    group_m: tl.constexpr,
):
    # Get the device scales
    scale_a = tl.load(scale_a_ptr, mask=True, other=0.0)
    scale_b = tl.load(scale_b_ptr, mask=True, other=0.0)

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k * split_k)

    pid_m, pid_n = col_major(pid, m, n, block_m)
    # pid_m, pid_n = grouped_launch(pid, m, n, block_m, block_n, group_m)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    # tl.device_print("pid_n", pid_n)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    # tl.device_print("offs_bn", offs_bn)

    # tl.device_print("stride_am", stride_am)
    # tl.device_print("stride_ak", stride_ak)
    # tl.device_print("stride_bk", stride_bk)
    # tl.device_print("stride_bn", stride_bn)
    # tl.device_print("stride_cm", stride_cm)
    # tl.device_print("stride_cn", stride_cn)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # tl.device_print("offs_bn * stride_bn", offs_bn * stride_bn)
    # tl.device_print("n", n)
    # tl.device_print("k", k)
    # tl.device_print("m", m)
    # tl.device_assert(offs_bn * stride_bn < n * k, "access b_ptr out of bounds along n")

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_ in range(0, grid_k):
        k_remaining = k - k_ * (block_k * split_k)
        m_remaining = m - pid_m * block_m
        n_remaining = n - pid_n * block_n

        valid_m = offs_m < m_remaining
        valid_k = offs_k < k_remaining

        a = tl.load(a_ptrs, mask=valid_k[None, :] & valid_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=valid_k[:, None], other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        # acc += tl.dot(a, b)

        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk
    acc = scale_a * scale_b * acc
    acc.to(tl.float16)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]

    tl.atomic_add(c_ptrs, acc, mask=mask)


def gemm_split_k(a, b, c, scale_a, scale_b):
    m, k = a.shape
    n, _ = b.shape
    b_strides = (1, k)

    # TODO(csullivan): good config for M, N, K = 16, 28672, 4096
    block_m = 64
    block_n = 64
    block_k = 128
    num_stages = 3
    num_warps = 4
    split_k = 2

    # TODO(csullivan): good config for M, N, K = 16, 4096, 4096
    # block_m = 64
    # block_n = 64
    # block_k = 256
    # num_stages = 3
    # num_warps = 4
    # split_k = 2

    # TODO(csullivan): good config for M, N, K = 17, 4096, 4096
    # block_m = 64
    # block_n = 64
    # block_k = 512
    # num_stages = 3
    # num_warps = 8
    # split_k = 2

    # ! works with ada instructions on hopper for (17, 28672, 4096)
    # block_m = 16
    # block_n = 64
    # block_k = 512
    # num_stages = 3
    # num_warps = 8
    # split_k = 2

    # TODO(csullivan): good config for M, N, K = 16, 4096, 14336
    # block_m = 64
    # block_n = 64
    # block_k = 256
    # num_stages = 4
    # num_warps = 4
    # split_k = 2

    # TODO(csullivan): good config for M, N, K = 16, 6144, 4096
    # block_m = 64
    # block_n = 64
    # block_k = 128
    # num_stages = 4
    # num_warps = 4
    # split_k = 2

    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    total_programs_mn = total_blocks_m * total_blocks_n
    total_programs_k = split_k

    grid = (total_programs_mn, total_programs_k)

    # print(f"problem m size: {m}, tile size m: {block_m}, total blocks m: {total_blocks_m}")
    # print(f"problem n size: {n}, tile size n: {block_n}, total blocks n: {total_blocks_n}")
    # print(f"problem k size: {k}, tile size k: {block_k}, total thread blocks k: {split_k}")

    # print(f"total thread blocks k: {k}, total thread blocks m and total thread blocks n = {total_blocks_m=} x {total_blocks_n} = {total_programs_mn}")
    # print(f"{total_programs_mn=}, {total_programs_k=}")

    print(grid)
    k = gemm_split_k_kernel[grid](
        a,  # 0*
        b,  # 1*
        c,  # 2*
        scale_a,  # 3*
        scale_b,  # 4*
        a.stride(0),  # 5*
        a.stride(1),  # 6
        b_strides[0],  # 7
        b_strides[1],  # 8*
        c.stride(0),  # 9*
        c.stride(1),  # 10
        m,  # 11*
        n,  # 12*
        k,  # 13*
        block_m,
        block_n,
        block_k,
        split_k,
        group_m=8,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    # print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n")

    with open("matmul_split_k2.ptx", "w") as f:
        #     print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)
        #     print("IR", k.asm['ttir'], file=f)
        #     print("TTGIR", k.asm['ttgir'], file=f)
        print(k.asm["ptx"], file=f)
        # print(
        #     f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n",
        #     file=f,
        # )


def bench(func, num_iterations, output=None):
    # warm up
    ret = func()
    if output != None:
        ret = output.cpu().numpy()
    start = time.time()
    for _ in range(num_iterations):
        func()
    stop = time.time()
    return ret, start, stop


if __name__ == "__main__":
    print("start")
    torch.cuda.manual_seed(0)
    num_iterations = 1000

    # M, N, K = 17, 4096, 4096
    # M, N, K = 17, 28672, 4096
    # M, N, K = 17, 4096, 14336
    # M, N, K = 17, 6144, 4096

    m_n_k_tuple = (
        (17, 4096, 4096),
        (17, 28672, 4096),
        (17, 4096, 14336),
        (17, 6144, 4096),
    )

    for M, N, K in m_n_k_tuple:
        print("Gemm shape:", M, N, K)
        # a_ = torch.ones((M, K), device="cuda", dtype=torch.float16)
        # # actual data layout is KN
        # b_ = torch.ones((N, K), device="cuda", dtype=torch.float16)
        # b_[0, 1] = 10
        a_ = torch.randn((M, K), device="cuda", dtype=torch.float16)
        # actual data layout is KN
        b_ = torch.randn((N, K), device="cuda", dtype=torch.float16)
        # scale_a = torch.randn((1,), device="cuda", dtype=torch.float32)
        # scale_b = torch.randn((1,), device="cuda", dtype=torch.float32)
        scale_a = torch.from_numpy(np.array([1])).to(device="cuda", dtype=torch.float32)
        scale_b = torch.from_numpy(np.array([1])).to(device="cuda", dtype=torch.float32)

        # a_ = torch.randn((M, K), device="cuda", dtype=torch.float16)
        # b_ = torch.randn((K, N), device="cuda", dtype=torch.float16).T
        a_ = a_.to(torch.float8_e4m3fn)
        b_ = b_.to(torch.float8_e4m3fn)

        ####
        ## After lunch, bring in TVM cublas and compare, possibly use ptx to make kernel and compare e2e thereafter
        ####

        c_ = torch.zeros((M, N), device=a_.device, dtype=torch.float16)

        result, start, stop = bench(
            lambda: gemm_split_k(a_, b_, c_, scale_a, scale_b),
            num_iterations,
            output=c_,
        )
        print(f"Triton FP8 {stop-start}\n")

        # ret1, start, stop = bench(
        #     lambda: torch._scaled_mm(
        #         a_, b_.T, out_dtype=torch.float16, use_fast_accum=True
        #     ),
        #     num_iterations,
        # )
        # print(f"cuBLAS FP8 {stop-start}\n")

        # a = torch.zeros((M, K), device="cuda", dtype=torch.float16)
        # b = torch.zeros((K, N), device="cuda", dtype=torch.float16).T
        # ret, start, stop = bench(lambda: torch.matmul(a, b), num_iterations)
        # print(f"Triton FP16 {stop-start}\n")

        # a_f16 = torch.ones((M, K), device="cuda", dtype=torch.float16)
        # b_f16 = torch.ones((K, N), device="cuda", dtype=torch.float16)
        # b_f16[1, 0] = 10
        a_f16 = a_.to(torch.float16)
        b_f16 = b_.to(torch.float16).T
        # ret, start, stop = bench(lambda: torch.matmul(a_f16, b_f16), num_iterations)

        golden = torch.matmul(a_f16, b_f16)
        print("C1:\n", result)
        # print("C2:\n", ret1)
        print("Ref:\n", golden)
        print("Ratio:\n", golden.cpu().numpy() / result)
        # assert_close(
        #     result, golden.cpu().numpy(), rtol=0.1, atol=1e-3, check_dtype=False
        # )
