import torch
import triton
import triton.language as tl
import time
import os
from torch.testing import assert_close

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


@triton.jit()
def col_major(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)

    pid_m = pid % grid_m
    pid_n = pid // grid_m

    return pid_m, pid_n


@triton.jit
def gemm_split_k_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k * split_k)

    pid_m, pid_n = grouped_launch(pid, m, n, block_m, block_n, group_m)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_ in range(0, grid_k):
        k_remaining = k - k_ * (block_k * split_k)

        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        # acc += tl.dot(a, b)

        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk
    acc.to(tl.float16)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]

    tl.atomic_add(c_ptrs, acc, mask=mask)


def gemm_split_k(a, b, c):
    m, k = a.shape
    _, n = b.shape

    block_m = 16
    block_n = 64
    block_k = 512
    num_stages = 3
    num_warps = 8
    split_k = 2
    group_m = 1

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

    k = gemm_split_k_kernel[grid](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        split_k,
        group_m,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    # print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n")

    with open("matmul_split_k.txt", "w") as f:
        #     print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)
        #     print("IR", k.asm['ttir'], file=f)
        #     print("TTGIR", k.asm['ttgir'], file=f)
        print("PTX", k.asm["ptx"], file=f)
        # print(
        #     f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n",
        #     file=f,
        # )


def bench(func, num_iterations):
    # warm up
    ret = func()
    start = time.time()
    for _ in range(num_iterations):
        func()
    stop = time.time()
    return ret, start, stop


if __name__ == "__main__":
    torch.cuda.manual_seed(0)
    num_iterations = 1000
    # a_ = torch.ones((16, 4096), device="cuda", dtype=torch.float16)
    # b_ = torch.ones((4096, 4096), device="cuda", dtype=torch.float16).T
    a_ = torch.randn((16, 4096), device="cuda", dtype=torch.float16)
    b_ = torch.randn((4096, 4096), device="cuda", dtype=torch.float16).T
    a_ = a_.to(torch.float8_e4m3fn)
    b_ = b_.to(torch.float8_e4m3fn)

    # ret1, start, stop = bench(
    #     lambda: torch._scaled_mm(a_, b_, out_dtype=torch.float16, use_fast_accum=True),
    #     num_iterations,
    # )
    # print(f"cuBLAS FP8 {stop-start}\n")

    c_ = torch.zeros((16, 4096), device=a_.device, dtype=torch.float16)
    ret2, start, stop = bench(lambda: gemm_split_k(a_, b_, c_), num_iterations)
    print(f"Triton FP8 {stop-start}\n")

    # a = torch.zeros((16, 4096), device="cuda", dtype=torch.float16)
    # b = torch.zeros((4096, 4096), device="cuda", dtype=torch.float16).T
    # ret, start, stop = bench(lambda: torch.matmul(a, b), num_iterations)
    # print(f"Triton FP16 {stop-start}\n")

    # a_f32 = a_.to(torch.float32)
    # b_f32 = b_.to(torch.float32)
    # golden = torch.matmul(a_f32, b_f32)
    # print("C1:\n", c_)
    # # print("C2:\n", ret1)
    # print("Ref:\n", golden)
    # print(golden.cpu().numpy()[10, 2175], c_.cpu().numpy()[10, 2175])
    # assert_close(c_, golden, rtol=10, atol=1e-3, check_dtype=False)
