# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import triton
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.activation import SiluAndMul
from v0_moe_fused import fused_moe as fused_moe_grouped
from v2_moe_fused import fused_moe as fused_moe_col
import time
import numpy as np


def torch_moe(a, w1, w2, topk_weight, topk_ids):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk_ids.shape[1], 1).reshape(-1, D)
    out = torch.zeros(
        B * topk_ids.shape[1], w2.shape[1], dtype=a.dtype, device=a.device
    )

    topk_ids = topk_ids.view(-1)
    topk_weight = topk_weight.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(
                0, 1
            )
    return (out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1)).sum(dim=1)


def torch_moe_first_layer(a, w1, topk_ids):
    B, D = a.shape
    a = a.view(B, -1, D)
    a = a.repeat(1, topk_ids.shape[1], 1)
    out = torch.zeros(B, topk_ids.shape[1], w1.shape[1], dtype=a.dtype, device=a.device)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = a[mask] @ w1[i].transpose(0, 1)
    return out


def modified_torch_moe_first_layer(a, w1, expert_ids, sorted_token_ids, topk):
    B, D = a.shape
    a = a.view(B, 1, D).repeat(1, topk, 1)
    out = torch.zeros(B, topk, w1.shape[1], dtype=a.dtype, device=a.device)

    for i in range(w1.shape[0]):  # Iterate over experts
        mask = expert_ids[sorted_token_ids // topk] == i
        if mask.sum():
            out[mask] = a[mask] @ w1[i].transpose(0, 1)

    return out


def verify_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    torch.cuda.manual_seed(3227)

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    score = torch.softmax(score, dim=-1)

    topk_weight, topk_ids = torch.topk(score, topk)

    start = time.time()
    triton_output_gl = fused_moe_grouped(a, w1, w2, topk_weight, topk_ids, False)
    end = time.time()
    gl_time = end - start
    gl_time = gl_time * 1000
    print("Grouped Launch Time (us): ", gl_time)

    start = time.time()
    triton_output_cm = fused_moe_col(a, w1, w2, topk_weight, topk_ids, False)
    end = time.time()
    cm_major_time = end - start
    cm_major_time = cm_major_time * 1000
    print("Columm Major Time (us): ", cm_major_time)

    torch_base = torch_moe_first_layer(a, w1, topk_ids)
    torch.testing.assert_close(triton_output_cm, torch_base, atol=1e-2, rtol=0)

    ggemm_out = torch.zeros(
        m, topk_ids.shape[1], w1.shape[1], dtype=a.dtype, device=a.device
    )

    expert_ids = topk_ids.view(-1)
    sorted_token_ids = torch.arange(m, device="cuda").repeat_interleave(topk)

    ggemm_out = moe_via_ggemm(a, w1, ggemm_out, topk_ids)
    torch.testing.assert_close(ggemm_out, torch_base, atol=1e-2, rtol=0)

    # print(f"{triton_output_cm=}\n")
    # print(f"{triton_output_gl=}\n")

    print(f"Col Major Speedup {((gl_time - cm_major_time)/(gl_time))*100}")


def moe_via_ggemm(A, B, C, topk_ids):
    E, N, K = B.shape
    M, topk = topk_ids.shape

    # Flatten and sort topk_ids
    expert_ids = topk_ids.flatten()
    sorted_indices = torch.argsort(expert_ids)
    sorted_expert_ids = expert_ids[sorted_indices]

    # Repeat and sort input A
    A_repeated = A.repeat_interleave(topk, dim=0)
    A_sorted = A_repeated[sorted_indices]

    # Calculate the bincount and indptr
    bincounts = torch.bincount(sorted_expert_ids, minlength=E)
    indptr = torch.cumsum(bincounts, dim=0)

    # Perform group gemm
    print(f"indptr = {indptr}")
    result = group_gemm(A_sorted, B, indptr)

    # Reshape and reorder the result
    C = torch.zeros((M, topk, N), device=A.device, dtype=A.dtype)
    C.view(-1, N)[sorted_indices] = result
    return C


def group_gemm(A, B, indptr):
    M, K = A.shape
    E, N, K_B = B.shape
    assert K == K_B, "K dimensions of A and B must match"

    result = torch.zeros((M, N), device=A.device, dtype=A.dtype)

    for e in range(E):
        if e == 0:
            start = 0
        else:
            start = indptr[e - 1]
        end = indptr[e]

        if start == end:
            # No tokens for this expert
            continue
        elif start > end:
            raise ValueError("indptr must be monotonically increasing")

        result[start:end] = torch.mm(A[start:end], B[e].T)

    return result


def test_numerics():
    verify_fused_moe(16, 14336 // 2, 4096, 8, 2, torch.float16)


def test_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m"],  # Argument names to use as an x-axis for the plot
            x_vals=[
                2**i for i in [4]  # range(0, 10)
            ],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=["cm", "gl"],
            # Label name for the lines
            line_names=["Fused MoE GEMM Kernel - Column Major", "vLLM MoE GEMM Kernel"],
            # Line styles
            styles=[("blue", "-"), ("green", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="test",  # Name for the plot, used also as a file name for saving the plot.
            args={},
        )
    )
    def benchmark(m, provider):

        m = m
        # n = 4096//2
        n = 14336 // 2
        k = 4096
        e = 8
        topk = 2

        torch.cuda.manual_seed(3227)
        dtype = torch.float16

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        score = torch.softmax(score, dim=-1)
        topk_weight, topk_ids = torch.topk(score, topk)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "cm":
            # w1_f8 = w1.to(torch.float8_e4m3fn)
            # w2_f8 = w2.to(torch.float8_e4m3fn)
            # w1_f8 = torch.permute(w1_f8, (0, 2, 1))
            # w2_f8 = torch.permute(w2_f8, (0, 2, 1))
            ms, min_ms, max_ms = triton.testing.do_bench(
                # lambda: fused_moe_col(a, w1_f8, w2_f8, topk_weight, topk_ids, False),
                lambda: fused_moe_col(a, w1, w2, topk_weight, topk_ids, False),
                quantiles=quantiles,
            )
        if provider == "gl":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_moe_grouped(a, w1, w2, topk_weight, topk_ids, False),
                quantiles=quantiles,
            )
        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True, save_path="./")
