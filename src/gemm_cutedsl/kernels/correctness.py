"""Correctness-first CuTe DSL GEMM baseline."""

from functools import cache

from .common import cute_modules


@cache
def correctness_kernel():
    """Baseline kernel used until a step-specific exercise kernel is written."""

    cutlass, cute, _from_dlpack, Float32, Int32 = cute_modules()

    @cute.kernel
    def gemm_kernel(
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        m: Int32,
        n: Int32,
        k: Int32,
    ):
        tid = cute.arch.thread_idx()[0]
        bx = cute.arch.block_idx()[0]
        by = cute.arch.block_idx()[1]
        row = bx * 16 + tid // 16
        col = by * 16 + tid % 16
        if row < m and col < n:
            acc = Float32(0.0)
            for kk in cutlass.range(k):
                acc += a[row, kk].to(Float32) * b[col, kk].to(Float32)
            c[row, col] = acc.to(c.element_type)

    return gemm_kernel


@cache
def correctness_launcher():
    """Host-side JIT wrapper that launches `correctness_kernel()`."""

    _cutlass, cute, _from_dlpack, _Float32, _Int32 = cute_modules()

    @cute.jit
    def launch_gemm(
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        m: int,
        n: int,
        k: int,
    ):
        correctness_kernel()(a, b, c, m, n, k).launch(
            grid=((m + 15) // 16, (n + 15) // 16, 1),
            block=(256, 1, 1),
        )

    return launch_gemm

