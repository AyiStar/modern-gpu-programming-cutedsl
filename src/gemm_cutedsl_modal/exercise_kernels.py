"""Write the Step 1-9 CuTe DSL exercise kernels in this file.

Each `step*_kernel()` function returns a compiled CuTe DSL kernel object.  The
project currently routes every step to `correctness_kernel()`, a tiny baseline
where one CUDA thread computes one output element.  Replace one function at a
time while working through the tutorial:

- Step 1-3: synchronous tiled GEMM basics
- Step 4-6: TMA, pipeline, persistent scheduling
- Step 7-9: warp specialization, clusters, multi-consumer scheduling

The surrounding Modal app, validation, benchmarking, and profiling code does
not need to change when you replace a step kernel.
"""

from functools import cache


@cache
def cute_modules():
    try:
        import cutlass  # type: ignore[import-not-found]
        import cutlass.cute as cute  # type: ignore[import-not-found]
        from cutlass.cutlass_dsl import Float32, Int32  # type: ignore[import-not-found]
        from cutlass.cute.runtime import from_dlpack  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "CuTe DSL is required. Run through Modal so the image installs "
            "nvidia-cutlass-dsl, torch, and CUDA."
        ) from exc
    return cutlass, cute, from_dlpack, Float32, Int32


@cache
def correctness_kernel():
    """Correctness-first baseline used until a step-specific kernel is written."""

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
    def launch_gemm(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, m: int, n: int, k: int):
        correctness_kernel()(a, b, c, m, n, k).launch(
            grid=((m + 15) // 16, (n + 15) // 16, 1),
            block=(256, 1, 1),
        )

    return launch_gemm


def step1_kernel():
    """TODO: single 128x128 tile, K=64."""

    return correctness_launcher()


def step2_kernel():
    """TODO: single 128x128 output tile with a K loop."""

    return correctness_launcher()


def step3_kernel():
    """TODO: 2D CTA grid, one CTA per 128x128 output tile."""

    return correctness_launcher()


def step4_kernel():
    """TODO: replace synchronous GMEM->SMEM copies with TMA async load/store."""

    return correctness_launcher()


def step5_kernel():
    """TODO: add PIPE_DEPTH=2 staged SMEM ring buffer."""

    return correctness_launcher()


def step6_kernel():
    """TODO: wrap Step 5 in a persistent CTA tile scheduler."""

    return correctness_launcher()


def step7_kernel():
    """TODO: split producer, MMA consumer, and writeback roles."""

    return correctness_launcher()


def step8_kernel():
    """TODO: 2-CTA cluster, 256x256 cluster output tile."""

    return correctness_launcher()


def step9_kernel():
    """TODO: two MMA consumers, shared B tile, 512x256 cluster output tile."""

    return correctness_launcher()


STEP_KERNELS = {
    1: step1_kernel,
    2: step2_kernel,
    3: step3_kernel,
    4: step4_kernel,
    5: step5_kernel,
    6: step6_kernel,
    7: step7_kernel,
    8: step8_kernel,
    9: step9_kernel,
}


def get_step_kernel(step: int):
    try:
        return STEP_KERNELS[step]()
    except KeyError as exc:
        raise ValueError(f"invalid step {step!r}; expected 1-9") from exc
