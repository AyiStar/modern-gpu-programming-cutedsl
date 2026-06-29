"""Correctness and benchmark helpers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from .kernel import describe_kernel, reference_gemm, run_gemm
from .spec import GemmProblem, problem_from_dims, validate_problem


@dataclass(frozen=True)
class ErrorStats:
    max_abs: float
    mean_abs: float


@dataclass(frozen=True)
class BenchmarkResult:
    arch: str
    step: int
    m: int
    n: int
    k: int
    gpu_name: str
    capability: str
    torch_version: str
    cuda_version: str | None
    cute_dsl_version: str | None
    warmup: int
    iters: int
    latency_ms: float
    tflops: float
    max_abs_error: float
    mean_abs_error: float
    kernel: dict[str, object]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tflops(problem: GemmProblem, latency_seconds: float) -> float:
    if latency_seconds <= 0:
        return 0.0
    return problem.flops / latency_seconds / 1.0e12


def _error_stats(actual: object, expected: object) -> ErrorStats:
    import torch

    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError("actual and expected must be torch.Tensor instances")
    diff = (actual.float() - expected.float()).abs()
    return ErrorStats(max_abs=float(diff.max().item()), mean_abs=float(diff.mean().item()))


def _cute_dsl_version() -> str | None:
    try:
        import cutlass  # type: ignore[import-not-found]
    except Exception:
        return None
    return getattr(cutlass, "__version__", "unknown")


def _gpu_info() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        return ("cpu", "none")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (props.name, f"{props.major}.{props.minor}")


def _make_inputs(problem: GemmProblem, seed: int) -> tuple[object, object]:
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn(problem.a_shape, device="cuda", dtype=torch.float16, generator=generator)
    b = torch.randn(problem.b_shape, device="cuda", dtype=torch.float16, generator=generator)
    return a.contiguous(), b.contiguous()


def benchmark(
    arch: str,
    step: int,
    m: int,
    n: int,
    k: int,
    *,
    warmup: int = 5,
    iters: int = 20,
    seed: int = 0,
) -> BenchmarkResult:
    import torch

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iters <= 0:
        raise ValueError("iters must be positive")
    problem = validate_problem(step, m, n, k)
    a, b = _make_inputs(problem, seed)

    actual = run_gemm(arch, step, a, b)
    expected = reference_gemm(a, b)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)
    errors = _error_stats(actual, expected)

    for _ in range(warmup):
        run_gemm(arch, step, a, b)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    for _ in range(iters):
        run_gemm(arch, step, a, b)
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - start_wall
    event_ms = float(start_event.elapsed_time(end_event)) / iters
    latency_seconds = event_ms / 1000.0 if event_ms > 0 else wall_seconds / iters

    gpu_name, capability = _gpu_info()
    return BenchmarkResult(
        arch=arch,
        step=step,
        m=problem.m,
        n=problem.n,
        k=problem.k,
        gpu_name=gpu_name,
        capability=capability,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cute_dsl_version=_cute_dsl_version(),
        warmup=warmup,
        iters=iters,
        latency_ms=latency_seconds * 1000.0,
        tflops=tflops(problem, latency_seconds),
        max_abs_error=errors.max_abs,
        mean_abs_error=errors.mean_abs,
        kernel=describe_kernel(arch, step),
    )


def run_suite(
    arch: str,
    steps: tuple[int, ...],
    *,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    warmup: int = 5,
    iters: int = 20,
    seed: int = 0,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for offset, step in enumerate(steps):
        problem = problem_from_dims(step, m, n, k)
        results.append(
            benchmark(
                arch,
                step,
                problem.m,
                problem.n,
                problem.k,
                warmup=warmup,
                iters=iters,
                seed=seed + offset,
            )
        )
    return results
