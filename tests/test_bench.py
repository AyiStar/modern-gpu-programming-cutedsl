from __future__ import annotations

from gemm_cutedsl.bench import (
    BenchmarkResult,
    CorrectnessResult,
    _correctness_problem_from_dims,
    tflops,
)
from gemm_cutedsl.profiling import ProfileResult, add_speedups
from gemm_cutedsl.spec import validate_problem


def test_tflops_calculation() -> None:
    problem = validate_problem(1, 128, 128, 64)
    assert tflops(problem, 0.0) == 0.0
    assert tflops(problem, 1.0) == problem.flops / 1.0e12


def test_benchmark_result_to_dict() -> None:
    result = BenchmarkResult(
        arch="h100",
        step=1,
        m=128,
        n=128,
        k=64,
        gpu_name="NVIDIA H100",
        capability="9.0",
        torch_version="2.x",
        cuda_version="12.x",
        cute_dsl_version="4.x",
        warmup=1,
        iters=2,
        latency_ms=0.5,
        tflops=1.0,
        max_abs_error=0.0,
        mean_abs_error=0.0,
        kernel={"name": "single_tile"},
    )
    data = result.to_dict()
    assert data["arch"] == "h100"
    assert data["latency_ms"] == 0.5


def test_correctness_result_to_dict() -> None:
    result = CorrectnessResult(
        arch="h100",
        step=4,
        m=256,
        n=384,
        k=128,
        gpu_name="NVIDIA H100",
        capability="9.0",
        torch_version="2.x",
        cuda_version="12.x",
        cute_dsl_version="4.x",
        max_abs_error=0.0,
        mean_abs_error=0.0,
        kernel={"name": "tma_async"},
    )
    data = result.to_dict()
    assert data["step"] == 4
    assert data["n"] == 384


def test_correctness_problem_selection_uses_step_defaults_without_dims() -> None:
    step4 = _correctness_problem_from_dims(4, None, None, None)
    step9 = _correctness_problem_from_dims(9, None, None, None)
    explicit = _correctness_problem_from_dims(3, 512, 512, 512)

    assert (step4.m, step4.n, step4.k) == (256, 384, 128)
    assert (step9.m, step9.n, step9.k) == (512, 256, 128)
    assert explicit.a_shape == (512, 512)


def test_add_speedups() -> None:
    first = BenchmarkResult(
        arch="h100",
        step=1,
        m=128,
        n=128,
        k=64,
        gpu_name="NVIDIA H100",
        capability="9.0",
        torch_version="2.x",
        cuda_version="12.x",
        cute_dsl_version="4.x",
        warmup=1,
        iters=2,
        latency_ms=2.0,
        tflops=1.0,
        max_abs_error=0.0,
        mean_abs_error=0.0,
        kernel={"name": "single_tile"},
    )
    second = BenchmarkResult(
        arch="h100",
        step=2,
        m=128,
        n=128,
        k=256,
        gpu_name="NVIDIA H100",
        capability="9.0",
        torch_version="2.x",
        cuda_version="12.x",
        cute_dsl_version="4.x",
        warmup=1,
        iters=2,
        latency_ms=1.0,
        tflops=2.0,
        max_abs_error=0.0,
        mean_abs_error=0.0,
        kernel={"name": "single_tile_k_loop"},
    )
    rows = add_speedups([first, second])
    assert rows[0]["speedup_vs_first"] == 1.0
    assert rows[1]["speedup_vs_first"] == 2.0


def test_profile_result_to_dict() -> None:
    result = ProfileResult(
        arch="b200",
        step=9,
        m=4096,
        n=4096,
        k=4096,
        warmup=1,
        active=2,
        profiler_table="cuda table",
        benchmark={"latency_ms": 1.0},
    )
    assert result.to_dict()["profiler_table"] == "cuda table"
