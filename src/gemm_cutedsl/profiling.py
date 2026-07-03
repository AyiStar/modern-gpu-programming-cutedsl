"""Modal-side performance tables and torch profiler summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .bench import BenchmarkResult, benchmark
from .kernel import reference_gemm, run_gemm
from .spec import GemmProblem, problem_from_dims, validate_problem


@dataclass(frozen=True)
class ProfileResult:
    arch: str
    step: int
    m: int
    n: int
    k: int
    warmup: int
    active: int
    profiler_table: str
    benchmark: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_speedups(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    rows = [result.to_dict() for result in results]
    if not rows:
        return rows
    baseline_ms = rows[0]["latency_ms"]
    for row in rows:
        latency_ms = row["latency_ms"]
        row["speedup_vs_first"] = baseline_ms / latency_ms if latency_ms > 0 else 0.0
    return rows


def _make_inputs(problem: GemmProblem, seed: int) -> tuple[object, object]:
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn(
        problem.a_shape,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    b = torch.randn(
        problem.b_shape,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    return a.contiguous(), b.contiguous()


def profile_step(
    arch: str,
    step: int,
    m: int,
    n: int,
    k: int,
    *,
    warmup: int = 5,
    active: int = 5,
    row_limit: int = 20,
    seed: int = 0,
) -> ProfileResult:
    import torch

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if active <= 0:
        raise ValueError("active profiler iterations must be positive")
    if row_limit <= 0:
        raise ValueError("row_limit must be positive")

    problem = validate_problem(step, m, n, k)
    a, b = _make_inputs(problem, seed)
    actual = run_gemm(arch, step, a, b)
    expected = reference_gemm(a, b)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)

    for _ in range(warmup):
        run_gemm(arch, step, a, b)
    torch.cuda.synchronize()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        for _ in range(active):
            run_gemm(arch, step, a, b)
            profiler.step()
        torch.cuda.synchronize()

    table = profiler.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=row_limit,
    )
    perf = benchmark(
        arch,
        step,
        problem.m,
        problem.n,
        problem.k,
        warmup=warmup,
        iters=active,
        seed=seed,
    )
    return ProfileResult(
        arch=arch,
        step=step,
        m=problem.m,
        n=problem.n,
        k=problem.k,
        warmup=warmup,
        active=active,
        profiler_table=table,
        benchmark=perf.to_dict(),
    )


def profile_suite(
    arch: str,
    steps: tuple[int, ...],
    *,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    warmup: int = 5,
    active: int = 5,
    row_limit: int = 20,
    seed: int = 0,
) -> list[ProfileResult]:
    profiles: list[ProfileResult] = []
    for offset, step in enumerate(steps):
        problem = problem_from_dims(step, m, n, k)
        profiles.append(
            profile_step(
                arch,
                step,
                problem.m,
                problem.n,
                problem.k,
                warmup=warmup,
                active=active,
                row_limit=row_limit,
                seed=seed + offset,
            )
        )
    return profiles
