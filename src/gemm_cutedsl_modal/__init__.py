"""Modal-hosted CuTe DSL GEMM practice kernels for H100 and B200."""

from .bench import BenchmarkResult, benchmark, run_suite
from .kernel import describe_kernel, run_gemm
from .profiling import ProfileResult, add_speedups, profile_step, profile_suite
from .spec import (
    ARCHES,
    STEP_SPECS,
    GemmProblem,
    StepSpec,
    default_problem,
    normalize_arch,
    normalize_steps,
    validate_problem,
)

__all__ = [
    "ARCHES",
    "STEP_SPECS",
    "BenchmarkResult",
    "GemmProblem",
    "ProfileResult",
    "StepSpec",
    "add_speedups",
    "benchmark",
    "describe_kernel",
    "default_problem",
    "normalize_arch",
    "normalize_steps",
    "profile_step",
    "profile_suite",
    "run_gemm",
    "run_suite",
    "validate_problem",
]
