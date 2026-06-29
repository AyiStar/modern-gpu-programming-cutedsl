"""Modal entrypoint for H100 and B200 GEMM exercises."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from gemm_cutedsl_modal.spec import normalize_arch, normalize_steps

try:
    import modal
except Exception:  # pragma: no cover - local tests do not require Modal
    modal = None  # type: ignore[assignment]


APP_NAME = "cutedsl-gemm-h100-b200"
CUDA_IMAGE = "nvidia/cuda:12.9.1-devel-ubuntu24.04"
PYTHON_VERSION = "3.12"


def _gpu_arg_from_argv(default: str = "h100") -> str:
    """Read --gpu early so Modal only registers the requested GPU functions."""

    for index, arg in enumerate(sys.argv):
        if arg == "--gpu" and index + 1 < len(sys.argv):
            return sys.argv[index + 1].lower().strip()
        if arg.startswith("--gpu="):
            return arg.split("=", 1)[1].lower().strip()
    return default


def select_arches(gpu: str) -> tuple[str, ...]:
    value = gpu.lower().strip()
    if value == "both":
        return ("h100", "b200")
    return (normalize_arch(value),)


def _build_payload(
    *,
    step: str,
    m: int,
    n: int,
    k: int,
    warmup: int,
    iters: int,
    mode: str,
    profile_rows: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "steps": normalize_steps(step),
        "m": None if m <= 0 else m,
        "n": None if n <= 0 else n,
        "k": None if k <= 0 else k,
        "warmup": warmup,
        "iters": iters,
        "mode": mode.lower().strip(),
        "profile_rows": profile_rows,
        "seed": seed,
    }


def _result_json(results: list[dict[str, Any]]) -> str:
    return json.dumps(results, indent=2, sort_keys=True)


if modal is not None:
    REGISTERED_ARCHES = select_arches(_gpu_arg_from_argv())
    image = (
        modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON_VERSION)
        .apt_install("build-essential", "git")
        .pip_install(
            "numpy>=1.26",
            "torch>=2.6",
            "nvidia-cutlass-dsl>=4.4.0",
        )
        .add_local_python_source("gemm_cutedsl_modal")
    )
    app = modal.App(APP_NAME, image=image)

    def _run_arch(arch: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        from gemm_cutedsl_modal.bench import run_suite
        from gemm_cutedsl_modal.profiling import add_speedups, profile_suite

        mode = payload["mode"]
        if mode == "benchmark":
            benchmark_results = run_suite(
                arch,
                tuple(payload["steps"]),
                m=payload["m"],
                n=payload["n"],
                k=payload["k"],
                warmup=payload["warmup"],
                iters=payload["iters"],
                seed=payload["seed"],
            )
            return add_speedups(benchmark_results)
        if mode == "profile":
            return [
                result.to_dict()
                for result in profile_suite(
                    arch,
                    tuple(payload["steps"]),
                    m=payload["m"],
                    n=payload["n"],
                    k=payload["k"],
                    warmup=payload["warmup"],
                    active=payload["iters"],
                    row_limit=payload["profile_rows"],
                    seed=payload["seed"],
                )
            ]
        if mode == "both":
            benchmark_results = run_suite(
                arch,
                tuple(payload["steps"]),
                m=payload["m"],
                n=payload["n"],
                k=payload["k"],
                warmup=payload["warmup"],
                iters=payload["iters"],
                seed=payload["seed"],
            )
            profiles = [
                result.to_dict()
                for result in profile_suite(
                    arch,
                    tuple(payload["steps"]),
                    m=payload["m"],
                    n=payload["n"],
                    k=payload["k"],
                    warmup=payload["warmup"],
                    active=payload["iters"],
                    row_limit=payload["profile_rows"],
                    seed=payload["seed"],
                )
            ]
            return [{"benchmark": add_speedups(benchmark_results), "profile": profiles}]
        raise ValueError("mode must be one of: benchmark, profile, both")

    if "h100" in REGISTERED_ARCHES:

        @app.function(gpu="H100", timeout=1200)
        def run_h100(payload: dict[str, Any]) -> list[dict[str, Any]]:
            return _run_arch("h100", payload)

    if "b200" in REGISTERED_ARCHES:

        @app.function(gpu="B200", timeout=1200)
        def run_b200(payload: dict[str, Any]) -> list[dict[str, Any]]:
            return _run_arch("b200", payload)

    @app.local_entrypoint()
    def main(
        gpu: str = "h100",
        step: str = "9",
        m: int = 0,
        n: int = 0,
        k: int = 0,
        warmup: int = 5,
        iters: int = 20,
        mode: str = "benchmark",
        profile_rows: int = 20,
        seed: int = 0,
    ) -> None:
        payload = _build_payload(
            step=step,
            m=m,
            n=n,
            k=k,
            warmup=warmup,
            iters=iters,
            mode=mode,
            profile_rows=profile_rows,
            seed=seed,
        )
        all_results: list[dict[str, Any]] = []
        for arch in select_arches(gpu):
            if arch == "h100":
                fn = globals().get("run_h100")
                if fn is None:
                    raise ValueError("H100 function was not registered; rerun with --gpu h100")
                all_results.extend(fn.remote(payload))
            elif arch == "b200":
                fn = globals().get("run_b200")
                if fn is None:
                    raise ValueError("B200 function was not registered; rerun with --gpu b200")
                all_results.extend(fn.remote(payload))
            else:  # pragma: no cover - normalize_arch prevents this branch
                raise ValueError(f"unsupported arch: {arch}")
        print(_result_json(all_results))
else:
    app = None
