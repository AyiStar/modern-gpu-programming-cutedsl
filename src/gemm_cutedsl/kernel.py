"""Modal-side GEMM dispatch and H100/B200 metadata."""

from __future__ import annotations

from .kernels.common import cute_modules
from .kernels.registry import get_step_kernel
from .spec import ARCHES, STEP_SPECS, GemmProblem, normalize_arch, validate_problem

ARCH_META = {
    "h100": {
        "modal_gpu": "H100",
        "cute_arch": "sm90a",
        "mma": "WGMMA",
        "accumulator": "register",
        "step8_note": "Hopper-compatible 2-CTA cluster exercise.",
        "step9_note": "Hopper-compatible multi-consumer cluster exercise.",
    },
    "b200": {
        "modal_gpu": "B200",
        "cute_arch": "sm100a",
        "mma": "tcgen05",
        "accumulator": "TMEM",
        "step8_note": "Blackwell CTA_GROUP=2 cluster exercise.",
        "step9_note": "Blackwell NUM_CONSUMER=2 multi-consumer exercise.",
    },
}


def describe_kernel(arch: str, step: int) -> dict[str, object]:
    arch = normalize_arch(arch)
    if step not in STEP_SPECS:
        raise ValueError(f"invalid step {step!r}; expected 1-9")
    spec = STEP_SPECS[step]
    meta = ARCH_META[arch]
    note = spec.summary
    if step == 8:
        note = str(meta["step8_note"])
    elif step == 9:
        note = str(meta["step9_note"])
    return {
        "arch": arch,
        "modal_gpu": meta["modal_gpu"],
        "cute_arch": meta["cute_arch"],
        "mma": meta["mma"],
        "accumulator": meta["accumulator"],
        "step": step,
        "name": spec.name,
        "tile": spec.cluster_tile,
        "pipe_depth": spec.pipe_depth,
        "cta_group": spec.cta_group,
        "consumers": spec.consumers,
        "warpgroups": spec.warpgroups,
        "note": note,
    }


def validate_tensors(step: int, a: object, b: object) -> GemmProblem:
    import torch

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("A and B must be torch.Tensor instances")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"A and B must be rank-2 tensors; got {a.ndim=} and {b.ndim=}")
    if a.dtype is not torch.float16 or b.dtype is not torch.float16:
        raise TypeError(f"A and B must be torch.float16; got {a.dtype=} and {b.dtype=}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("A and B must be CUDA tensors; run this through Modal")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("A and B must be contiguous tensors")
    if a.shape[1] != b.shape[1]:
        msg = f"A[M,K] and B[N,K] must share K; got {a.shape=} and {b.shape=}"
        raise ValueError(msg)
    return validate_problem(step, int(a.shape[0]), int(b.shape[0]), int(a.shape[1]))


def reference_gemm(a: object, b: object) -> object:
    import torch

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("A and B must be torch.Tensor instances")
    return torch.matmul(a.float(), b.float().T).half()


def run_gemm(arch: str, step: int, a: object, b: object) -> object:
    """Run the CuTe DSL GEMM kernel on a Modal GPU.

    Step 1-9 differences are expressed as shape rules and architecture metadata.
    The current kernel body is a correctness-first CuTe DSL implementation,
    intentionally kept small for exercises and future replacement by optimized
    WGMMA/tcgen05/TMA bodies.
    """

    arch = normalize_arch(arch)
    if arch not in ARCHES:
        raise ValueError(f"unsupported arch {arch!r}")
    problem = validate_tensors(step, a, b)

    import torch

    _cutlass, _cute, from_dlpack, _Float32, _Int32 = cute_modules()
    c = torch.empty(problem.output_shape, device=a.device, dtype=torch.float16)
    launch_gemm = get_step_kernel(arch, step)
    launch_gemm(
        from_dlpack(a, assumed_align=16),
        from_dlpack(b, assumed_align=16),
        from_dlpack(c, assumed_align=16),
        problem.m,
        problem.n,
        problem.k,
    )
    return c
