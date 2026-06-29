from __future__ import annotations

import importlib

from gemm_cutedsl_modal.exercise_kernels import STEP_KERNELS
from gemm_cutedsl_modal.kernel import ARCH_META, describe_kernel
from gemm_cutedsl_modal.modal_app import _build_payload, select_arches


def test_arch_metadata() -> None:
    assert ARCH_META["h100"]["modal_gpu"] == "H100"
    assert ARCH_META["h100"]["mma"] == "WGMMA"
    assert ARCH_META["b200"]["modal_gpu"] == "B200"
    assert ARCH_META["b200"]["mma"] == "tcgen05"


def test_describe_kernel() -> None:
    h100_step9 = describe_kernel("h100", 9)
    b200_step9 = describe_kernel("b200", 9)
    assert h100_step9["accumulator"] == "register"
    assert b200_step9["accumulator"] == "TMEM"
    assert b200_step9["tile"] == (512, 256)
    assert b200_step9["consumers"] == 2
    assert b200_step9["pipe_depth"] == 4


def test_exercise_kernel_slots() -> None:
    assert set(STEP_KERNELS) == set(range(1, 10))
    assert STEP_KERNELS[1].__name__ == "step1_kernel"
    assert STEP_KERNELS[9].__name__ == "step9_kernel"


def test_modal_helpers_without_modal_runtime() -> None:
    assert select_arches("both") == ("h100", "b200")
    assert select_arches("H100") == ("h100",)
    payload = _build_payload(
        step="all",
        m=0,
        n=0,
        k=0,
        warmup=1,
        iters=2,
        mode="profile",
        profile_rows=11,
        seed=3,
    )
    assert payload["steps"] == tuple(range(1, 10))
    assert payload["m"] is None
    assert payload["iters"] == 2
    assert payload["mode"] == "profile"
    assert payload["profile_rows"] == 11


def test_modal_module_importable() -> None:
    module = importlib.import_module("gemm_cutedsl_modal.modal_app")
    assert module.APP_NAME == "cutedsl-gemm-h100-b200"
