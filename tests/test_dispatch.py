from __future__ import annotations

import importlib

import pytest

from gemm_cutedsl.kernel import ARCH_META, describe_kernel
from gemm_cutedsl.kernels.registry import STEP_KERNELS, get_step_kernel
from gemm_cutedsl.modal_app import _build_payload, select_arches


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
    assert set(STEP_KERNELS) == {"h100", "b200"}
    assert set(STEP_KERNELS["h100"]) == set(range(1, 10))
    assert set(STEP_KERNELS["b200"]) == set(range(1, 10))
    assert STEP_KERNELS["h100"][1].__name__ == "build_kernel"
    assert STEP_KERNELS["b200"][9].__name__ == "build_kernel"


def test_step_kernel_files_are_importable() -> None:
    for arch in ("h100", "b200"):
        for step in range(1, 10):
            module = importlib.import_module(f"gemm_cutedsl.kernels.{arch}.step{step}")
            assert callable(module.build_kernel)


def test_kernel_registry_validation() -> None:
    with pytest.raises(ValueError, match="unsupported GPU arch"):
        get_step_kernel("a100", 1)
    with pytest.raises(ValueError, match="expected 1-9"):
        get_step_kernel("h100", 10)


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
    module = importlib.import_module("gemm_cutedsl.modal_app")
    assert module.APP_NAME == "cutedsl-gemm-h100-b200"
