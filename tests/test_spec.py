from __future__ import annotations

import pytest

from gemm_cutedsl.spec import (
    STEP_SPECS,
    correctness_problem,
    default_problem,
    normalize_arch,
    normalize_steps,
    problem_from_dims,
    validate_problem,
)


def test_arch_normalization() -> None:
    assert normalize_arch("H100") == "h100"
    assert normalize_arch(" b200 ") == "b200"
    with pytest.raises(ValueError):
        normalize_arch("a100")


def test_step_normalization() -> None:
    assert normalize_steps("all") == tuple(range(1, 10))
    assert normalize_steps("9") == (9,)
    assert normalize_steps(1) == (1,)
    with pytest.raises(ValueError):
        normalize_steps("10")


def test_step_specs_cover_all_steps() -> None:
    assert set(STEP_SPECS) == set(range(1, 10))
    assert STEP_SPECS[5].pipe_depth == 2
    assert STEP_SPECS[8].cta_group == 2
    assert STEP_SPECS[8].m_multiple == 256
    assert STEP_SPECS[9].consumers == 2
    assert STEP_SPECS[9].warpgroups == 3
    assert STEP_SPECS[9].cluster_tile == (512, 256)


def test_step_shape_rules() -> None:
    assert validate_problem(1, 128, 128, 64).flops == 2 * 128 * 128 * 64
    assert validate_problem(2, 128, 128, 256).output_shape == (128, 128)
    assert validate_problem(3, 256, 384, 256).a_shape == (256, 256)
    assert validate_problem(8, 512, 256, 64).b_shape == (256, 64)
    assert validate_problem(9, 512, 256, 64).output_shape == (512, 256)


@pytest.mark.parametrize(
    ("step", "m", "n", "k", "match"),
    [
        (1, 256, 128, 64, "M=128"),
        (1, 128, 128, 128, "K=64"),
        (3, 130, 128, 64, "multiple of 128"),
        (8, 128, 256, 64, "multiple of 256"),
        (8, 256, 256, 65, "multiple of 64"),
        (9, 256, 256, 64, "multiple of 512"),
        (9, 512, 128, 64, "multiple of 256"),
    ],
)
def test_invalid_shapes(step: int, m: int, n: int, k: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_problem(step, m, n, k)


def test_default_problem_and_overrides() -> None:
    assert default_problem(1).output_shape == (128, 128)
    assert default_problem(8).output_shape == (4096, 4096)
    assert default_problem(9).output_shape == (4096, 4096)
    assert problem_from_dims(3, None, None, None).output_shape == (256, 256)
    assert problem_from_dims(3, 512, 512, 512).a_shape == (512, 512)


def test_correctness_problem_uses_step_specific_shapes() -> None:
    expected = {
        1: (128, 128, 64),
        2: (128, 128, 256),
        3: (256, 256, 128),
        4: (256, 384, 128),
        5: (384, 256, 128),
        6: (384, 384, 128),
        7: (512, 384, 128),
        8: (512, 512, 128),
        9: (512, 256, 128),
    }
    for step, shape in expected.items():
        problem = correctness_problem(step)
        assert (problem.m, problem.n, problem.k) == shape
