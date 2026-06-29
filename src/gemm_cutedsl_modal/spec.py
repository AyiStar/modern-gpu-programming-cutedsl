"""Small, explicit Step 1-9 problem spec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Arch = Literal["h100", "b200"]

ARCHES: tuple[Arch, ...] = ("h100", "b200")
STEPS: tuple[int, ...] = tuple(range(1, 10))
BLK_M = 128
BLK_N = 128
BLK_K = 64


@dataclass(frozen=True)
class StepSpec:
    step: int
    name: str
    summary: str
    m_multiple: int
    n_multiple: int
    k_multiple: int
    exact_m: int | None = None
    exact_n: int | None = None
    exact_k: int | None = None
    pipe_depth: int = 1
    cta_group: int = 1
    consumers: int = 1
    warpgroups: int = 1
    uses_tma: bool = False
    persistent: bool = False
    warp_specialized: bool = False

    @property
    def cluster_tile(self) -> tuple[int, int]:
        return (self.m_multiple, self.n_multiple)


STEP_SPECS: dict[int, StepSpec] = {
    1: StepSpec(
        1,
        "single_tile",
        "Single 128x128 output tile with K=64.",
        BLK_M,
        BLK_N,
        BLK_K,
        exact_m=128,
        exact_n=128,
        exact_k=64,
        warpgroups=1,
    ),
    2: StepSpec(
        2,
        "single_tile_k_loop",
        "Single 128x128 output tile with a K loop.",
        BLK_M,
        BLK_N,
        BLK_K,
        exact_m=128,
        exact_n=128,
        warpgroups=1,
    ),
    3: StepSpec(
        3,
        "cta_grid",
        "2D CTA grid, one CTA per 128x128 output tile.",
        BLK_M,
        BLK_N,
        BLK_K,
        warpgroups=1,
    ),
    4: StepSpec(
        4,
        "tma_async",
        "TMA async load/store boundary with mbarrier synchronization.",
        BLK_M,
        BLK_N,
        BLK_K,
        uses_tma=True,
        warpgroups=1,
    ),
    5: StepSpec(
        5,
        "double_buffer",
        "Two-stage SMEM ring buffer pipeline.",
        BLK_M,
        BLK_N,
        BLK_K,
        pipe_depth=2,
        uses_tma=True,
        warpgroups=1,
    ),
    6: StepSpec(
        6,
        "persistent",
        "Persistent CTA scheduler driven by runtime SM count.",
        BLK_M,
        BLK_N,
        BLK_K,
        pipe_depth=2,
        uses_tma=True,
        persistent=True,
        warpgroups=1,
    ),
    7: StepSpec(
        7,
        "warp_specialized",
        "Producer, MMA consumer, and writeback roles.",
        BLK_M,
        BLK_N,
        BLK_K,
        pipe_depth=2,
        uses_tma=True,
        persistent=True,
        warp_specialized=True,
        warpgroups=2,
    ),
    8: StepSpec(
        8,
        "cluster_2cta",
        "2-CTA cluster exercise with a 256x256 cluster tile.",
        256,
        256,
        BLK_K,
        pipe_depth=4,
        cta_group=2,
        warpgroups=2,
        uses_tma=True,
        persistent=True,
        warp_specialized=True,
    ),
    9: StepSpec(
        9,
        "multi_consumer",
        "Two MMA consumers share one B tile; cluster tile is 512x256.",
        512,
        256,
        BLK_K,
        pipe_depth=4,
        cta_group=2,
        consumers=2,
        warpgroups=3,
        uses_tma=True,
        persistent=True,
        warp_specialized=True,
    ),
}


@dataclass(frozen=True)
class GemmProblem:
    step: int
    m: int
    n: int
    k: int

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.m, self.n)

    @property
    def a_shape(self) -> tuple[int, int]:
        return (self.m, self.k)

    @property
    def b_shape(self) -> tuple[int, int]:
        return (self.n, self.k)


def normalize_arch(arch: str) -> Arch:
    normalized = arch.lower().strip()
    if normalized not in ARCHES:
        msg = f"unsupported GPU arch {arch!r}; expected one of {ARCHES}"
        raise ValueError(msg)
    return normalized  # type: ignore[return-value]


def normalize_steps(step: int | str) -> tuple[int, ...]:
    if isinstance(step, str):
        value = step.lower().strip()
        if value == "all":
            return STEPS
        try:
            step = int(value)
        except ValueError as exc:
            msg = f"invalid step {step!r}; expected 1-9 or 'all'"
            raise ValueError(msg) from exc
    if step not in STEP_SPECS:
        msg = f"invalid step {step!r}; expected 1-9"
        raise ValueError(msg)
    return (int(step),)


def validate_problem(step: int, m: int, n: int, k: int) -> GemmProblem:
    if step not in STEP_SPECS:
        msg = f"invalid step {step!r}; expected 1-9"
        raise ValueError(msg)
    if min(m, n, k) <= 0:
        msg = f"M, N, and K must be positive; got M={m}, N={n}, K={k}"
        raise ValueError(msg)

    spec = STEP_SPECS[step]
    exacts = (("M", m, spec.exact_m), ("N", n, spec.exact_n), ("K", k, spec.exact_k))
    for name, value, exact in exacts:
        if exact is not None and value != exact:
            msg = f"Step {step} requires {name}={exact}; got {value}"
            raise ValueError(msg)

    multiples = (
        ("M", m, spec.m_multiple),
        ("N", n, spec.n_multiple),
        ("K", k, spec.k_multiple),
    )
    for name, value, multiple in multiples:
        if value % multiple != 0:
            msg = f"Step {step} requires {name} to be a multiple of {multiple}; got {value}"
            raise ValueError(msg)
    return GemmProblem(step=step, m=m, n=n, k=k)


def default_problem(step: int) -> GemmProblem:
    if step == 1:
        return validate_problem(step, 128, 128, 64)
    if step == 2:
        return validate_problem(step, 128, 128, 256)
    if step == 3:
        return validate_problem(step, 256, 256, 256)
    if step in (4, 5, 6, 7):
        return validate_problem(step, 4096, 4096, 4096)
    if step in (8, 9):
        return validate_problem(step, 4096, 4096, 4096)
    msg = f"invalid step {step!r}; expected 1-9"
    raise ValueError(msg)


def problem_from_dims(step: int, m: int | None, n: int | None, k: int | None) -> GemmProblem:
    default = default_problem(step)
    return validate_problem(
        step,
        default.m if m is None or m <= 0 else m,
        default.n if n is None or n <= 0 else n,
        default.k if k is None or k <= 0 else k,
    )
