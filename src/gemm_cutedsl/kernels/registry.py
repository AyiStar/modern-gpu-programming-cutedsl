"""Architecture-aware Step 1-9 kernel registry."""

from ..spec import normalize_arch
from .b200 import step1 as b200_step1
from .b200 import step2 as b200_step2
from .b200 import step3 as b200_step3
from .b200 import step4 as b200_step4
from .b200 import step5 as b200_step5
from .b200 import step6 as b200_step6
from .b200 import step7 as b200_step7
from .b200 import step8 as b200_step8
from .b200 import step9 as b200_step9
from .h100 import step1 as h100_step1
from .h100 import step2 as h100_step2
from .h100 import step3 as h100_step3
from .h100 import step4 as h100_step4
from .h100 import step5 as h100_step5
from .h100 import step6 as h100_step6
from .h100 import step7 as h100_step7
from .h100 import step8 as h100_step8
from .h100 import step9 as h100_step9

STEP_KERNELS = {
    "h100": {
        1: h100_step1.build_kernel,
        2: h100_step2.build_kernel,
        3: h100_step3.build_kernel,
        4: h100_step4.build_kernel,
        5: h100_step5.build_kernel,
        6: h100_step6.build_kernel,
        7: h100_step7.build_kernel,
        8: h100_step8.build_kernel,
        9: h100_step9.build_kernel,
    },
    "b200": {
        1: b200_step1.build_kernel,
        2: b200_step2.build_kernel,
        3: b200_step3.build_kernel,
        4: b200_step4.build_kernel,
        5: b200_step5.build_kernel,
        6: b200_step6.build_kernel,
        7: b200_step7.build_kernel,
        8: b200_step8.build_kernel,
        9: b200_step9.build_kernel,
    },
}


def get_step_kernel(arch: str, step: int):
    arch = normalize_arch(arch)
    try:
        return STEP_KERNELS[arch][step]()
    except KeyError as exc:
        raise ValueError(f"invalid step {step!r}; expected 1-9") from exc
