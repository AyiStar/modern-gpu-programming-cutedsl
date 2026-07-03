"""Shared CuTe DSL imports for exercise kernels."""

from functools import cache


@cache
def cute_modules():
    try:
        import cutlass  # type: ignore[import-not-found]
        import cutlass.cute as cute  # type: ignore[import-not-found]
        from cutlass.cute.runtime import from_dlpack  # type: ignore[import-not-found]
        from cutlass.cutlass_dsl import Float32, Int32  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "CuTe DSL is required. Run through Modal so the image installs "
            "nvidia-cutlass-dsl, torch, and CUDA."
        ) from exc
    return cutlass, cute, from_dlpack, Float32, Int32

