# H100 + B200 Modal CuTe DSL GEMM Exercises

This project builds a Modal-only GEMM practice environment for the MLC Modern
GPU Programming GEMM chapters.  The kernel computes:

```text
D = A @ B.T
```

where `A` is `[M, K]`, `B` is `[N, K]`, inputs are `float16`, accumulation is
`float32`, and output is `float16`.

The package exposes separate H100 and B200 backends and Step 1-9 dispatch.  The
step constraints intentionally match the tutorial progression: early steps are
small correctness exercises; later steps use larger tiled shapes.

Implementation note: execution happens on Modal H100/B200 functions only.  The
current Step 1-9 functions route to a small correctness-first CuTe DSL kernel
where one CUDA thread computes one output element.

## Step Map

| Step | Exercise | Shape rule |
| ---: | --- | --- |
| 1 | Single 128x128 tile, K=64 | `M=N=128`, `K=64` |
| 2 | Single output tile with K loop | `M=N=128`, `K % 64 == 0` |
| 3 | 2D CTA grid, one CTA per 128x128 output tile | `M,N % 128 == 0`, `K % 64 == 0` |
| 4 | TMA async copy boundary | Same as Step 3 |
| 5 | `PIPE_DEPTH=2` staged pipeline | Same as Step 3 |
| 6 | Persistent CTA scheduling | Same as Step 3 |
| 7 | Warp specialization roles | Same as Step 3 |
| 8 | 2-CTA cluster exercise | `M,N % 256 == 0`, `K % 64 == 0` |
| 9 | Multi-consumer cluster exercise | `M % 512 == 0`, `N % 256 == 0`, `K % 64 == 0` |

## Local Setup

Use `uv` for all Python commands:

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest -q
```

The local tests do not require a GPU or CuTe DSL.  They validate shape rules,
Step 1-9 metadata, and Modal argument handling.  Kernel execution is Modal-only.

## Where To Write Kernels

Write H100 exercise kernels in:

```text
src/gemm_cutedsl/kernels/h100/step1.py
src/gemm_cutedsl/kernels/h100/step2.py
...
src/gemm_cutedsl/kernels/h100/step9.py
```

Write B200 exercise kernels in:

```text
src/gemm_cutedsl/kernels/b200/step1.py
src/gemm_cutedsl/kernels/b200/step2.py
...
src/gemm_cutedsl/kernels/b200/step9.py
```

Each step file exposes `build_kernel()` and currently returns the shared
`correctness_launcher()` from `kernels/correctness.py`.  Replace one step file at
a time with your CuTe DSL implementation while following the tutorial.  For
example, H100 Step 4 lives in `kernels/h100/step4.py`, while B200 Step 8 lives in
`kernels/b200/step8.py`.

Do not edit `modal_app.py`, `bench.py`, or `profiling.py` for normal kernel
exercises; those files already handle Modal launch, correctness checks,
performance timing, and profiler summaries.

The Step 1-9 shape rules and tutorial metadata live in `spec.py`.  The H100/B200
metadata shown in benchmark output lives in `kernel.py`, and architecture/step
dispatch lives in `kernels/registry.py`.

## Modal Setup

Log in to Modal once:

```bash
modal setup
```

Run one H100 correctness check:

```bash
uv run modal run src/gemm_cutedsl/modal_app.py \
  --gpu h100 --step 1 --m 128 --n 128 --k 64 --mode correctness
```

Run per-step correctness across all steps.  When `M/N/K` are omitted in
`--mode correctness`, each step uses its own small legal smoke-test shape:
Step 1 `128x128x64`, Step 2 `128x128x256`, Step 3 `256x256x128`,
Step 4 `256x384x128`, Step 5 `384x256x128`, Step 6 `384x384x128`,
Step 7 `512x384x128`, Step 8 `512x512x128`, and Step 9 `512x256x128`.

```bash
uv run modal run src/gemm_cutedsl/modal_app.py \
  --gpu h100 --step all --mode correctness
```

Run B200 Step 9:

```bash
uv run modal run src/gemm_cutedsl/modal_app.py --gpu b200 --step 9 --m 4096 --n 4096 --k 4096
```

Run smoke checks across both GPUs and all steps:

```bash
uv run modal run src/gemm_cutedsl/modal_app.py --gpu both --step all --iters 3 --warmup 1
```

## Performance And Profiling

Benchmark one step on B200:

```bash
uv run modal run src/gemm_cutedsl/modal_app.py \
  --gpu b200 --step 9 --m 4096 --n 4096 --k 4096 \
  --mode benchmark --warmup 10 --iters 50
```

Benchmark all steps on one GPU.  The JSON output includes `latency_ms`,
`tflops`, correctness error, kernel metadata, and `speedup_vs_first`:

```bash
uv run modal run src/gemm_cutedsl/modal_app.py \
  --gpu b200 --step all --mode benchmark --warmup 5 --iters 20
```

Collect a torch profiler summary for one step:

```bash
uv run modal run src/gemm_cutedsl/modal_app.py \
  --gpu b200 --step 9 --m 4096 --n 4096 --k 4096 \
  --mode profile --warmup 5 --iters 5 --profile-rows 30
```

Run both benchmark and profiler in one Modal call:

```bash
uv run modal run src/gemm_cutedsl/modal_app.py \
  --gpu h100 --step 7 --mode both --warmup 5 --iters 5
```
