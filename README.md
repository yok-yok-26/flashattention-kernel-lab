# FlashAttention Kernel Lab

CUDA learning project for implementing and benchmarking FlashAttention forward kernels on NVIDIA GPUs. The lab keeps the handwritten CUDA kernels separate from the surrounding engineering harness: CMake builds, CPU/library references, correctness tests, sanitizer scripts, profiler scripts, and benchmark CSV output.

## Contents

- `kernels/`: handwritten CUDA kernel variants.
- `include/`: public parameter structs, launch declarations, and CUDA/library check helpers.
- `reference/`: CPU reference and cuBLAS/cuDNN library-composed baselines.
- `tests/`: deterministic correctness harness.
- `benchmarks/`: release benchmark harness.
- `scripts/`: build, test, benchmark, sanitizer, and profiler entrypoints.
- `reports/benchmark/`: selected public benchmark CSVs.

## Requirements

- Linux with an NVIDIA GPU.
- CUDA Toolkit 12.8 or newer with `nvcc`.
- cuBLAS and cuDNN development libraries.
- CMake 3.24 or newer.
- Ninja.

The original benchmark environment used an NVIDIA GeForce RTX 5070 with compute capability 12.0 and CUDA 12.8.

## Build

Debug build for correctness and sanitizer work:

```bash
./scripts/build_debug.sh
```

Release build for benchmark and profiling work:

```bash
./scripts/build_release.sh
```

The helper scripts default to `CUDA_HOME=/usr/local/cuda-12.8`. Override `CUDA_HOME` before running the scripts if your CUDA toolkit is installed elsewhere.

## Verify

Run the correctness harness after a debug build:

```bash
./scripts/run_correctness.sh
```

Optional sanitizer entrypoints:

```bash
./scripts/run_memcheck.sh
./scripts/run_synccheck.sh
./scripts/run_racecheck.sh
```

`racecheck` can be very slow on the large FlashAttention test cases.

## Benchmark

Build release first, then run:

```bash
./scripts/run_benchmark.sh --mode all --batch 1 --heads 1 --seq 2048 --dim 128 --iters 20 --warmup 5
```

Public benchmark CSVs included in this repository:

- `reports/benchmark/baseline_precision_split_bh_2048x128.csv`
- `reports/benchmark/bh_sweep_2048x128_comparison_latest.csv`
- `reports/benchmark/bh_sweep_2048x128_latest.csv`

## Baselines and Precision

The main library-composed baseline computes:

1. `QK^T` with cuBLAS.
2. Row-wise softmax with cuDNN.
3. `PV` with cuBLAS.

Two explicit cuBLAS math modes are available:

- `library_composed_cublas_cudnn_tf32`: TF32 tensor-op math where available. This represents a fast generic library path.
- `library_composed_cublas_cudnn_fp32_pedantic`: pedantic FP32 math. This is the fair precision baseline for the current scalar FP32 user kernels.

The legacy mode name `library_composed_cublas_cudnn` is kept as a compatibility alias for the TF32 path.

## Kernel Contract

Current tensor contract:

- Inputs: contiguous row-major `float32` `Q`, `K`, and `V`, shape `[B, H, S, D]`.
- Output: contiguous row-major `float32` `O`, shape `[B, H, S, D]`.
- Math: `O[b,h,i,:] = softmax(Q[b,h,i,:] @ K[b,h,:,:]^T / sqrt(D)) @ V[b,h,:,:]`.
- Current optimized variants target `S=2048,D=128`.

## License

Apache License 2.0.
