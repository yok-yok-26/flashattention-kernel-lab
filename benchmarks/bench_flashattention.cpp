#include "flashattention.h"
#include "cuda_check.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using LaunchFn = cudaError_t (*)(const FlashAttentionParams&, cudaStream_t);

struct ModeSpec {
  const char* name;
  LaunchFn launch;
};

static int arg_int(int argc, char** argv, const char* name, int def) {
  int value = def;
  for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == name) value = std::stoi(argv[i + 1]);
  return value;
}
static std::string arg_str(int argc, char** argv, const char* name, const std::string& def) {
  std::string value = def;
  for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == name) value = argv[i + 1];
  return value;
}

static double attention_tflops(const FlashAttentionShape& shape, float latency_ms) {
  double flops = 4.0 * shape.batch * shape.heads * shape.seq_len * shape.seq_len * shape.head_dim;
  return flops / (latency_ms * 1e-3) / 1e12;
}

static bool benchmark_mode_shape_supported(const ModeSpec& mode, const FlashAttentionShape& shape, const char** reason) {
  std::string mode_name(mode.name);
  if (mode_name == "v1" || mode_name == "v2") {
    if (shape.seq_len != 2048 || shape.head_dim != 128) {
      if (reason) *reason = "only_S2048_D128";
      return false;
    }
  }
  if (reason) *reason = nullptr;
  return true;
}

static void write_user_timing(std::ofstream& out,
                              const ModeSpec& mode,
                              const FlashAttentionShape& shape,
                              const FlashAttentionParams& params,
                              int warmup,
                              int iters) {
  const char* domain_reason = nullptr;
  if (!benchmark_mode_shape_supported(mode, shape, &domain_reason)) {
    out << mode.name << ',' << shape.batch << ',' << shape.heads << ',' << shape.seq_len << ',' << shape.head_dim
        << ",SKIP," << domain_reason << ",,\n";
    return;
  }
  if (!mode.launch) {
    out << mode.name << ',' << shape.batch << ',' << shape.heads << ',' << shape.seq_len << ',' << shape.head_dim
        << ",SKIP,missing_launch_symbol,,\n";
    return;
  }
  cudaError_t first = mode.launch(params, 0);
  if (first == cudaErrorNotSupported) {
    out << mode.name << ',' << shape.batch << ',' << shape.heads << ',' << shape.seq_len << ',' << shape.head_dim
        << ",SKIP,kernel_todo,,\n";
    return;
  }
  CUDA_CHECK(first);
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int i = 0; i < warmup; ++i) CUDA_CHECK(mode.launch(params, 0));
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) CUDA_CHECK(mode.launch(params, 0));
  CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  float latency = ms / iters;
  out << mode.name << ',' << shape.batch << ',' << shape.heads << ',' << shape.seq_len << ',' << shape.head_dim
      << ",PASS,," << latency << ',' << attention_tflops(shape, latency) << "\n";
}

static void write_library_timing(std::ofstream& out,
                                 const char* mode_name,
                                 FlashAttentionLibraryBaseline* baseline,
                                 const FlashAttentionShape& shape,
                                 const FlashAttentionParams& params,
                                 int warmup,
                                 int iters) {
  CUDA_CHECK(launch_flashattention_library_composed_cublas_cudnn(baseline, params, 0));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(launch_flashattention_library_composed_cublas_cudnn(baseline, params, 0));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(launch_flashattention_library_composed_cublas_cudnn(baseline, params, 0));
  }
  CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  float latency = ms / iters;
  out << mode_name << ',' << shape.batch << ',' << shape.heads << ',' << shape.seq_len << ',' << shape.head_dim
      << ",PASS,," << latency << ',' << attention_tflops(shape, latency) << "\n";
}

static LaunchFn find_user_mode(const std::string& mode) {
  if (mode == "v1") return launch_flashattention_v1;
  if (mode == "v2") return launch_flashattention_v2;
  if (mode == "user") return launch_flashattention_user;
  return nullptr;
}

int main(int argc, char** argv) {
  FlashAttentionShape shape{arg_int(argc, argv, "--batch", 1), arg_int(argc, argv, "--heads", 4), arg_int(argc, argv, "--seq", 128), arg_int(argc, argv, "--dim", 64)};
  int warmup = arg_int(argc, argv, "--warmup", 10);
  int iters = arg_int(argc, argv, "--iters", 100);
  bool single = arg_int(argc, argv, "--single", 0) != 0;
  std::string mode = arg_str(argc, argv, "--mode", "all");
  std::string csv = arg_str(argc, argv, "--csv", "reports/benchmark/latest.csv");
  std::string reason;
  if (!flashattention_shape_valid(shape, &reason)) { std::cerr << "SKIP " << reason << "\n"; return 0; }
  size_t n = flashattention_numel(shape);
  std::vector<float> h(n);
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : h) x = dist(rng);
  float *dq=nullptr, *dk=nullptr, *dv=nullptr, *do_=nullptr;
  CUDA_CHECK(cudaMalloc(&dq, n * sizeof(float))); CUDA_CHECK(cudaMalloc(&dk, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dv, n * sizeof(float))); CUDA_CHECK(cudaMalloc(&do_, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dq, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  FlashAttentionParams params{dq, dk, dv, do_, shape, 1.0f / std::sqrt(static_cast<float>(shape.head_dim))};
  FlashAttentionLibraryBaseline* baseline_tf32 = flashattention_library_baseline_create(
      shape, FlashAttentionBaselineMath::kDefaultTf32);
  FlashAttentionLibraryBaseline* baseline_fp32 = flashattention_library_baseline_create(
      shape, FlashAttentionBaselineMath::kFp32Pedantic);

  if (single) {
    if (mode == "library_composed_cublas_cudnn" || mode == "library_composed_cublas_cudnn_tf32") {
      CUDA_CHECK(launch_flashattention_library_composed_cublas_cudnn(baseline_tf32, params, 0));
    } else if (mode == "library_composed_cublas_cudnn_fp32_pedantic") {
      CUDA_CHECK(launch_flashattention_library_composed_cublas_cudnn(baseline_fp32, params, 0));
    } else {
      LaunchFn launch = find_user_mode(mode);
      if (!launch) {
        std::cout << "SKIP: missing launch symbol or unknown mode " << mode << "\n";
      } else {
        cudaError_t status = launch(params, 0);
        if (status == cudaErrorNotSupported) std::cout << "SKIP: kernel TODO for mode " << mode << "\n";
        else CUDA_CHECK(status);
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    flashattention_library_baseline_destroy(baseline_tf32);
    flashattention_library_baseline_destroy(baseline_fp32);
    CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk)); CUDA_CHECK(cudaFree(dv)); CUDA_CHECK(cudaFree(do_));
    return 0;
  }

  std::ofstream out(csv);
  out << "mode,batch,heads,seq,dim,status,skip_reason,latency_ms,tflops\n";
  if (mode == "all" || mode == "library_composed_cublas_cudnn" || mode == "library_composed_cublas_cudnn_tf32") {
    write_library_timing(out, flashattention_baseline_math_name(FlashAttentionBaselineMath::kDefaultTf32),
                         baseline_tf32, shape, params, warmup, iters);
  }
  if (mode == "all" || mode == "library_composed_cublas_cudnn_fp32_pedantic") {
    write_library_timing(out, flashattention_baseline_math_name(FlashAttentionBaselineMath::kFp32Pedantic),
                         baseline_fp32, shape, params, warmup, iters);
  }
  std::vector<ModeSpec> user_modes = {
    {"v1", launch_flashattention_v1},
    {"v2", launch_flashattention_v2},
    {"user", launch_flashattention_user},
  };
  for (const auto& spec : user_modes) {
    if (mode == "all" || mode == spec.name) write_user_timing(out, spec, shape, params, warmup, iters);
  }
  out.close();

  flashattention_library_baseline_destroy(baseline_tf32);
  flashattention_library_baseline_destroy(baseline_fp32);
  CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk)); CUDA_CHECK(cudaFree(dv)); CUDA_CHECK(cudaFree(do_));
  std::cout << "wrote " << csv << "\n";
  return 0;
}
