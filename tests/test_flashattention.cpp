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

static void fill_inputs(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : q) x = dist(rng);
  for (auto& x : k) x = dist(rng);
  for (auto& x : v) x = dist(rng);
}

static bool check_result(std::ofstream& log,
                         const char* mode,
                         const FlashAttentionShape& shape,
                         const std::vector<float>& got,
                         const std::vector<float>& ref,
                         float atol,
                         float rtol) {
  double max_abs = 0.0, max_rel = 0.0;
  size_t bad = 0, worst = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double abs_err = std::abs(static_cast<double>(got[i]) - ref[i]);
    double rel_err = abs_err / std::max(1e-6, std::abs(static_cast<double>(ref[i])));
    if (abs_err > max_abs) { max_abs = abs_err; worst = i; }
    max_rel = std::max(max_rel, rel_err);
    if (abs_err > atol + rtol * std::abs(ref[i])) ++bad;
  }
  log << (bad ? "FAIL " : "PASS ") << mode << " " << flashattention_shape_string(shape)
      << " max_abs=" << max_abs << " max_rel=" << max_rel
      << " bad_count=" << bad << " worst_idx=" << worst << "\n";
  return bad == 0;
}

static bool mode_shape_supported(const ModeSpec& mode, const FlashAttentionShape& shape, const char** reason) {
  std::string mode_name(mode.name);
  if (mode_name == "v1" || mode_name == "v2") {
    if (shape.seq_len != 2048 || shape.head_dim != 128) {
      if (reason) *reason = "mode currently supports only S=2048,D=128";
      return false;
    }
  }
  if (reason) *reason = nullptr;
  return true;
}

static bool run_mode(std::ofstream& log,
                     const ModeSpec& mode,
                     const FlashAttentionParams& params,
                     const std::vector<float>& ref,
                     std::vector<float>& out,
                     float atol,
                     float rtol) {
  const char* domain_reason = nullptr;
  if (!mode_shape_supported(mode, params.shape, &domain_reason)) {
    log << "SKIP " << mode.name << " " << flashattention_shape_string(params.shape)
        << " reason=" << domain_reason << "\n";
    return true;
  }
  if (!mode.launch) {
    log << "SKIP " << mode.name << " " << flashattention_shape_string(params.shape)
        << " reason=missing launch symbol\n";
    return true;
  }
  CUDA_CHECK(cudaMemset(params.out, 0, ref.size() * sizeof(float)));
  cudaError_t launch_status = mode.launch(params, 0);
  if (launch_status == cudaErrorNotSupported) {
    log << "SKIP " << mode.name << " " << flashattention_shape_string(params.shape)
        << " reason=kernel TODO\n";
    return true;
  }
  CUDA_CHECK(launch_status);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(out.data(), params.out, ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return check_result(log, mode.name, params.shape, out, ref, atol, rtol);
}

static bool run_library_reference_case(std::ofstream& log,
                                       const ModeSpec& mode,
                                       const FlashAttentionShape& shape,
                                       int seed,
                                       float atol,
                                       float rtol) {
  const size_t n = flashattention_numel(shape);
  std::vector<float> hq(n), hk(n), hv(n), href(n), hout(n, 0.0f);
  fill_inputs(hq, hk, hv, seed);

  float *dq=nullptr, *dk=nullptr, *dv=nullptr, *do_=nullptr;
  CUDA_CHECK(cudaMalloc(&dq, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dk, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dv, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&do_, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dq, hq.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, hk.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, hv.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
  FlashAttentionParams params{dq, dk, dv, do_, shape, scale};
  FlashAttentionLibraryBaseline* baseline = flashattention_library_baseline_create(
      shape, FlashAttentionBaselineMath::kFp32Pedantic);

  CUDA_CHECK(cudaMemset(do_, 0, n * sizeof(float)));
  CUDA_CHECK(launch_flashattention_library_composed_cublas_cudnn(baseline, params, 0));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(href.data(), do_, n * sizeof(float), cudaMemcpyDeviceToHost));
  flashattention_library_baseline_destroy(baseline);

  CUDA_CHECK(cudaMemset(do_, 0, n * sizeof(float)));
  cudaError_t launch_status = mode.launch(params, 0);
  if (launch_status == cudaErrorNotSupported) {
    log << "SKIP " << mode.name << " " << flashattention_shape_string(shape) << " reason=kernel TODO\n";
    CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk)); CUDA_CHECK(cudaFree(dv)); CUDA_CHECK(cudaFree(do_));
    return true;
  }
  CUDA_CHECK(launch_status);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hout.data(), do_, n * sizeof(float), cudaMemcpyDeviceToHost));
  std::string case_name = std::string(mode.name) + "_library_ref";
  bool ok = check_result(log, case_name.c_str(), shape, hout, href, atol, rtol);

  CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk)); CUDA_CHECK(cudaFree(dv)); CUDA_CHECK(cudaFree(do_));
  return ok;
}

int main(int argc, char** argv) {
  std::string log_path = "reports/correctness/latest.log";
  for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == "--log") log_path = argv[++i];
  std::ofstream log(log_path);
  std::vector<FlashAttentionShape> cases = {{1,1,1,8}, {1,1,2,16}, {1,2,7,32}, {2,2,17,32}, {1,4,33,64}};
  std::vector<ModeSpec> user_modes = {
    {"v1", launch_flashattention_v1},
    {"v2", launch_flashattention_v2},
    {"user", launch_flashattention_user},
  };
  const float atol = 2e-3f;
  const float rtol = 2e-3f;
  bool failed = false;
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    FlashAttentionShape shape = cases[ci];
    std::string reason;
    if (!flashattention_shape_valid(shape, &reason)) {
      log << "SKIP " << flashattention_shape_string(shape) << " reason=" << reason << "\n";
      continue;
    }
    size_t n = flashattention_numel(shape);
    std::vector<float> hq(n), hk(n), hv(n), href(n), hout(n, 0.0f);
    fill_inputs(hq, hk, hv, 1234 + static_cast<int>(ci));
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    flashattention_reference_cpu(hq, hk, hv, href, shape, scale);
    float *dq=nullptr, *dk=nullptr, *dv=nullptr, *do_=nullptr;
    CUDA_CHECK(cudaMalloc(&dq, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dk, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dv, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&do_, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dq, hq.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, hk.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, hv.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    FlashAttentionParams params{dq, dk, dv, do_, shape, scale};

    CUDA_CHECK(cudaMemset(do_, 0, n * sizeof(float)));
    for (FlashAttentionBaselineMath math_mode : {
             FlashAttentionBaselineMath::kDefaultTf32,
             FlashAttentionBaselineMath::kFp32Pedantic}) {
      FlashAttentionLibraryBaseline* baseline = flashattention_library_baseline_create(shape, math_mode);
      CUDA_CHECK(cudaMemset(do_, 0, n * sizeof(float)));
      CUDA_CHECK(launch_flashattention_library_composed_cublas_cudnn(baseline, params, 0));
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(hout.data(), do_, n * sizeof(float), cudaMemcpyDeviceToHost));
      if (!check_result(log, flashattention_baseline_math_name(math_mode), shape, hout, href, atol, rtol)) failed = true;
      flashattention_library_baseline_destroy(baseline);
    }

    for (const auto& mode : user_modes) {
      if (!run_mode(log, mode, params, href, hout, atol, rtol)) failed = true;
    }
    CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk)); CUDA_CHECK(cudaFree(dv)); CUDA_CHECK(cudaFree(do_));
  }
  std::vector<FlashAttentionShape> large_cases = {
    {1, 1, 2048, 128}, {1, 2, 2048, 128}, {1, 4, 2048, 128},
    {2, 1, 2048, 128}, {2, 2, 2048, 128}, {2, 4, 2048, 128},
    {4, 1, 2048, 128}, {4, 2, 2048, 128}, {4, 4, 2048, 128},
  };
  std::vector<ModeSpec> large_modes = {
    {"v1", launch_flashattention_v1},
    {"v2", launch_flashattention_v2},
  };
  for (const auto& mode : large_modes) {
    for (size_t i = 0; i < large_cases.size(); ++i) {
      if (!run_library_reference_case(log, mode, large_cases[i], 92048 + static_cast<int>(i), atol, rtol)) failed = true;
    }
  }
  if (failed) return 1;
  std::cout << "PASS/ SKIP summary written to " << log_path << "\n";
  return 0;
}
