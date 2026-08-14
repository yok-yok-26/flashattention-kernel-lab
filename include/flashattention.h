#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>

struct FlashAttentionShape {
  int batch;  // 批数
  int heads;  // 头数         
  int seq_len;  // 序列长度    M/N
  int head_dim;  // 头维度      D
};

struct FlashAttentionParams {
  const float* q;
  const float* k;
  const float* v;
  float* out;
  FlashAttentionShape shape;
  float scale;
};

enum class FlashAttentionBaselineMath {
  kDefaultTf32,
  kFp32Pedantic,
};

struct FlashAttentionLibraryBaseline;

size_t flashattention_numel(const FlashAttentionShape& shape);
std::string flashattention_shape_string(const FlashAttentionShape& shape);
bool flashattention_shape_valid(const FlashAttentionShape& shape, std::string* reason);

void flashattention_reference_cpu(const std::vector<float>& q,
                                  const std::vector<float>& k,
                                  const std::vector<float>& v,
                                  std::vector<float>& out,
                                  const FlashAttentionShape& shape,
                                  float scale);

cudaError_t launch_flashattention_user(const FlashAttentionParams& params,
                                       cudaStream_t stream);

namespace flashattention_v1 {
cudaError_t launch_flashattention_v1(const FlashAttentionParams& params,
                                     cudaStream_t stream);
}
using flashattention_v1::launch_flashattention_v1;

namespace flashattention_v2 {
cudaError_t launch_flashattention_v2(const FlashAttentionParams& params,
                                     cudaStream_t stream);
}
using flashattention_v2::launch_flashattention_v2;

const char* flashattention_baseline_math_name(FlashAttentionBaselineMath math_mode);
FlashAttentionLibraryBaseline* flashattention_library_baseline_create(
    const FlashAttentionShape& shape,
    FlashAttentionBaselineMath math_mode = FlashAttentionBaselineMath::kDefaultTf32);
void flashattention_library_baseline_destroy(FlashAttentionLibraryBaseline* baseline);
cudaError_t launch_flashattention_library_composed_cublas_cudnn(
    FlashAttentionLibraryBaseline* baseline,
    const FlashAttentionParams& params,
    cudaStream_t stream);
