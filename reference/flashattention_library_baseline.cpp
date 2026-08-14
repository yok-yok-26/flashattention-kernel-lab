#include "flashattention.h"
#include "cuda_check.h"
#include "library_check.h"
#include <cstddef>

struct FlashAttentionLibraryBaseline {
  FlashAttentionShape shape;
  FlashAttentionBaselineMath math_mode = FlashAttentionBaselineMath::kDefaultTf32;
  cublasHandle_t cublas = nullptr;
  cudnnHandle_t cudnn = nullptr;
  cudnnTensorDescriptor_t softmax_desc = nullptr;
  float* scores = nullptr;
  float* probs = nullptr;
};

const char* flashattention_baseline_math_name(FlashAttentionBaselineMath math_mode) {
  switch (math_mode) {
    case FlashAttentionBaselineMath::kDefaultTf32:
      return "library_composed_cublas_cudnn_tf32";
    case FlashAttentionBaselineMath::kFp32Pedantic:
      return "library_composed_cublas_cudnn_fp32_pedantic";
  }
  return "library_composed_cublas_cudnn_unknown";
}

FlashAttentionLibraryBaseline* flashattention_library_baseline_create(
    const FlashAttentionShape& shape,
    FlashAttentionBaselineMath math_mode) {
  auto* baseline = new FlashAttentionLibraryBaseline();
  baseline->shape = shape;
  baseline->math_mode = math_mode;
  CUBLAS_CHECK(cublasCreate(&baseline->cublas));
  if (math_mode == FlashAttentionBaselineMath::kFp32Pedantic) {
    CUBLAS_CHECK(cublasSetMathMode(baseline->cublas, CUBLAS_PEDANTIC_MATH));
  } else {
    CUBLAS_CHECK(cublasSetMathMode(baseline->cublas, CUBLAS_TF32_TENSOR_OP_MATH));
  }
  CUDNN_CHECK(cudnnCreate(&baseline->cudnn));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&baseline->softmax_desc));
  const int rows = shape.batch * shape.heads * shape.seq_len;
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(baseline->softmax_desc,
                                         CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT,
                                         rows,
                                         shape.seq_len,
                                         1,
                                         1));
  const size_t score_elems = static_cast<size_t>(shape.batch) * shape.heads * shape.seq_len * shape.seq_len;
  CUDA_CHECK(cudaMalloc(&baseline->scores, score_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&baseline->probs, score_elems * sizeof(float)));
  return baseline;
}

void flashattention_library_baseline_destroy(FlashAttentionLibraryBaseline* baseline) {
  if (!baseline) return;
  if (baseline->scores) CUDA_CHECK(cudaFree(baseline->scores));
  if (baseline->probs) CUDA_CHECK(cudaFree(baseline->probs));
  if (baseline->softmax_desc) CUDNN_CHECK(cudnnDestroyTensorDescriptor(baseline->softmax_desc));
  if (baseline->cudnn) CUDNN_CHECK(cudnnDestroy(baseline->cudnn));
  if (baseline->cublas) CUBLAS_CHECK(cublasDestroy(baseline->cublas));
  delete baseline;
}

cudaError_t launch_flashattention_library_composed_cublas_cudnn(
    FlashAttentionLibraryBaseline* baseline,
    const FlashAttentionParams& params,
    cudaStream_t stream) {
  if (!baseline) return cudaErrorInvalidValue;
  const FlashAttentionShape s = params.shape;
  if (s.batch != baseline->shape.batch || s.heads != baseline->shape.heads ||
      s.seq_len != baseline->shape.seq_len || s.head_dim != baseline->shape.head_dim) {
    return cudaErrorInvalidValue;
  }

  CUBLAS_CHECK(cublasSetStream(baseline->cublas, stream));
  CUDNN_CHECK(cudnnSetStream(baseline->cudnn, stream));

  const int groups = s.batch * s.heads;
  const int S = s.seq_len;
  const int D = s.head_dim;
  const long long qkv_stride = static_cast<long long>(S) * D;
  const long long score_stride = static_cast<long long>(S) * S;
  const float zero = 0.0f;
  const float one = 1.0f;

  // scores_rm = Q_rm * K_rm^T. cuBLAS sees row-major buffers as transposed column-major matrices.
  CUBLAS_CHECK(cublasSgemmStridedBatched(baseline->cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         S,
                                         S,
                                         D,
                                         &params.scale,
                                         params.k,
                                         D,
                                         qkv_stride,
                                         params.q,
                                         D,
                                         qkv_stride,
                                         &zero,
                                         baseline->scores,
                                         S,
                                         score_stride,
                                         groups));

  CUDNN_CHECK(cudnnSoftmaxForward(baseline->cudnn,
                                  CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_CHANNEL,
                                  &one,
                                  baseline->softmax_desc,
                                  baseline->scores,
                                  &zero,
                                  baseline->softmax_desc,
                                  baseline->probs));

  // out_rm = probs_rm * V_rm. The column-major view computes out_rm^T = V_rm^T * probs_rm^T.
  CUBLAS_CHECK(cublasSgemmStridedBatched(baseline->cublas,
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         D,
                                         S,
                                         S,
                                         &one,
                                         params.v,
                                         D,
                                         qkv_stride,
                                         baseline->probs,
                                         S,
                                         score_stride,
                                         &zero,
                                         params.out,
                                         D,
                                         qkv_stride,
                                         groups));
  return cudaGetLastError();
}
