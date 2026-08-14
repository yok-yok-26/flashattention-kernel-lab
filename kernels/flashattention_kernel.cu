#include "flashattention.h"
#include "cuda_check.h"

cudaError_t launch_flashattention_user(const FlashAttentionParams& params,
                                       cudaStream_t stream) {
  (void)params;
  (void)stream;
  // TODO(silenceduke): implement FlashAttention forward kernel and launch policy here.
  // Contract: Q/K/V/O are contiguous row-major float32 tensors with shape [B,H,S,D].
  // The harness treats cudaErrorNotSupported as SKIP until the user implementation exists.
  return cudaErrorNotSupported;
}
