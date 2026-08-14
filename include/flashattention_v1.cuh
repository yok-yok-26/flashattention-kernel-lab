#pragma once
#include "flashattention.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace flashattention_v1 {

constexpr int kWarpSize = 32;

struct ProblemSize {
  int batch;
  int heads;
  int seq_len;
  int head_dim;
  int groups;
  int rows;
  int64_t qkv_elements;
  int64_t score_elements;
};

inline ProblemSize make_problem_size(const FlashAttentionShape& shape) {
  return ProblemSize{shape.batch,
                     shape.heads,
                     shape.seq_len,
                     shape.head_dim,
                     shape.batch * shape.heads,
                     shape.batch * shape.heads * shape.seq_len,
                     static_cast<int64_t>(shape.seq_len) * shape.head_dim,
                     static_cast<int64_t>(shape.seq_len) * shape.seq_len};
}

inline bool v1_shape_supported(const FlashAttentionShape& shape, const char** reason) {
  if (shape.batch <= 0 || shape.heads <= 0 || shape.seq_len <= 0 || shape.head_dim <= 0) {
    if (reason) *reason = "all dimensions must be positive";
    return false;
  }
  if (shape.head_dim > 256) {
    if (reason) *reason = "v1 scaffold caps head_dim at 256";
    return false;
  }
  if (reason) *reason = nullptr;
  return true;
}

__host__ __device__ inline int64_t qkv_offset(const FlashAttentionShape& shape,
                                              int batch,
                                              int head,
                                              int seq,
                                              int dim) {
  return (((static_cast<int64_t>(batch) * shape.heads + head) * shape.seq_len + seq) *
          shape.head_dim) + dim;
}

__host__ __device__ inline int64_t row_base_offset(const FlashAttentionShape& shape,
                                                   int batch,
                                                   int head,
                                                   int seq) {
  return qkv_offset(shape, batch, head, seq, 0);
}

__host__ __device__ inline void decode_row(const FlashAttentionShape& shape,
                                           int row,
                                           int* batch,
                                           int* head,
                                           int* seq) {
  const int rows_per_batch = shape.heads * shape.seq_len;
  *batch = row / rows_per_batch;
  const int local = row - (*batch) * rows_per_batch;
  *head = local / shape.seq_len;
  *seq = local - (*head) * shape.seq_len;
}

}  // namespace flashattention_v1
