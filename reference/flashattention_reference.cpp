#include "flashattention.h"
#include <algorithm>
#include <cmath>
#include <sstream>

size_t flashattention_numel(const FlashAttentionShape& s) {
  return static_cast<size_t>(s.batch) * s.heads * s.seq_len * s.head_dim;
}

std::string flashattention_shape_string(const FlashAttentionShape& s) {
  std::ostringstream os;
  os << "B=" << s.batch << ",H=" << s.heads << ",S=" << s.seq_len << ",D=" << s.head_dim;
  return os.str();
}

bool flashattention_shape_valid(const FlashAttentionShape& s, std::string* reason) {
  if (s.batch <= 0 || s.heads <= 0 || s.seq_len <= 0 || s.head_dim <= 0) {
    if (reason) *reason = "all shape dimensions must be positive";
    return false;
  }
  if (s.head_dim > 256) {
    if (reason) *reason = "initial lab reference caps head_dim at 256";
    return false;
  }
  return true;
}

void flashattention_reference_cpu(const std::vector<float>& q,
                                  const std::vector<float>& k,
                                  const std::vector<float>& v,
                                  std::vector<float>& out,
                                  const FlashAttentionShape& s,
                                  float scale) {
  out.assign(flashattention_numel(s), 0.0f);
  const int D = s.head_dim;
  const int S = s.seq_len;
  for (int b = 0; b < s.batch; ++b) {
    for (int h = 0; h < s.heads; ++h) {
      const size_t base = (static_cast<size_t>(b) * s.heads + h) * S * D;
      for (int i = 0; i < S; ++i) {
        float max_score = -INFINITY;
        for (int j = 0; j < S; ++j) {
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) dot += q[base + i * D + d] * k[base + j * D + d];
          max_score = std::max(max_score, dot * scale);
        }
        float denom = 0.0f;
        for (int j = 0; j < S; ++j) {
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) dot += q[base + i * D + d] * k[base + j * D + d];
          denom += std::exp(dot * scale - max_score);
        }
        for (int d = 0; d < D; ++d) {
          float acc = 0.0f;
          for (int j = 0; j < S; ++j) {
            float dot = 0.0f;
            for (int kk = 0; kk < D; ++kk) dot += q[base + i * D + kk] * k[base + j * D + kk];
            float p = std::exp(dot * scale - max_score) / denom;
            acc += p * v[base + j * D + d];
          }
          out[base + i * D + d] = acc;
        }
      }
    }
  }
}
