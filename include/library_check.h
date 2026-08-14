#pragma once
#include <cublas_v2.h>
#include <cudnn.h>
#include <cstdio>
#include <cstdlib>

inline const char* cublas_status_string(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "CUBLAS_STATUS_UNKNOWN";
  }
}

#define CUBLAS_CHECK(expr)                                                    \
  do {                                                                        \
    cublasStatus_t _status = (expr);                                          \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                   \
      std::fprintf(stderr, "cuBLAS error %s:%d: %s\n", __FILE__, __LINE__,   \
                   cublas_status_string(_status));                           \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define CUDNN_CHECK(expr)                                                     \
  do {                                                                        \
    cudnnStatus_t _status = (expr);                                           \
    if (_status != CUDNN_STATUS_SUCCESS) {                                    \
      std::fprintf(stderr, "cuDNN error %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudnnGetErrorString(_status));                            \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)
