#include "flashattention.h"
#include "flashattention_v1.cuh"
#include "cuda_check.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace flashattention_v3 {

// User-owned area starts here.
// Define your __global__ kernel(s), device helpers, and launch_flashattention_v3
// in this file. Codex owns only the surrounding harness and shared scaffolding.
//
// Expected public entrypoint for the harness:
//



__device__ __forceinline__ 
void warp_all_reduce_max_sum(unsigned mask, float& max_val, float& sum_val){

  float max_val_old = max_val;

  for (int stride = 16 / 2; stride > 0; stride >>= 1)
  {
    float other = __shfl_xor_sync(mask, max_val, stride, 16);
    if (other > max_val)
    {
      max_val = other;
    }
  }

  sum_val *= expf(max_val_old - max_val);

  for (int stride = 16 / 2; stride > 0; stride >>= 1)
  {
    sum_val += __shfl_xor_sync(mask, sum_val, stride, 16);
  }

}



template <size_t D = 128, size_t M = 2048, size_t N = 2048, size_t TILE_M = 8, size_t TILE_N = 4, 
size_t BLOCK_M = 64, size_t BLOCK_N = 64, size_t BLOCK_D = 16, 
size_t BLOCKSIZE = 128>
__global__ void flashattention_v3(FlashAttentionParams params) {

  if (params.shape.head_dim != D || params.shape.seq_len != M) return;
  // if (params.shape.heads != 1 || params.shape.batch != 1) return;

  __shared__ float smem_q[BLOCK_M][BLOCK_D + 4];   // 64 * 16  /  128 = 8
  __shared__ float smem_k[BLOCK_N][BLOCK_D + 4];
  __shared__ float smem_v[BLOCK_N][BLOCK_D];
  __shared__ float smem_p[BLOCK_M][BLOCK_N];    // 64 * 64 / 128 = 32

  __shared__ float smem_sum[BLOCK_M];
  __shared__ float smem_max[BLOCK_M];
  __shared__ float smem_max_old[BLOCK_M];


  int tx = threadIdx.x;
  int tx64 = (threadIdx.x % (BLOCKSIZE / 2));
  int idx_d = tx64 % (BLOCKSIZE / 2 / BLOCK_M); // [0,1)
  int idx_m = tx64 / (BLOCKSIZE / 2 / BLOCK_M); // [0, 64)
  int idx_tile_m = threadIdx.x / 16;   // [0, 8)
  int idx_tile_n = threadIdx.x % 16;   // [0, 16)
  int idx_tile2_m = threadIdx.x / 8;   // [0, 16)
  int idx_tile2_n = threadIdx.x % 8;   // [0, 8)
  float tmp[8][4] = {0.0f}, tmp2[4][2] = {0.0f};
  float o_acc[(D / BLOCK_D)][4][2] = {0.0f};
  size_t offset = blockIdx.y * N * D + blockIdx.z * N * D * params.shape.heads;
  



  if ((tx & 1) == 0)
  {
    smem_max[tx / 2] = -INFINITY;
    smem_max_old[tx / 2] = -INFINITY;
    smem_sum[tx / 2] = 0.0f;
  }
  __syncthreads();


  for (int iloop_n = 0; iloop_n < N; iloop_n += BLOCK_N){

    float* q_tile = const_cast<float*>(params.q + offset + blockIdx.x * BLOCK_M * D);
    float* k_tile = const_cast<float*>(params.k + offset + iloop_n * D);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            tmp[i][j] = 0.0f;
        }
    }


    for (int iloop_d = 0; iloop_d < D; iloop_d += BLOCK_D)
    {
      if (threadIdx.x < (BLOCKSIZE / 2))
      {
        *reinterpret_cast<float4*>(&smem_q[tx64][0]) = 
        *reinterpret_cast<float4*>(q_tile + iloop_d + tx64 * D + 0);
        *reinterpret_cast<float4*>(&smem_q[tx64][4]) = 
        *reinterpret_cast<float4*>(q_tile + iloop_d + tx64 * D + 4);
        *reinterpret_cast<float4*>(&smem_q[tx64][8]) = 
        *reinterpret_cast<float4*>(q_tile + iloop_d + tx64 * D + 8);
        *reinterpret_cast<float4*>(&smem_q[tx64][12]) = 
        *reinterpret_cast<float4*>(q_tile + iloop_d + tx64 * D + 12);
      }
      else
      {
        *reinterpret_cast<float4*>(&smem_k[tx64][0]) = 
        *reinterpret_cast<float4*>(k_tile + iloop_d + tx64 * D + 0);
        *reinterpret_cast<float4*>(&smem_k[tx64][4]) = 
        *reinterpret_cast<float4*>(k_tile + iloop_d + tx64 * D + 4);
        *reinterpret_cast<float4*>(&smem_k[tx64][8]) = 
        *reinterpret_cast<float4*>(k_tile + iloop_d + tx64 * D + 8);
        *reinterpret_cast<float4*>(&smem_k[tx64][12]) = 
        *reinterpret_cast<float4*>(k_tile + iloop_d + tx64 * D + 12);
      }
      __syncthreads();
      

      for (int i = 0; i < BLOCK_D; i++)
      {
        float a0 = smem_q[idx_tile_m * 8 + 0][i];
        float a1 = smem_q[idx_tile_m * 8 + 1][i];
        float a2 = smem_q[idx_tile_m * 8 + 2][i];
        float a3 = smem_q[idx_tile_m * 8 + 3][i];
        float a4 = smem_q[idx_tile_m * 8 + 4][i];
        float a5 = smem_q[idx_tile_m * 8 + 5][i];
        float a6 = smem_q[idx_tile_m * 8 + 6][i];
        float a7 = smem_q[idx_tile_m * 8 + 7][i];

        float b0 = smem_k[idx_tile_n * 4 + 0][i];
        float b1 = smem_k[idx_tile_n * 4 + 1][i];
        float b2 = smem_k[idx_tile_n * 4 + 2][i];
        float b3 = smem_k[idx_tile_n * 4 + 3][i];

        tmp[0][0] += a0 * b0;
        tmp[0][1] += a0 * b1;
        tmp[0][2] += a0 * b2;
        tmp[0][3] += a0 * b3;

        tmp[1][0] += a1 * b0;
        tmp[1][1] += a1 * b1;
        tmp[1][2] += a1 * b2;
        tmp[1][3] += a1 * b3;

        tmp[2][0] += a2 * b0;
        tmp[2][1] += a2 * b1;
        tmp[2][2] += a2 * b2;
        tmp[2][3] += a2 * b3;

        tmp[3][0] += a3 * b0;
        tmp[3][1] += a3 * b1;
        tmp[3][2] += a3 * b2;
        tmp[3][3] += a3 * b3;

        tmp[4][0] += a4 * b0;
        tmp[4][1] += a4 * b1;
        tmp[4][2] += a4 * b2;
        tmp[4][3] += a4 * b3;

        tmp[5][0] += a5 * b0;
        tmp[5][1] += a5 * b1;
        tmp[5][2] += a5 * b2;
        tmp[5][3] += a5 * b3;

        tmp[6][0] += a6 * b0;
        tmp[6][1] += a6 * b1;
        tmp[6][2] += a6 * b2;
        tmp[6][3] += a6 * b3;

        tmp[7][0] += a7 * b0;
        tmp[7][1] += a7 * b1;
        tmp[7][2] += a7 * b2;
        tmp[7][3] += a7 * b3;

      }

      __syncthreads();
      
    }


    // 128个线程，64组规约。128个线程一次可以完成2组规约
    float max_val, sum_val;
    for (int irow = 0; irow < 8; irow++){

        max_val = -INFINITY;
        sum_val = 0.0f;

      for (int icol = 0; icol < 4; icol++){
        if (max_val < tmp[irow][icol] * params.scale)
        {
          max_val = tmp[irow][icol] * params.scale;
        }
      }

      for (int icol = 0; icol < 4; icol++){
        sum_val += expf(tmp[irow][icol] * params.scale - max_val);
      }

      // !!!!!!!!!!!!!!!!!!!!!! 先不考虑边界情况 !!!!!!!!!!!!!!!!!!!!!!
      unsigned mask = 0xffffffff;
      warp_all_reduce_max_sum(mask, max_val, sum_val);

      if ((tx & 15) == 0)
      {
        smem_max[idx_tile_m * 8 + irow] = max_val;

        if (smem_max_old[idx_tile_m * 8 + irow] < smem_max[idx_tile_m * 8 + irow])
        {
          smem_sum[idx_tile_m * 8 + irow] = 
            smem_sum[idx_tile_m * 8 + irow] * 
            expf(smem_max_old[idx_tile_m * 8 + irow] - smem_max[idx_tile_m * 8 + irow]) + sum_val;
        }
        else
        {
          smem_sum[idx_tile_m * 8 + irow] += 
            sum_val * expf(smem_max[idx_tile_m * 8 + irow] - smem_max_old[idx_tile_m * 8 + irow]);
        }
      }
      __syncthreads();

      for (int icol = 0; icol < 4; icol++){
        tmp[irow][icol] = expf((tmp[irow][icol] * params.scale - smem_max[idx_tile_m * 8 + irow]));
      }


    }

    smem_p[idx_tile_m * 8 + 0][idx_tile_n * 4 + 0] = tmp[0][0];
    smem_p[idx_tile_m * 8 + 0][idx_tile_n * 4 + 1] = tmp[0][1];
    smem_p[idx_tile_m * 8 + 0][idx_tile_n * 4 + 2] = tmp[0][2];
    smem_p[idx_tile_m * 8 + 0][idx_tile_n * 4 + 3] = tmp[0][3];

    smem_p[idx_tile_m * 8 + 1][idx_tile_n * 4 + 0] = tmp[1][0];
    smem_p[idx_tile_m * 8 + 1][idx_tile_n * 4 + 1] = tmp[1][1];
    smem_p[idx_tile_m * 8 + 1][idx_tile_n * 4 + 2] = tmp[1][2];
    smem_p[idx_tile_m * 8 + 1][idx_tile_n * 4 + 3] = tmp[1][3];

    smem_p[idx_tile_m * 8 + 2][idx_tile_n * 4 + 0] = tmp[2][0];
    smem_p[idx_tile_m * 8 + 2][idx_tile_n * 4 + 1] = tmp[2][1];
    smem_p[idx_tile_m * 8 + 2][idx_tile_n * 4 + 2] = tmp[2][2];
    smem_p[idx_tile_m * 8 + 2][idx_tile_n * 4 + 3] = tmp[2][3];

    smem_p[idx_tile_m * 8 + 3][idx_tile_n * 4 + 0] = tmp[3][0];
    smem_p[idx_tile_m * 8 + 3][idx_tile_n * 4 + 1] = tmp[3][1];
    smem_p[idx_tile_m * 8 + 3][idx_tile_n * 4 + 2] = tmp[3][2];
    smem_p[idx_tile_m * 8 + 3][idx_tile_n * 4 + 3] = tmp[3][3];

    smem_p[idx_tile_m * 8 + 4][idx_tile_n * 4 + 0] = tmp[4][0];
    smem_p[idx_tile_m * 8 + 4][idx_tile_n * 4 + 1] = tmp[4][1];
    smem_p[idx_tile_m * 8 + 4][idx_tile_n * 4 + 2] = tmp[4][2];
    smem_p[idx_tile_m * 8 + 4][idx_tile_n * 4 + 3] = tmp[4][3];

    smem_p[idx_tile_m * 8 + 5][idx_tile_n * 4 + 0] = tmp[5][0];
    smem_p[idx_tile_m * 8 + 5][idx_tile_n * 4 + 1] = tmp[5][1];
    smem_p[idx_tile_m * 8 + 5][idx_tile_n * 4 + 2] = tmp[5][2];
    smem_p[idx_tile_m * 8 + 5][idx_tile_n * 4 + 3] = tmp[5][3];

    smem_p[idx_tile_m * 8 + 6][idx_tile_n * 4 + 0] = tmp[6][0];
    smem_p[idx_tile_m * 8 + 6][idx_tile_n * 4 + 1] = tmp[6][1];
    smem_p[idx_tile_m * 8 + 6][idx_tile_n * 4 + 2] = tmp[6][2];
    smem_p[idx_tile_m * 8 + 6][idx_tile_n * 4 + 3] = tmp[6][3];

    smem_p[idx_tile_m * 8 + 7][idx_tile_n * 4 + 0] = tmp[7][0];
    smem_p[idx_tile_m * 8 + 7][idx_tile_n * 4 + 1] = tmp[7][1];
    smem_p[idx_tile_m * 8 + 7][idx_tile_n * 4 + 2] = tmp[7][2];
    smem_p[idx_tile_m * 8 + 7][idx_tile_n * 4 + 3] = tmp[7][3];

    __syncthreads();

    float* v_tile = const_cast<float*>(params.v + offset + iloop_n * D);
    for (int iloop_d = 0; iloop_d < D; iloop_d += BLOCK_D)
    {

      #pragma unroll
      for (int j = 0; j < 4; ++j) {
          #pragma unroll
          for (int k = 0; k < 2; ++k) {
            tmp2[j][k] = 0.0f;
          }
      }


      int id = iloop_d / BLOCK_D;
      // 64 ---> 128 threads   16*64/128=8
      *reinterpret_cast<float4*>(&smem_v[(tx / 2)][(tx % 2) * 8 + 0]) = 
      *reinterpret_cast<float4*>(v_tile + iloop_d + (tx / 2) * D + (tx % 2) * 8 + 0);
      *reinterpret_cast<float4*>(&smem_v[(tx / 2)][(tx % 2) * 8 + 4]) = 
      *reinterpret_cast<float4*>(v_tile + iloop_d + (tx / 2) * D + (tx % 2) * 8 + 4);

      __syncthreads();
      
      // 64*64/128=32
      // [64,64]  [64, 16]---->64*16/128=8--->[4,2]
      // int idx_tile2_m = threadIdx.x / 8;   // [0, 16)
      // int idx_tile2_n = threadIdx.x % 8;   // [0, 8)

      for (int i = 0; i < BLOCK_N; i++)
      {
        float a0 = smem_p[idx_tile2_m * 4 + 0][i];
        float a1 = smem_p[idx_tile2_m * 4 + 1][i];
        float a2 = smem_p[idx_tile2_m * 4 + 2][i];
        float a3 = smem_p[idx_tile2_m * 4 + 3][i];

        float b0 = smem_v[i][idx_tile2_n * 2 + 0];
        float b1 = smem_v[i][idx_tile2_n * 2 + 1];

        tmp2[0][0] += a0 * b0;
        tmp2[0][1] += a0 * b1;
        tmp2[1][0] += a1 * b0;
        tmp2[1][1] += a1 * b1;
        tmp2[2][0] += a2 * b0;
        tmp2[2][1] += a2 * b1;
        tmp2[3][0] += a3 * b0;
        tmp2[3][1] += a3 * b1;
      }
      __syncthreads();


      if (smem_max_old[idx_tile2_m * 4 + 0] < smem_max[idx_tile2_m * 4 + 0])
      {
        o_acc[id][0][0] = tmp2[0][0] + expf(
          smem_max_old[idx_tile2_m * 4 + 0] - smem_max[idx_tile2_m * 4 + 0]
        ) * o_acc[id][0][0];
        o_acc[id][0][1] = tmp2[0][1] + expf(
          smem_max_old[idx_tile2_m * 4 + 0] - smem_max[idx_tile2_m * 4 + 0]
        ) * o_acc[id][0][1];
      }
      else
      {
        o_acc[id][0][0] = expf(
          smem_max[idx_tile2_m * 4 + 0] - smem_max_old[idx_tile2_m * 4 + 0]
        ) * tmp2[0][0] 
        + o_acc[id][0][0];
        o_acc[id][0][1] = expf(
          smem_max[idx_tile2_m * 4 + 0] - smem_max_old[idx_tile2_m * 4 + 0]
        ) * tmp2[0][1] 
        + o_acc[id][0][1];
      }
      

      if (smem_max_old[idx_tile2_m * 4 + 1] < smem_max[idx_tile2_m * 4 + 1])
      {
        o_acc[id][1][0] = tmp2[1][0] + expf(
          smem_max_old[idx_tile2_m * 4 + 1] - smem_max[idx_tile2_m * 4 + 1]
        ) * o_acc[id][1][0];
        o_acc[id][1][1] = tmp2[1][1] + expf(
          smem_max_old[idx_tile2_m * 4 + 1] - smem_max[idx_tile2_m * 4 + 1]
        ) * o_acc[id][1][1];
      }
      else
      {
        o_acc[id][1][0] = expf(
          smem_max[idx_tile2_m * 4 + 1] - smem_max_old[idx_tile2_m * 4 + 1]
        ) * tmp2[1][0] 
        + o_acc[id][1][0];
        o_acc[id][1][1] = expf(
          smem_max[idx_tile2_m * 4 + 1] - smem_max_old[idx_tile2_m * 4 + 1]
        ) * tmp2[1][1] 
        + o_acc[id][1][1];
      }


      if (smem_max_old[idx_tile2_m * 4 + 2] < smem_max[idx_tile2_m * 4 + 2])
      {
        o_acc[id][2][0] = tmp2[2][0] + expf(
          smem_max_old[idx_tile2_m * 4 + 2] - smem_max[idx_tile2_m * 4 + 2]
        ) * o_acc[id][2][0];
        o_acc[id][2][1] = tmp2[2][1] + expf(
          smem_max_old[idx_tile2_m * 4 + 2] - smem_max[idx_tile2_m * 4 + 2]
        ) * o_acc[id][2][1];
      }
      else
      {
        o_acc[id][2][0] = expf(
          smem_max[idx_tile2_m * 4 + 2] - smem_max_old[idx_tile2_m * 4 + 2]
        ) * tmp2[2][0] 
        + o_acc[id][2][0];
        o_acc[id][2][1] = expf(
          smem_max[idx_tile2_m * 4 + 2] - smem_max_old[idx_tile2_m * 4 + 2]
        ) * tmp2[2][1] 
        + o_acc[id][2][1];
      }


      if (smem_max_old[idx_tile2_m * 4 + 3] < smem_max[idx_tile2_m * 4 + 3])
      {
        o_acc[id][3][0] = tmp2[3][0] + expf(
          smem_max_old[idx_tile2_m * 4 + 3] - smem_max[idx_tile2_m * 4 + 3]
        ) * o_acc[id][3][0];
        o_acc[id][3][1] = tmp2[3][1] + expf(
          smem_max_old[idx_tile2_m * 4 + 3] - smem_max[idx_tile2_m * 4 + 3]
        ) * o_acc[id][3][1];
      }
      else
      {
        o_acc[id][3][0] = expf(
          smem_max[idx_tile2_m * 4 + 3] - smem_max_old[idx_tile2_m * 4 + 3]
        ) * tmp2[3][0] 
        + o_acc[id][3][0];
        o_acc[id][3][1] = expf(
          smem_max[idx_tile2_m * 4 + 3] - smem_max_old[idx_tile2_m * 4 + 3]
        ) * tmp2[3][1] 
        + o_acc[id][3][1];
      }
      
    }



    if ((tx & 7) == 0)
    {
      for (int i = 0; i < 4; i++)
      {
        smem_max_old[idx_tile2_m * 4 + i] = fmaxf(
          smem_max_old[idx_tile2_m * 4 + i], smem_max[idx_tile2_m * 4 + i]
        );
      }
    }

  }


  // write back
  #pragma unroll
  for (int i = 0; i < (BLOCKSIZE / BLOCK_D); ++i) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
          #pragma unroll
          for (int k = 0; k < 2; ++k) {
            *(
              params.out + offset + blockIdx.x * BLOCK_M * D + 
              i * BLOCK_D + idx_tile2_n * 2 + k + 
              (idx_tile2_m * 4 + j) * D
            ) = o_acc[i][j][k] / smem_sum[idx_tile2_m * 4 + j];
          }
      }
  }

}





cudaError_t launch_flashattention_v3(const FlashAttentionParams& params,
                                     cudaStream_t stream) {
  const char* reason = nullptr;
  if (!flashattention_v1::v1_shape_supported(params.shape, &reason)) {
    return cudaErrorInvalidValue;
  }

  dim3 grid{params.shape.seq_len / 64, params.shape.heads, params.shape.batch};
  dim3 block{128};
  flashattention_v3<<<grid, block, 0, stream>>>(params);
  CUDA_KERNEL_CHECK();

  // Your launch policy and kernel launch go here.
  // kernel<<<grid, block, smem, stream>>>(...);
  // CUDA_KERNEL_CHECK();
  // #ifdef DEBUG_CUDA_SYNC
  // CUDA_CHECK(cudaStreamSynchronize(stream));
  // #endif
  return cudaSuccess;
}

}  // namespace flashattention_v3
