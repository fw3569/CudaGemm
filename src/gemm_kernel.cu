#include "gemm_kernel.cuh"

// threads of block
#define TOBM 16
#define TOBN 16
// elements of thread
#define EOTM 8
#define EOTN 8
// elements of block
#define EOBM (TOBM * EOTM)
#define EOBN (TOBN * EOTN)
// stride k
#define STRIDE_K 16

__global__ void __launch_bounds__(TOBM* TOBN, 2)
    gemm_kernel(const float* __restrict__ a, const float* __restrict__ b,
                float* __restrict__ c, int M, int N, int K) {
  // align to cache line
  __shared__ alignas(128) float sa[STRIDE_K][EOBM + 4], sb[STRIDE_K][EOBN];
  int block_row = blockIdx.y * EOBM;
  int block_col = blockIdx.x * EOBN;
  float accum[EOTM][EOTN] = {};
  for (int kh = 0; kh < K; kh += STRIDE_K) {
    // copy to shared memory
#pragma unroll
    for (int i = 0, o = threadIdx.y * TOBN + threadIdx.x;
         i * TOBN < STRIDE_K * EOTM; ++i, o += TOBN * TOBM) {
      // linear indexing to enable coalesced access
      int row = o / STRIDE_K;
      int col = o & (STRIDE_K - 1);
      // transpose to support float4 load latter
      // multiple instead of if statement, friendly to instruction reordering
      // swizzle 0-3, shift 4 bank each row
      sa[col][row ^ (col >> 2)] =
          (block_row + row < M && kh + col < K) *
          a[min((block_row + row) * K + kh + col, M * K - 1)];
    }
#pragma unroll
    for (int i = 0, o = threadIdx.y * TOBN + threadIdx.x;
         i * TOBM < STRIDE_K * EOTN; ++i, o += TOBN * TOBM) {
      // linear indexing to enable coalesced access
      int row = o / EOBN;
      int col = o & (EOBN - 1);
      // multiple instead of if statement, friendly to instruction reordering
      sb[row][col] = (kh + row < K && block_col + col < N) *
                     b[min((kh + row) * N + block_col + col, K * N - 1)];
    }
    __syncthreads();

    // calculate
    // outer-product based register blocking to maximize data reuse
    for (int kl = 0; kl < STRIDE_K; ++kl) {
      float rega[EOTM], regb[EOTN];
#pragma unroll
      for (int i = 0, p = (threadIdx.y << 2); i < EOTM; i += 4, p += 4 * TOBM) {
        // float4 to reduce io instructions, deal with mio throttle
        *(float4*)(&rega[i]) = *(float4*)(&sa[kl][p]);
      }
#pragma unroll
      for (int j = 0, p = (threadIdx.x << 2); j < EOTN; j += 4, p += 4 * TOBN) {
        // float4 to reduce io instructions, deal with mio throttle
        *(float4*)(&regb[j]) = *(float4*)(&sb[kl][p]);
      }
#pragma unroll
      for (int i = 0; i < EOTM; ++i) {
#pragma unroll
        for (int j = 0; j < EOTN; ++j) {
          // swizzle 0-3, shift 4 bank each row
          accum[i][j] += rega[i ^ (kl >> 2)] * regb[j];
        }
      }
    }
    __syncthreads();
  }
  if ((N & 3) == 0 && ((uintptr_t)c & 3) == 0) [[likely]] {
#pragma unroll
    for (int i = 0;
         i < EOTM &&
         block_row + (threadIdx.y << 2) + (i & ~3) * TOBM + (i & 3) < M;
         ++i) {
#pragma unroll
      for (int j = 0;
           j < EOTN &&
           block_col + (threadIdx.x << 2) + (j & ~3) * TOBN + (j & 3) < N;
           j += 4) {
        // float4 to reduce instructions and address calculate
        *(float4*)(&c[(block_row + (threadIdx.y << 2) + (i & ~3) * TOBM +
                       (i & 3)) *
                          N +
                      block_col + (threadIdx.x << 2) + j * TOBN]) =
            *(float4*)(&accum[i][j]);
      }
    }
  } else {
#pragma unroll
    for (int i = 0;
         i < EOTM &&
         block_row + (threadIdx.y << 2) + (i & ~3) * TOBM + (i & 3) < M;
         ++i) {
#pragma unroll
      for (int j = 0;
           j < EOTN &&
           block_col + (threadIdx.x << 2) + (j & ~3) * TOBN + (j & 3) < N;
           ++j) {
        c[(block_row + (threadIdx.y << 2) + (i & ~3) * TOBM + (i & 3)) * N +
          block_col + (threadIdx.x << 2) + (j & ~3) * TOBN + (j & 3)] =
            accum[i][j];
      }
    }
  }
}

void gemm(float* a, float* b, float* c, int M, int N, int K) {
  dim3 grid((N + EOBN - 1) / EOBN, (M + EOBM - 1) / EOBM);
  dim3 block(TOBN, TOBM);
  gemm_kernel<<<grid, block>>>(a, b, c, M, N, K);
}
