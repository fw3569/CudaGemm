#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <functional>
#include <iostream>
#include <random>
#include <vector>

// stride k
#define STRIDE_K 16
// threads of block
#define TOBM 16
#define TOBN 16
// elements of thread
#define EOTM 8
#define EOTN 8
// thread of warp
#define TOWM (32 / TOBN)
// elements of block
#define EOBM (TOBM * EOTM)
#define EOBN (TOBN * EOTN)

#ifndef USE_DOUBLE_BUFFER
#define USE_DOUBLE_BUFFER false
#if USE_DOUBLE_BUFFER
#include <cooperative_groups/memcpy_async.h>
#endif
#endif

__global__ void __launch_bounds__(TOBM* TOBN, 2)
    gemm_kernel(const float* __restrict__ a, const float* __restrict__ b,
                float* __restrict__ c, int M, int N, int K) {
  // align to cache line
#if USE_DOUBLE_BUFFER
  __shared__ alignas(128) float sa[2][STRIDE_K][EOBM + 4],
      sb[2][STRIDE_K][EOBN];
  int cache_id = 0;
#else
  __shared__ alignas(128) float sa[STRIDE_K][EOBM + 4], sb[STRIDE_K][EOBN];
#endif
  int block_row = blockIdx.y * EOBM;
  int block_col = blockIdx.x * EOBN;
  float accum[EOTM][EOTN] = {};
#if USE_DOUBLE_BUFFER
#pragma unroll
  for (int i = 0, o = threadIdx.y * TOBN + threadIdx.x;
       i * TOBN < STRIDE_K * EOTM; ++i, o += TOBN * TOBM) {
    int row = o / STRIDE_K;
    int col = o & (STRIDE_K - 1);
    __pipeline_memcpy_async(&sa[cache_id][col][row ^ (col / (STRIDE_K >> 2))],
                            &a[min((block_row + row) * K + col, M * K - 1)], 4,
                            4 * !((block_row + row < M) & (col < K)));
  }
#pragma unroll
  for (int i = 0, o = threadIdx.y * TOBN + threadIdx.x;
       i * TOBM < STRIDE_K * EOTN; ++i, o += TOBN * TOBM) {
    int row = o / EOBN;
    int col = o & (EOBN - 1);
    __pipeline_memcpy_async(&sb[cache_id][row][col],
                            &b[min((row)*N + block_col + col, K * N - 1)], 4,
                            4 * !((row < K) & (block_col + col < N)));
  }
  __pipeline_commit();
  cache_id ^= 1;
#else
#pragma unroll
  for (int i = 0, o = threadIdx.y * TOBN + threadIdx.x;
       i * TOBN < STRIDE_K * EOTM; ++i, o += TOBN * TOBM) {
    // linear indexing to enable coalesced access
    int row = o / STRIDE_K;
    int col = o & (STRIDE_K - 1);
    // transpose to support float4 load latter
    // multiple instead of if statement, friendly to instruction reordering
    // swizzle 0-3, shift 4 bank each row
    sa[col][row ^ (col / (STRIDE_K >> 2))] =
        ((block_row + row < M) & (col < K)) *
        a[min((block_row + row) * K + col, M * K - 1)];
  }
#endif
  for (int kh = 0; kh < K; kh += STRIDE_K) {
    // copy to shared memory
#if USE_DOUBLE_BUFFER
#endif
#pragma unroll
    for (int i = 0, o = threadIdx.y * TOBN + threadIdx.x;
         i * TOBM < STRIDE_K * EOTN; ++i, o += TOBN * TOBM) {
      // linear indexing to enable coalesced access
      int row = o / EOBN;
      int col = o & (EOBN - 1);
      // multiple instead of if statement, friendly to instruction reordering
#if USE_DOUBLE_BUFFER
      __pipeline_memcpy_async(
          &sb[cache_id][row][col],
          &b[min((kh + STRIDE_K + row) * N + block_col + col, K * N - 1)], 4,
          4 * !((kh + STRIDE_K + row < K) & (block_col + col < N)));
#else
      sb[row][col] = ((kh + row < K) & (block_col + col < N)) *
                     b[min((kh + row) * N + block_col + col, K * N - 1)];
#endif
    }
#if USE_DOUBLE_BUFFER
    __pipeline_commit();
    __pipeline_wait_prior(1);
    cache_id ^= 1;
#endif
    __syncthreads();

    // calculate
    // outer-product based register blocking to maximize data reuse
    for (int kl = 0; kl < STRIDE_K; ++kl) {
      float rega[EOTM], regb[EOTN];
#pragma unroll
      for (int i = 0, p = (threadIdx.y << 2) * (EOTM >> 2); i < EOTM;
           i += 4, p += 4) {
        // float4 to reduce io instructions, deal with mio throttle
#if USE_DOUBLE_BUFFER
        *(float4*)(&rega[i]) = *(float4*)(&sa[cache_id][kl][p]);
#else
        *(float4*)(&rega[i]) = *(float4*)(&sa[kl][p]);
#endif
      }
#pragma unroll
      for (int j = 0, p = (threadIdx.x << 2); j < EOTN; j += 4, p += 4 * TOBN) {
        // float4 to reduce io instructions, deal with mio throttle
#if USE_DOUBLE_BUFFER
        *(float4*)(&regb[j]) = *(float4*)(&sb[cache_id][kl][p]);
#else
        *(float4*)(&regb[j]) = *(float4*)(&sb[kl][p]);
#endif
      }
#pragma unroll
      for (int i = 0; i < EOTM; ++i) {
#pragma unroll
        for (int j = 0; j < EOTN; ++j) {
          // swizzle 0-3, shift 4 bank each row
          accum[i][j] += rega[i ^ (kl / (STRIDE_K >> 2))] * regb[j];
        }
      }
    }
#if !USE_DOUBLE_BUFFER
#pragma unroll
    for (int i = 0, o = ((threadIdx.y & (~(TOWM - 1))) * STRIDE_K * EOTM) +
                        (threadIdx.y & (TOWM - 1)) * TOBN + threadIdx.x;
         i * TOBN < STRIDE_K * EOTM; ++i, o += 32) {
      int row = o / STRIDE_K;
      int col = o & (STRIDE_K - 1);
      // be reordered to interleaved execution, perfetch data used by this warp
      sa[col][row ^ (col / (STRIDE_K >> 2))] =
          ((block_row + row < M) & (kh + STRIDE_K + col < K)) *
          a[min((block_row + row) * K + kh + STRIDE_K + col, M * K - 1)];
    }
#else
#pragma unroll
    for (int i = 0, o = ((threadIdx.y & (~(TOWM - 1))) * STRIDE_K * EOTM) +
                        (threadIdx.y & (TOWM - 1)) * TOBN + threadIdx.x;
         i * TOBN < STRIDE_K * EOTM; ++i, o += 32) {
      int row = o / STRIDE_K;
      int col = o & (STRIDE_K - 1);
      // be reordered to interleaved execution, perfetch data used by this warp
      __pipeline_memcpy_async(
          &sa[cache_id ^ 1][col][row ^ (col / (STRIDE_K >> 2))],
          &a[min((block_row + row) * K + kh + STRIDE_K + col, M * K - 1)], 4,
          4 * !((block_row + row < M) & (kh + STRIDE_K + col < K)));
    }
#endif
    __syncthreads();
  }
  if (((N & 3) == 0) & (((uintptr_t)c & 3) == 0)) [[likely]] {
#pragma unroll
    for (int i = 0; i < EOTM; ++i) {
      if (block_row + (threadIdx.y << 2) * (EOTM >> 2) + (i & (EOTM - 1)) >=
          M) {
        break;
      }
#pragma unroll
      for (int j = 0; j < EOTN; j += 4) {
        if (block_col + (threadIdx.x << 2) + j * TOBN >= N) {
          break;
        }
        // float4 to reduce instructions and address calculate
        *(float4*)(&c[(block_row + (threadIdx.y << 2) * (EOTM >> 2) +
                       (i & (EOTM - 1))) *
                          N +
                      block_col + (threadIdx.x << 2) + j * TOBN]) =
            *(float4*)(&accum[i][j]);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < EOTM; ++i) {
      if ((threadIdx.y << 2) * (EOTM >> 2) + (i & (EOTM - 1)) >= M) {
        break;
      }
#pragma unroll
      for (int j = 0; j < EOTN; ++j) {
        if (block_col + (threadIdx.x << 2) + (j & ~3) * TOBN + (j & 3) >= N) {
          break;
        }
        c[(block_row + (threadIdx.y << 2) * (EOTM >> 2) + (i & (EOTM - 1))) *
              N +
          block_col + (threadIdx.x << 2) + (j & ~3) * TOBN + (j & 3)] =
            accum[i][j];
      }
    }
  }
}

extern "C" __declspec(dllexport) void run(float* a, float* b, float* c, int M,
                                          int N, int K, float* ms_out,
                                          int warmup_runs, int bench_runs) {
  dim3 grid((N + EOBN - 1) / EOBN, (M + EOBM - 1) / EOBM);
  dim3 block(TOBN, TOBM);
  std::vector<float> ms(bench_runs, 0.0f);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int i = 0; i < warmup_runs; ++i) {
    gemm_kernel<<<grid, block>>>(a, b, c, M, N, K);
  }
  cudaDeviceSynchronize();

  for (int i = 0; i < bench_runs; ++i) {
    cudaEventRecord(start);
    gemm_kernel<<<grid, block>>>(a, b, c, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms[i], start, stop);
  }
  std::nth_element(ms.begin(), ms.begin() + bench_runs / 2, ms.end());
  if (bench_runs % 2 == 0) {
    std::nth_element(ms.begin(), ms.begin() + bench_runs / 2 - 1,
                     ms.begin() + bench_runs / 2);
    *ms_out = (ms[bench_runs / 2 - 1] + ms[bench_runs / 2]) / 2;
  } else {
    *ms_out = ms[bench_runs / 2];
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
