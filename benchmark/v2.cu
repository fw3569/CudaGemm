#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <functional>
#include <iostream>
#include <random>
#include <vector>

#define THREAD_SIZE 16
#define LOCAL_SIZE 4

__global__ void gemm(float* a, float* b, float* c, int N, int M, int K) {
  __shared__ alignas(
      128) float sa[THREAD_SIZE * LOCAL_SIZE][THREAD_SIZE * LOCAL_SIZE];
  __shared__ alignas(
      128) float sb[THREAD_SIZE * LOCAL_SIZE][THREAD_SIZE * LOCAL_SIZE];
  int base_row = blockIdx.x * THREAD_SIZE * LOCAL_SIZE;
  int base_col = blockIdx.y * THREAD_SIZE * LOCAL_SIZE;
  float ans[LOCAL_SIZE][LOCAL_SIZE];
  memset(ans, 0, sizeof(ans));
  for (int kh = 0; kh < K; kh += THREAD_SIZE * LOCAL_SIZE) {
    for (int i = 0; i < LOCAL_SIZE * LOCAL_SIZE; ++i) {
      int col = threadIdx.y * THREAD_SIZE + threadIdx.x +
                i * THREAD_SIZE * THREAD_SIZE;
      int row = col / (THREAD_SIZE * LOCAL_SIZE);
      col %= THREAD_SIZE * LOCAL_SIZE;
      sa[row][col] = (base_row + row < N && kh + col < K) *
                     (a[(base_row + row) * K + kh + col]);
    }
    for (int i = 0; i < LOCAL_SIZE * LOCAL_SIZE; ++i) {
      int col = threadIdx.y * THREAD_SIZE + threadIdx.x +
                i * THREAD_SIZE * THREAD_SIZE;
      int row = col / (THREAD_SIZE * LOCAL_SIZE);
      col %= THREAD_SIZE * LOCAL_SIZE;
      sb[row][col] = (kh + row < K && base_col + col < M) *
                     (b[(kh + row) * M + base_col + col]);
    }
    __syncthreads();

    for (int kl = 0; kl < THREAD_SIZE * LOCAL_SIZE; ++kl) {
      float rega[LOCAL_SIZE];
      float regb[LOCAL_SIZE];
      for (int i = 0; i < LOCAL_SIZE; ++i) {
        rega[i] = sa[threadIdx.y + i * THREAD_SIZE][kl];
      }
      for (int i = 0; i < LOCAL_SIZE; ++i) {
        regb[i] = sb[kl][threadIdx.x + i * THREAD_SIZE];
      }
      for (int i = 0; i < LOCAL_SIZE; ++i) {
        for (int j = 0; j < LOCAL_SIZE; ++j) {
          ans[i][j] += rega[i] * regb[j];
        }
      }
    }
    __syncthreads();
  }
  for (int i = 0;
       i < LOCAL_SIZE && base_row + threadIdx.y + i * THREAD_SIZE < N; ++i) {
    for (int j = 0;
         j < LOCAL_SIZE && base_col + threadIdx.x + j * THREAD_SIZE < M; ++j) {
      c[(base_row + threadIdx.y + i * THREAD_SIZE) * M + base_col +
        threadIdx.x + j * THREAD_SIZE] = ans[i][j];
    }
  }
}

extern "C" __declspec(dllexport) void run(float* d_A, float* d_B, float* d_C,
                                          int n, int m, int k, float* ms_out,
                                          int warmup_runs, int bench_runs) {
  dim3 threads(THREAD_SIZE, THREAD_SIZE);
  dim3 blocks(
      (n + (THREAD_SIZE * LOCAL_SIZE - 1)) / (THREAD_SIZE * LOCAL_SIZE),
      (m + (THREAD_SIZE * LOCAL_SIZE - 1)) / (THREAD_SIZE * LOCAL_SIZE));
  std::vector<float> ms(bench_runs, 0.0f);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int i = 0; i < warmup_runs; ++i) {
    gemm<<<blocks, threads>>>(d_A, d_B, d_C, n, m, k);
  }

  for (int i = 0; i < bench_runs; ++i) {
    cudaEventRecord(start);
    gemm<<<blocks, threads>>>(d_A, d_B, d_C, n, m, k);
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
