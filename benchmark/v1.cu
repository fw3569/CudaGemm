#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <functional>
#include <iostream>
#include <random>
#include <vector>

#define TILE_SIZE 16

__global__ void gemm(float* a, float* b, float* c, int N, int M, int K) {
  __shared__ float sa[TILE_SIZE][TILE_SIZE + 1];
  __shared__ float sb[TILE_SIZE][TILE_SIZE + 1];
  int base_row = blockIdx.x * TILE_SIZE;
  int base_col = blockIdx.y * TILE_SIZE;
  float ans = 0.0f;
  for (int kh = 0; kh < K; kh += TILE_SIZE) {
    if (base_row + threadIdx.y < N && kh + threadIdx.x < K) {
      sa[threadIdx.y][threadIdx.x] =
          a[(base_row + threadIdx.y) * K + kh + threadIdx.x];
    } else {
      sa[threadIdx.y][threadIdx.x] = 0.0f;
    }
    if (base_col + threadIdx.x < M && kh + threadIdx.y < K) {
      sb[threadIdx.x][threadIdx.y] =
          b[(kh + threadIdx.y) * M + base_col + threadIdx.x];
    } else {
      sb[threadIdx.x][threadIdx.y] = 0.0f;
    }
    __syncthreads();

    for (int kl = 0; kl < TILE_SIZE; ++kl) {
      ans += sa[threadIdx.x][kl] * sb[threadIdx.y][kl];
    }
    __syncthreads();
  }
  if (base_row + threadIdx.x < N && base_col + threadIdx.y < M) {
    c[(base_row + threadIdx.x) * N + base_col + threadIdx.y] = ans;
  }
}

extern "C" __declspec(dllexport) void run(float* d_A, float* d_B, float* d_C,
                                          int n, int m, int k, float* ms_out,
                                          int warmup_runs, int bench_runs) {
  dim3 threads(TILE_SIZE, TILE_SIZE);
  dim3 blocks((n + (TILE_SIZE - 1)) / TILE_SIZE,
              (m + (TILE_SIZE - 1)) / TILE_SIZE);
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
