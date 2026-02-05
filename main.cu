#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <functional>
#include <iostream>
#include <random>
#include <vector>

#define ROW_NUM 1024
#define COL_NUM 1024
#define MID_NUM 1024
#define TILE_SIZE 16
#define VALUE_MAX 1000

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

void gemm_cpu(float a[ROW_NUM][MID_NUM], float b[MID_NUM][COL_NUM], float* c) {
  for (int i = 0; i < ROW_NUM; ++i) {
    for (int j = 0; j < COL_NUM; ++j) {
      c[i * COL_NUM + j] = 0;
      for (int k = 0; k < MID_NUM; ++k) {
        c[i * COL_NUM + j] += a[i][k] * b[k][j];
      }
    }
  }
}

int compare_result(float c[ROW_NUM][COL_NUM],
                   float ground_truth[ROW_NUM][COL_NUM]) {
  for (int i = 0; i < ROW_NUM; ++i) {
    for (int j = 0; j < COL_NUM; ++j) {
      if (c[i][j] != ground_truth[i][j]) {
        return 1;
      }
    }
  }
  return 0;
}

void generate_tset_data(float a[ROW_NUM][MID_NUM], float b[MID_NUM][COL_NUM]) {
  for (int i = 0; i < ROW_NUM; ++i) {
    for (int j = 0; j < MID_NUM; ++j) {
      a[i][j] = double(rand()) / RAND_MAX * VALUE_MAX;
    }
  }
  for (int i = 0; i < MID_NUM; ++i) {
    for (int j = 0; j < COL_NUM; ++j) {
      b[i][j] = double(rand()) / RAND_MAX * VALUE_MAX;
    }
  }
}

class exit_guard {
 public:
  ~exit_guard() {
    for (std::function<void()> func : funcs) {
      func();
    }
  }
  void Register(std::function<void()> func) { funcs.emplace_back(func); }

 private:
  std::vector<std::function<void()>> funcs;
} global_exit_guard;

#define CHECK_CUDA_WITH_CLEANUP(STAMENT, CLEANUP_FUN)   \
  {                                                     \
    if ((STAMENT) == cudaSuccess) {                     \
      global_exit_guard.Register(CLEANUP_FUN);          \
    } else {                                            \
      std::cout << "exit in " << __LINE__ << std::endl; \
      exit(1);                                          \
    }                                                   \
  }

#define CHECK_CUDA(STAMENT)                             \
  {                                                     \
    if ((STAMENT) == cudaSuccess) {                     \
    } else {                                            \
      std::cout << "exit in " << __LINE__ << std::endl; \
      exit(1);                                          \
    }                                                   \
  }

float a[ROW_NUM][MID_NUM], b[MID_NUM][COL_NUM], c[ROW_NUM][COL_NUM],
    ground_truth[ROW_NUM][COL_NUM];
float *dev_a, *dev_b, *dev_c;

int main() {
  generate_tset_data(a, b);
  CHECK_CUDA_WITH_CLEANUP(cudaMalloc(&dev_a, sizeof(a)),
                          []() -> void { cudaFree(dev_a); });
  CHECK_CUDA_WITH_CLEANUP(cudaMalloc(&dev_b, sizeof(b)),
                          []() -> void { cudaFree(dev_b); });
  CHECK_CUDA_WITH_CLEANUP(cudaMalloc(&dev_c, sizeof(c)),
                          []() -> void { cudaFree(dev_c); });
  float time_ms_memcpy_in, time_ms_kernel, time_ms_memcpy_out;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  CHECK_CUDA(cudaMemcpy(dev_a, a, sizeof(a), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_b, b, sizeof(b), cudaMemcpyHostToDevice));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_memcpy_in, start, stop);
  cudaEventRecord(start);
  gemm<<<dim3((ROW_NUM + (TILE_SIZE - 1)) / TILE_SIZE,
              (COL_NUM + (TILE_SIZE - 1)) / TILE_SIZE),
         dim3(TILE_SIZE, TILE_SIZE)>>>(dev_a, dev_b, dev_c, ROW_NUM, COL_NUM,
                                       MID_NUM);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_kernel, start, stop);
  cudaEventRecord(start);
  CHECK_CUDA(cudaMemcpy(c, dev_c, sizeof(c), cudaMemcpyDeviceToHost));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_memcpy_out, start, stop);

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  float alpha = 1.0f;
  float beta = 0.0f;
  float time_ms_cublas_kernel;
  cudaEventRecord(start);
  cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, COL_NUM, ROW_NUM,
              MID_NUM, &alpha, dev_b, COL_NUM, dev_a, MID_NUM, &beta, dev_c,
              COL_NUM);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_cublas_kernel, start, stop);
  CHECK_CUDA(cudaMemcpy(ground_truth, dev_c, sizeof(ground_truth),
                        cudaMemcpyDeviceToHost));

  // cudaDeviceSynchronize();
  float tflops =
      (2.0 * ROW_NUM * COL_NUM * MID_NUM) / (time_ms_kernel * 1e-3) / 1e12;
  if (compare_result(c, ground_truth) == 0) {
    std::cout << "ok kernel time: " << time_ms_kernel << "ms, total time: "
              << time_ms_memcpy_in + time_ms_kernel + time_ms_memcpy_out
              << "ms, tflops: " << tflops << std::endl;
    std::cout << "cublas kernel time: " << time_ms_cublas_kernel
              << "ms, rate: " << time_ms_cublas_kernel / time_ms_kernel
              << std::endl;
  } else {
    std::cout << "failed" << std::endl;
  }
  return 0;
}