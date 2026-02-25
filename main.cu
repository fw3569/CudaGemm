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
#define THREAD_SIZE 16
#define LOCAL_SIZE 8
#define VALUE_MAX 100.0f

// min 2 blocks per SM to hide latency while balancing register usage
__global__ void __launch_bounds__(THREAD_SIZE* THREAD_SIZE, 2)
    gemm(float* a, float* b, float* c, int N, int M, int K) {
  // align to cache line
  __shared__ alignas(128) float sa[THREAD_SIZE * LOCAL_SIZE][THREAD_SIZE];
  __shared__ alignas(128) float sb[THREAD_SIZE][THREAD_SIZE * LOCAL_SIZE];
  int base_row = blockIdx.y * THREAD_SIZE * LOCAL_SIZE;
  int base_col = blockIdx.x * THREAD_SIZE * LOCAL_SIZE;
  float ans[LOCAL_SIZE][LOCAL_SIZE];
  memset(ans, 0, sizeof(ans));
  for (int kh = 0; kh < K; kh += THREAD_SIZE) {
    // copy to shared memory
    for (int i = 0; i < LOCAL_SIZE; i += 4) {
      // linear indexing to enable coalesced access, and avoid bank conflict
      int col = threadIdx.y * THREAD_SIZE + threadIdx.x +
                i * THREAD_SIZE * THREAD_SIZE;
      int row = col / (THREAD_SIZE);
      // unroll to reuse row/col registers and maintain occupancy
      col %= THREAD_SIZE;
      sa[row][col] = (base_row + row < N && kh + col < K) *
                     (a[(base_row + row) * K + kh + col]);
      row += THREAD_SIZE;
      sa[row][col] = (base_row + row < N && kh + col < K) *
                     (a[(base_row + row) * K + kh + col]);
      row += THREAD_SIZE;
      sa[row][col] = (base_row + row < N && kh + col < K) *
                     (a[(base_row + row) * K + kh + col]);
      row += THREAD_SIZE;
      sa[row][col] = (base_row + row < N && kh + col < K) *
                     (a[(base_row + row) * K + kh + col]);
    }
    for (int i = 0; i < LOCAL_SIZE; i += 4) {
      // linear indexing to enable coalesced access, and avoid bank conflict
      int col = threadIdx.y * THREAD_SIZE + threadIdx.x +
                i * THREAD_SIZE * THREAD_SIZE;
      int row = col / (THREAD_SIZE * LOCAL_SIZE);
      // unroll to reuse row/col registers and maintain occupancy
      col %= THREAD_SIZE * LOCAL_SIZE;
      sb[row][col] = (kh + row < K && base_col + col < M) *
                     (b[(kh + row) * M + base_col + col]);
      row += THREAD_SIZE / LOCAL_SIZE;
      sb[row][col] = (kh + row < K && base_col + col < M) *
                     (b[(kh + row) * M + base_col + col]);
      row += THREAD_SIZE / LOCAL_SIZE;
      sb[row][col] = (kh + row < K && base_col + col < M) *
                     (b[(kh + row) * M + base_col + col]);
      row += THREAD_SIZE / LOCAL_SIZE;
      sb[row][col] = (kh + row < K && base_col + col < M) *
                     (b[(kh + row) * M + base_col + col]);
    }
    __syncthreads();

    // calculate
    float rega[LOCAL_SIZE] = {};
    float regb[LOCAL_SIZE] = {};
    // Outer-product based register blocking to maximize data reuse
    for (int kl = 0; kl < THREAD_SIZE; ++kl) {
      for (int i = 0; i < LOCAL_SIZE; ++i) {
        rega[i] = sa[threadIdx.y + i * THREAD_SIZE][kl];
      }
      for (int i = 0; i < LOCAL_SIZE; i += 4) {
        // float4 to reduce io instructions, deal with mio stall
        *(float4*)(regb + i) =
            *(float4*)(&sb[kl][threadIdx.x * 4 + THREAD_SIZE * i]);
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
    if (M % 4 == 0) {
      for (int j = 0; j < LOCAL_SIZE &&
                      base_col + threadIdx.x * 4 + THREAD_SIZE * j + 3 < M;
           j += 4) {
        // float4 to reduce io instructions, deal with mio stall
        *(float4*)(&c[(base_row + threadIdx.y + i * THREAD_SIZE) * M +
                      base_col + threadIdx.x * 4 + THREAD_SIZE * j]) =
            *(float4*)(ans[i] + j);
      }
    } else {
      // handling boundaries
      for (int j = 0; j < LOCAL_SIZE; j += 4) {
        for (int k = 0;
             k < 4 && base_col + threadIdx.x + j * THREAD_SIZE + k < M; ++k) {
          c[(base_row + threadIdx.y + i * THREAD_SIZE) * M + base_col +
            threadIdx.x * 4 + j * THREAD_SIZE + k] = ans[i][j + k];
        }
      }
    }
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
      if (!std::isfinite(c[i][j]) || !std::isfinite(ground_truth[i][j]) ||
          fabs(c[i][j] - ground_truth[i][j]) >=
              1e-4 * max(fabs(c[i][j]), fabs(ground_truth[i][j]))) {
        return 1;
      }
    }
  }
  return 0;
}

void generate_tset_data(float a[ROW_NUM][MID_NUM], float b[MID_NUM][COL_NUM]) {
  for (int i = 0; i < ROW_NUM; ++i) {
    for (int j = 0; j < MID_NUM; ++j) {
      a[i][j] = double(rand()) / RAND_MAX * VALUE_MAX - (VALUE_MAX / 2);
    }
  }
  for (int i = 0; i < MID_NUM; ++i) {
    for (int j = 0; j < COL_NUM; ++j) {
      b[i][j] = double(rand()) / RAND_MAX * VALUE_MAX - (VALUE_MAX / 2);
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
  srand(time(NULL));
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
  gemm<<<dim3((ROW_NUM + ((THREAD_SIZE * LOCAL_SIZE) - 1)) /
                  (THREAD_SIZE * LOCAL_SIZE),
              (COL_NUM + ((THREAD_SIZE * LOCAL_SIZE) - 1)) /
                  (THREAD_SIZE * LOCAL_SIZE)),
         dim3(THREAD_SIZE, THREAD_SIZE)>>>(dev_a, dev_b, dev_c, ROW_NUM,
                                           COL_NUM, MID_NUM);
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
  if (compare_result(c, ground_truth) == 0) {
    std::cout << "ok" << std::endl;
    for (int i = 0; i < 10; ++i) {
      gemm<<<dim3((ROW_NUM + ((THREAD_SIZE * LOCAL_SIZE) - 1)) /
                      (THREAD_SIZE * LOCAL_SIZE),
                  (COL_NUM + ((THREAD_SIZE * LOCAL_SIZE) - 1)) /
                      (THREAD_SIZE * LOCAL_SIZE)),
             dim3(THREAD_SIZE, THREAD_SIZE)>>>(dev_a, dev_b, dev_c, ROW_NUM,
                                               COL_NUM, MID_NUM);
    }
    time_ms_kernel = 0.0f;
    for (int i = 0; i < 100; ++i) {
      float time_ms;
      cudaEventRecord(start);
      gemm<<<dim3((ROW_NUM + ((THREAD_SIZE * LOCAL_SIZE) - 1)) /
                      (THREAD_SIZE * LOCAL_SIZE),
                  (COL_NUM + ((THREAD_SIZE * LOCAL_SIZE) - 1)) /
                      (THREAD_SIZE * LOCAL_SIZE)),
             dim3(THREAD_SIZE, THREAD_SIZE)>>>(dev_a, dev_b, dev_c, ROW_NUM,
                                               COL_NUM, MID_NUM);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time_ms, start, stop);
      time_ms_kernel += time_ms;
    }
    time_ms_kernel /= 100;
    for (int i = 0; i < 10; ++i) {
      cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, COL_NUM, ROW_NUM,
                  MID_NUM, &alpha, dev_b, COL_NUM, dev_a, MID_NUM, &beta, dev_c,
                  COL_NUM);
    }
    time_ms_cublas_kernel = 0.0f;
    for (int i = 0; i < 100; ++i) {
      float time_ms;
      cudaEventRecord(start);
      cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, COL_NUM, ROW_NUM,
                  MID_NUM, &alpha, dev_b, COL_NUM, dev_a, MID_NUM, &beta, dev_c,
                  COL_NUM);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time_ms, start, stop);
      time_ms_cublas_kernel += time_ms;
    }
    time_ms_cublas_kernel /= 100;
    float tflops =
        (2.0 * ROW_NUM * COL_NUM * MID_NUM) / (time_ms_kernel * 1e-3) / 1e12;
    std::cout << "kernel time : " << time_ms_kernel << "ms, total time: "
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
