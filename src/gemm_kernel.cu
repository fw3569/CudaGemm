#include "gemm_kernel.cuh"

#define THREAD_SIZE 16
#define LOCAL_SIZE 8

// min 2 blocks per SM to hide latency while balancing register usage
__global__ void __launch_bounds__(THREAD_SIZE* THREAD_SIZE, 2)
    gemm_kernel(float* a, float* b, float* c, int N, int M, int K) {
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
      col &= THREAD_SIZE - 1;
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
      col &= THREAD_SIZE * LOCAL_SIZE - 1;
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
    if ((M & 0x11) == 0) {
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
             k < 4 && base_col + threadIdx.x * 4 + j * THREAD_SIZE + k < M;
             ++k) {
          c[(base_row + threadIdx.y + i * THREAD_SIZE) * M + base_col +
            threadIdx.x * 4 + j * THREAD_SIZE + k] = ans[i][j + k];
        }
      }
    }
  }
}

void gemm(float* a, float* b, float* c, int N, int M, int K) {
  gemm_kernel<<<
      dim3((M + ((THREAD_SIZE * LOCAL_SIZE) - 1)) / (THREAD_SIZE * LOCAL_SIZE),
           (N + ((THREAD_SIZE * LOCAL_SIZE) - 1)) / (THREAD_SIZE * LOCAL_SIZE)),
      dim3(THREAD_SIZE, THREAD_SIZE)>>>(a, b, c, N, M, K);
}
