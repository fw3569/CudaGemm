#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
using namespace nvcuda;
#define EOBM 128
#define EOBN 128
#define STRIDE_K 32
#define EOIM 16
#define EOIN 8
#define EOIK 16
#define EOFA (EOIM * EOIK / 32)
#define EOFB (EOIK * EOIN / 32)
#define EOFC (EOIM * EOIN / 32)
#define IOWM 8
#define IOWN 4
#define WOBM (EOBM / EOIM / IOWM)
#define WOBN (EOBN / EOIN / IOWN)
#define WARPS (WOBM * WOBN)
#define THREADS (WARPS * 32)
#define STAGES 3
#define SMEM_A_STRIDE (STRIDE_K + 8)
#define SMEM_B_STRIDE (EOBN + 8)
#define SMEM_C_STRIDE (WARPS * 32 + 8)
#define IOCN (32 / EOIN)
#define USE_SEME_EPILOGUE (IOWN >= IOCN)
#define ENABLE_L2_PREFETCH true
#define BLOCK_SWIZZLE_STRIDE 1024
struct GemmSmem {
  __align__(128) __half A[STAGES][EOBM][SMEM_A_STRIDE];
  __align__(128) __half B[STAGES][STRIDE_K][SMEM_B_STRIDE];
};
__device__ __forceinline__ void cp_async_4b(const void* __restrict__ smem_ptr,
                                            const void* __restrict__ gmem_ptr,
                                            int src_size = 4) {
  unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
  asm volatile(
#if ENABLE_L2_PREFETCH
      "cp.async.ca.shared.global.L2::128B [%0], [%1], 4, %2;\n"
#else
      "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
#endif
      ::"r"(smem_addr),
      "l"((uint64_t)gmem_ptr), "r"(src_size));
}
__device__ __forceinline__ void cp_async_16b(const void* __restrict__ smem_ptr,
                                             const void* __restrict__ gmem_ptr,
                                             int src_size) {
  unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
  asm volatile(
#if ENABLE_L2_PREFETCH
      "cp.async.ca.shared.global.L2::128B [%0], [%1], 16, %2;\n"
#else
      "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
#endif
      ::"r"(smem_addr),
      "l"((uint64_t)gmem_ptr), "r"(src_size));
}
__device__ __forceinline__ void cp_async_16b(
    const void* __restrict__ smem_ptr, const void* __restrict__ gmem_ptr) {
  unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
  asm volatile(
#if ENABLE_L2_PREFETCH
      "cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"
#else
      "cp.async.ca.shared.global [%0], [%1], 16;\n"
#endif
      ::"r"(smem_addr),
      "l"((uint64_t)gmem_ptr));
}
__device__ __forceinline__ void cp_async_16b_cg(
    const void* __restrict__ smem_ptr, const void* __restrict__ gmem_ptr,
    int src_size) {
  unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
  asm volatile(
#if ENABLE_L2_PREFETCH
      "cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
#else
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
#endif
      ::"r"(smem_addr),
      "l"((uint64_t)gmem_ptr), "r"(src_size));
}
__device__ __forceinline__ void cp_async_16b_cg(
    const void* __restrict__ smem_ptr, const void* __restrict__ gmem_ptr) {
  unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
  asm volatile(
#if ENABLE_L2_PREFETCH
      "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
#else
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
#endif
      ::"r"(smem_addr),
      "l"((uint64_t)gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm(int2& D, void const* __restrict__ ptr) {
  uint32_t addr = __cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%0, %1}, [%2];"
               : "=r"(D.x), "=r"(D.y)
               : "r"(addr));
}
__device__ __forceinline__ void ldsm(int4& D, void const* __restrict__ ptr) {
  uint32_t addr = __cvta_generic_to_shared(ptr);
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(D.x), "=r"(D.y), "=r"(D.z), "=r"(D.w)
      : "r"(addr));
}
template <class Frag>
__device__ __forceinline__ void load_matrix_a(Frag& frag,
                                              const __half* __restrict__ ptr,
                                              int stride, int lane_id) {
  ldsm(reinterpret_cast<int4&>(frag),
       ptr + (lane_id % 16) * stride + lane_id / 16 * 8);
}
template <class Frag>
__device__ __forceinline__ void load_matrix_b(Frag& frag,
                                              const __half* __restrict__ ptr,
                                              int stride, int lane_id) {
  ldsm(reinterpret_cast<int2&>(frag), ptr + lane_id * stride);
}
template <class FragA, class FragB, class FragC>
__device__ __forceinline__ void mma(FragC& frag_d, FragA& frag_a, FragB& frag_b,
                                    FragC& frag_c) {
  auto& u_frag_a = reinterpret_cast<uint32_t (&)[]>(frag_a);
  auto& u_frag_b = reinterpret_cast<uint32_t (&)[]>(frag_b);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(frag_d[0]), "=f"(frag_d[1]), "=f"(frag_d[2]), "=f"(frag_d[3])
      : "r"(u_frag_a[0]), "r"(u_frag_a[1]), "r"(u_frag_a[2]), "r"(u_frag_a[3]),
        "r"(u_frag_b[0]), "r"(u_frag_b[1]), "f"(frag_c[0]), "f"(frag_c[1]),
        "f"(frag_c[2]), "f"(frag_c[3]));
}
template <class Frag>
__device__ __forceinline__ void store_matrix_c(__half* __restrict__ ptr,
                                               Frag& frag, int stride,
                                               int lane_id, int m, int n) {
  if (lane_id / 4 < m && (lane_id % 4) * 2 < n) {
    ptr[(lane_id / 4) * stride + (lane_id % 4) * 2] = __half(frag[0]);
  }
  if (lane_id / 4 < m && (lane_id % 4) * 2 + 1 < n) {
    ptr[(lane_id / 4) * stride + (lane_id % 4) * 2 + 1] = __half(frag[1]);
  }
  if (lane_id / 4 + 8 < m && (lane_id % 4) * 2 < n) {
    ptr[(lane_id / 4 + 8) * stride + (lane_id % 4) * 2] = __half(frag[2]);
  }
  if (lane_id / 4 + 8 < m && (lane_id % 4) * 2 + 1 < n) {
    ptr[(lane_id / 4 + 8) * stride + (lane_id % 4) * 2 + 1] = __half(frag[3]);
  }
}
template <class Frag>
__device__ __forceinline__ void store_matrix_c(__half* __restrict__ ptr,
                                               Frag& frag, int stride,
                                               int lane_id) {
  ptr[(lane_id / 4) * stride + (lane_id % 4) * 2] = __half(frag[0]);
  ptr[(lane_id / 4) * stride + (lane_id % 4) * 2 + 1] = __half(frag[1]);
  ptr[(lane_id / 4 + 8) * stride + (lane_id % 4) * 2] = __half(frag[2]);
  ptr[(lane_id / 4 + 8) * stride + (lane_id % 4) * 2 + 1] = __half(frag[3]);
}
template <bool aligned_block, bool aligned_16>
__device__ __forceinline__ void load_A_tile(GemmSmem& smem, int stage,
                                            const __half* __restrict__ A,
                                            int m_base, int k_base, int M,
                                            int K) {
  constexpr int chunks = EOBM * STRIDE_K / 8;
#pragma unroll
  for (int c = 0; c < chunks; c += THREADS) {
    int bm = (c + threadIdx.x) / (STRIDE_K / 8);
    int bk = ((c + threadIdx.x) % (STRIDE_K / 8)) * 8;
    int gm = m_base + bm;
    int gk = k_base + bk;
    __half* dst = &smem.A[stage][bm][bk];
    if constexpr (aligned_block) {
      cp_async_16b(dst, A + gm * K + gk);
    } else if constexpr (aligned_16) {
      cp_async_16b(dst, A + gm * K + gk, ((gm < M) & (gk < K)) ? 16 : 0);
    } else {
      for (int i = 0; i < 8; ++i)
        dst[i] = A[min(gm * K + gk + i, M * K - 1)] *
                 __half((gm < M) & (K > gk - i));
    }
  }
}

template <bool aligned_block, bool aligned_16>
__device__ __forceinline__ void load_B_tile(GemmSmem& smem, int stage,
                                            const __half* __restrict__ B,
                                            int k_base, int n_base, int K,
                                            int N) {
  constexpr int chunks = STRIDE_K * EOBN / 8;
#pragma unroll
  for (int c = 0; c < chunks; c += THREADS) {
    int bk = (c + threadIdx.x) / (EOBN / 8);
    int bn = ((c + threadIdx.x) % (EOBN / 8)) * 8;
    int gk = k_base + bk;
    int gn = n_base + bn;
    __half* dst = &smem.B[stage][bk][bn];
    if constexpr (aligned_block) {
      cp_async_16b_cg(dst, B + gk * N + gn);
    } else if constexpr (aligned_16) {
      cp_async_16b_cg(dst, B + gk * N + gn, ((gk < K) & (gn < N)) ? 16 : 0);
    } else {
      for (int i = 0; i < 8; ++i)
        dst[i] = B[min(gk * N + gn + i, N * K - 1)] *
                 __half((gk < K) & (N > gn - i));
    }
  }
}

template <bool aligned_block, bool aligned_16>
__global__ __launch_bounds__(THREADS, 2) void gemm_tf32_kernel(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half* __restrict__ C, int M, int N, int K) {
  extern __shared__ __align__(128) char smem_raw[];
  GemmSmem& smem = *reinterpret_cast<GemmSmem*>(smem_raw);
  const int wid = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int wm = wid / WOBN;
  const int wn = wid % WOBN;
  const int bm =
      ((blockIdx.z & 1) ? (gridDim.y - blockIdx.y - 1) : blockIdx.y) * EOBM;
  const int bn = (blockIdx.z * gridDim.x + blockIdx.x) * EOBN;
  float acc[IOWM][IOWN][EOFC] = {};
  __half frag_a[2][IOWM][EOFA];
  __half frag_b[2][IOWN][EOFB];
  int num_k_tiles = (K + STRIDE_K - 1) / STRIDE_K;
  int stage = 0;
  load_A_tile<aligned_block, aligned_16>(smem, 0, A, bm, 0, M, K);
  load_B_tile<aligned_block, aligned_16>(smem, 0, B, 0, bn, K, N);
  cp_async_commit();
  for (int k_tile = 1; k_tile <= num_k_tiles; ++k_tile) {
    int nxt = (stage + 1) % STAGES;
    if (k_tile < num_k_tiles) {
      load_A_tile<aligned_block, aligned_16>(smem, nxt, A, bm,
                                             (k_tile)*STRIDE_K, M, K);
      load_B_tile<aligned_block, aligned_16>(smem, nxt, B, (k_tile)*STRIDE_K,
                                             bn, K, N);
    }
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    int reg_cache_idx = 0;
    {
#pragma unroll
      for (int wi = 0; wi < IOWM; ++wi) {
        int m_off = (wm * IOWM + wi) * EOIM;
        load_matrix_a(frag_a[reg_cache_idx][wi], &smem.A[stage][m_off][0],
                      SMEM_A_STRIDE, lane_id);
      }
#pragma unroll
      for (int wj = 0; wj < IOWN; ++wj) {
        int n_off = (wn * IOWN + wj) * EOIN;
        load_matrix_b(frag_b[reg_cache_idx][wj], &smem.B[stage][0][n_off],
                      SMEM_B_STRIDE, lane_id);
      }
    }
#pragma unroll
    for (int k = 0; k < STRIDE_K; k += EOIK) {
#pragma unroll
      for (int wi = 0; wi < IOWM; ++wi) {
#pragma unroll
        for (int wj = 0; wj < IOWN; ++wj) {
          mma(acc[wi][wj], frag_a[reg_cache_idx][wi], frag_b[reg_cache_idx][wj],
              acc[wi][wj]);
        }
      }
      reg_cache_idx ^= 1;
#pragma unroll
      for (int wi = 0; wi < IOWM; ++wi) {
        int m_off = (wm * IOWM + wi) * EOIM;
        load_matrix_a(frag_a[reg_cache_idx][wi],
                      &smem.A[stage][m_off][k + EOIK], SMEM_A_STRIDE, lane_id);
      }
#pragma unroll
      for (int wj = 0; wj < IOWN; ++wj) {
        int n_off = (wn * IOWN + wj) * EOIN;
        load_matrix_b(frag_b[reg_cache_idx][wj],
                      &smem.B[stage][k + EOIK][n_off], SMEM_B_STRIDE, lane_id);
      }
    }
    stage = nxt;
  }
  __syncthreads();

#if USE_SEME_EPILOGUE
  __half(&epilogue_smem)[EOIM][SMEM_C_STRIDE] =
      reinterpret_cast<__half(&)[EOIM][SMEM_C_STRIDE]>(smem_raw);
#pragma unroll
  for (int wi = 0; wi < IOWM; ++wi) {
#pragma unroll
    for (int wj_h = 0; wj_h < IOWN; wj_h += IOCN) {
#pragma unroll
      for (int wj_l = 0; wj_l < IOCN; ++wj_l) {
        store_matrix_c(
            &epilogue_smem[0][wid * 32 + ((wj_h + wj_l) * EOIN) % 32],
            acc[wi][(wj_h + wj_l)], SMEM_C_STRIDE, lane_id);
      }
#pragma unroll
      for (int i = 0; i < EOIM; ++i) {
        int gm = bm + i + (wm * IOWM + wi) * EOIM;
        int gn = bn + lane_id + (wn * IOWN + wj_h) * EOIN;
        if (gm >= M || gn >= N) continue;
        C[gm * N + gn] = epilogue_smem[i][threadIdx.x];
      }
    }
  }
#else
#pragma unroll
  for (int wi = 0; wi < IOWM; ++wi) {
#pragma unroll
    for (int wj = 0; wj < IOWN; ++wj) {
      int sm_row = (wm * IOWM + wi) * EOIM;
      int sm_col = (wn * IOWN + wj) * EOIN;
      int gm = bm + sm_row;
      int gn = bn + sm_col;
      store_matrix_c(&C[gm * N + gn], acc[wi][wj], N, lane_id, M - gm, N - gn);
    }
  }
#endif
}

template <bool aligned_block, bool aligned_16>
void run(const __half* d_A, const __half* d_B, __half* d_C, int m, int n, int k,
         float* ms_out, int warmup_runs, int bench_runs) {
  const int N_SWIZZLE = (n + (BLOCK_SWIZZLE_STRIDE)-1) / (BLOCK_SWIZZLE_STRIDE);
  dim3 block(THREADS);
  dim3 grid(((n + EOBN - 1) / EOBN + N_SWIZZLE - 1) / N_SWIZZLE,
            (m + EOBM - 1) / EOBM, N_SWIZZLE);
  size_t smem_pipeline = sizeof(GemmSmem);
#if USE_SEME_EPILOGUE
  size_t smem_epilogue = EOIM * SMEM_C_STRIDE * sizeof(__half);
  size_t smem_size =
      smem_epilogue > smem_pipeline ? smem_epilogue : smem_pipeline;
#else
  size_t smem_size = smem_pipeline;
#endif
  cudaFuncSetAttribute(gemm_tf32_kernel<aligned_block, aligned_16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  cudaFuncSetAttribute(gemm_tf32_kernel<aligned_block, aligned_16>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  std::vector<float> ms(bench_runs, 0.0f);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int i = 0; i < warmup_runs; ++i) {
    gemm_tf32_kernel<aligned_block, aligned_16>
        <<<grid, block, smem_size>>>(d_A, d_B, d_C, m, n, k);
  }
  cudaDeviceSynchronize();

  for (int i = 0; i < bench_runs; ++i) {
    cudaEventRecord(start);
    gemm_tf32_kernel<aligned_block, aligned_16>
        <<<grid, block, smem_size>>>(d_A, d_B, d_C, m, n, k);
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

extern "C" void run(const __half* d_A, const __half* d_B, __half* d_C, int m,
                    int n, int k, float* ms_out, int warmup_runs,
                    int bench_runs) {
  bool aligned_block =
      (m % EOBM == 0) && (n % EOBN == 0) && (k % STRIDE_K == 0);
  bool aligned_16 = (reinterpret_cast<uint64_t>(d_A) % 16 == 0) &&
                    (reinterpret_cast<uint64_t>(d_B) % 16 == 0) &&
                    (reinterpret_cast<uint64_t>(d_C) % 16 == 0) &&
                    ((sizeof(__half) * n) % 16 == 0) &&
                    ((sizeof(__half) * k) % 16 == 0);
  if (aligned_block) {
    if (aligned_16) {
      run<true, true>(d_A, d_B, d_C, m, n, k, ms_out, warmup_runs, bench_runs);
    } else {
      run<true, false>(d_A, d_B, d_C, m, n, k, ms_out, warmup_runs, bench_runs);
    }
  } else {
    if (aligned_16) {
      run<false, true>(d_A, d_B, d_C, m, n, k, ms_out, warmup_runs, bench_runs);
    } else {
      run<false, false>(d_A, d_B, d_C, m, n, k, ms_out, warmup_runs,
                        bench_runs);
    }
  }
}
