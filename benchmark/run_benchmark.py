"""
cuda_benchmark.py
-----------------
Compiles multiple .cu files, runs their kernels via ctypes, and
plots a performance comparison chart.

Requirements:
  - NVIDIA GPU + CUDA toolkit (nvcc)
  - Python packages: numpy, matplotlib, ctypes (stdlib)

  Install extras if missing:
    pip install numpy matplotlib
"""

import ctypes
import subprocess
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend – works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────
# 1. Kernel registry
# ─────────────────────────────────────────────
KERNELS = [
    {
        "label": "baseline\n(cublas)",
        "short":  "baseline",
        "src":    "baseline.cu",
        "fn":     "run",
        "color":  "#22c55e",
    },
    {
        "label": "v1\n(native)",
        "short":  "v1",
        "src":    "v1.cu",
        "fn":     "run",
        "color":  "#ef4444",
    },
    {
        "label": "v2\n(tiled, coalesced access)",
        "short":  "v2",
        "src":    "v2.cu",
        "fn":     "run",
        "color":  "#3b82f6",
    },
    {
        "label": "v3\n(float4, register limit, swizzle)",
        "short":  "v3",
        "src":    "v3.cu",
        "fn":     "run",
        "color":  "#ffffff",
    },
]

# Array sizes to benchmark (must be divisible by 4 for vec4 kernel)
SIZES = [1 << k for k in range(0, 13)]
WARMUP_RUNS = 20
BENCH_RUNS  = 100
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 2. Compilation helpers
# ─────────────────────────────────────────────

def compile_kernel(src_name: str, out_so: str) -> bool:
    """Compile a .cu file to a shared library. Returns True on success."""
    src = os.path.join(SCRIPT_DIR, src_name)
    cmd = [
        "nvcc", "-arch=sm_75", "-ccbin", "D:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.44.35207\\bin\\Hostx64\\x64\\cl.exe", "-O3", "--shared", "-lcublas", "-Xcompiler", "/LD,/O2", src, "-o", out_so
    ]
    print(f"  Compiling {src_name} …", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FAILED")
        print(result.stderr)
        return False
    print("OK")
    return True


def load_library(so_path: str):
    lib = ctypes.CDLL(so_path)
    return lib


def setup_kernel(lib, fn_name: str):
    """Set ctypes argument / return types for the kernel wrapper function."""
    fn = getattr(lib, fn_name)
    fn.restype  = None
    fn.argtypes = [
        ctypes.c_void_p,   # d_A
        ctypes.c_void_p,   # d_B
        ctypes.c_void_p,   # d_C
        ctypes.c_int,      # n
        ctypes.c_int,      # m
        ctypes.c_int,      # k
        ctypes.POINTER(ctypes.c_float),  # ms_out
        ctypes.c_int,      # warmup_runs
        ctypes.c_int,      # bench_runs
    ]
    return fn

# ─────────────────────────────────────────────
# 3. CUDA memory helpers via libcuda / libcudart
# ─────────────────────────────────────────────
cuda_bin_path = r"D:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin"
os.add_dll_directory(cuda_bin_path)
try:
    cudart = ctypes.CDLL("cudart64_12.dll")
except OSError as e:
    print(f"failed to load cudart: {e}")
    sys.exit(0)

cudart.cudaMemcpy.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int
]
cudart.cudaMemcpy.restype = ctypes.c_int
def cuda_malloc(nbytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    cudart.cudaMalloc(ctypes.byref(ptr), nbytes)
    return ptr

def cuda_free(ptr):
    cudart.cudaFree(ptr)

def cuda_memcpy_h2d(dst, src_np: np.ndarray):
    cudart.cudaMemcpy(dst, src_np.ctypes.data, src_np.nbytes, ctypes.c_int(1))  # H2D=1

def cuda_memset(ptr, val: int, nbytes: int):
    cudart.cudaMemset(ptr, val, nbytes)

# ─────────────────────────────────────────────
# 4. Benchmark a single kernel at one size
# ─────────────────────────────────────────────

def benchmark(fn, d_A, d_B, d_C, n: int, m: int, k: int) -> float:
    ms_val = ctypes.c_float(0.0)
    fn(d_A, d_B, d_C, n, m, k, ctypes.byref(ms_val), WARMUP_RUNS, BENCH_RUNS)
    return ms_val.value

# ─────────────────────────────────────────────
# 5. Main: compile → run → collect results
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CUDA Kernel Benchmark")
    print("=" * 60)

    loaded = []
    if not os.path.exists(SCRIPT_DIR+"\\dll"):
        os.makedirs(SCRIPT_DIR+"\\dll")

    # Compile all kernels
    print("\n[1/3] Compiling kernels …")
    for k in KERNELS:
        out = os.path.join(SCRIPT_DIR+"\\dll", f"{k['short']}.dll")
        if not compile_kernel(k["src"], out):
            print(f"Skipping {k['label']} due to compilation error.")
            loaded.append(None)
            continue
        lib = load_library(out)
        fn  = setup_kernel(lib, k["fn"])
        loaded.append(fn)

    # Allocate maximum-size GPU buffers once
    print("\n[2/3] Allocating GPU memory …")
    n_max   = max(SIZES)
    nbytes  = n_max * n_max * 4   # float32
    rng = np.random.default_rng(42)
    h_A = rng.random(n_max * n_max, dtype=np.float32)
    h_B = rng.random(n_max * n_max, dtype=np.float32)

    d_A = cuda_malloc(nbytes)
    d_B = cuda_malloc(nbytes)
    d_C = cuda_malloc(nbytes)
    cuda_memcpy_h2d(d_A, h_A)
    cuda_memcpy_h2d(d_B, h_B)

    # Benchmark
    print("\n[3/3] Running benchmarks …\n")
    results = {k["short"]: [] for k in KERNELS}

    for n in SIZES:
        cuda_memset(d_C, 0, n * n * 4)
        row = f"  n={n:>10,}"
        for k, fn in zip(KERNELS, loaded):
            if fn is None:
                results[k["short"]].append(float("nan"))
                row += f"  {k['short']:>8}: N/A"
                continue
            ms = benchmark(fn, d_A, d_B, d_C, n, n, n)
            tflops = 2 * n * n * n / (ms / 1e3) / 1e12
            results[k["short"]].append(ms)
            row += f"  {k['short']:>8}: {ms:6.3f} ms ({tflops:5.3f} TFLOPS)"
        print(row)

    cuda_free(d_A)
    cuda_free(d_B)
    cuda_free(d_C)

    plot_results(results)

# ─────────────────────────────────────────────
# 6. Plotting
# ─────────────────────────────────────────────

def plot_results(results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#f1f5f9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    x_labels = [f"{n}" if n < 1024 else f"{n//1024}K"
                 for n in SIZES]
    x = np.arange(len(SIZES))

    # ── Left: Latency ──────────────────────────────────────────────
    ax = axes[0]
    for k in KERNELS:
        y = np.array(results[k["short"]], dtype=float)
        ax.plot(x, y, color=k["color"], marker="o", markersize=4,
                linewidth=2, label=k["label"].replace("\n", " "))
        for i in range(-3,0):
            point_x = x[i]
            point_y = y[i]
            ax.annotate(
                f"{point_y:.2f}",
                xy=(point_x, point_y),
                xytext=(5, 5),
                textcoords="offset points",
                color=k["color"],
                fontsize=8,
                fontweight="bold"
            )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Matrix size (elements)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Kernel Latency vs Matrix Size", pad=12, fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, color="#334155", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(framealpha=0.15, labelcolor="#f1f5f9", fontsize=8.5,
              loc="upper left", facecolor="#1e293b", edgecolor="#475569")

    # ── Right: Performance (TFLOPS) ────────────────────────────────
    ax = axes[1]
    for k in KERNELS:
        y_ms = np.array(results[k["short"]], dtype=float)
        tflops = np.array([2 * n * n * n / 1e12 for n in SIZES])
        tflops = tflops / (y_ms / 1e3)
        ax.plot(x, tflops, color=k["color"], marker="s", markersize=4,
                linewidth=2, label=k["label"].replace("\n", " "))
        for i in range(-3,0):
            point_x = x[i]
            point_y = tflops[i]
            ax.annotate(
                f"{point_y:.3f}",
                xy=(point_x, point_y),
                xytext=(5, 5),
                textcoords="offset points",
                color=k["color"],
                fontsize=8,
                fontweight="bold"
            )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Matrix size (elements)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title("Performance vs Matrix Size", pad=12, fontsize=12, fontweight="bold")
    ax.grid(True, color="#334155", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(framealpha=0.15, labelcolor="#f1f5f9", fontsize=8.5,
              loc="upper left", facecolor="#1e293b", edgecolor="#475569")

    title = "CUDA GEMM Kernel Comparison"
    fig.suptitle(title, color="#f8fafc", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = os.path.join(SCRIPT_DIR, "benchmark_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nChart saved → {out}")


if __name__ == "__main__":
    main()