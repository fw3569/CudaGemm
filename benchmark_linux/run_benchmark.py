import ctypes
import subprocess
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

KERNELS = [
    {
        "label": "baseline\n(cublas)",
        "short":  "baseline",
        "src":    "baseline.cu",
        "fn":     "run",
        "color":  "#22c55e",
    },
    {
        "label": "custom_gemm\n(ampere fp16)",
        "short":  "v4",
        "src":    "v4.cu",
        "fn":     "run",
        "color":  "#ffffff",
    },
]

SIZES       = [1 << k for k in range(0, 14)]
WARMUP_RUNS = 20
BENCH_RUNS  = 100
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
VERIFY_N    = 1024
VERIFY_TOL  = 1e-1

def compile_kernel(src_name: str, out_so: str) -> bool:
    src = os.path.join(SCRIPT_DIR, src_name)
    cmd = [
        "nvcc", "-arch=sm_89", "-O3", "--shared", "-lcublas", "-Xcompiler", "-fPIC", src, "-o", out_so
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
    fn = getattr(lib, fn_name)
    fn.restype  = None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    return fn

try:
    cudart = ctypes.CDLL("libcudart.so")
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
    cudart.cudaMemcpy(dst, src_np.ctypes.data, src_np.nbytes, ctypes.c_int(1))

def cuda_memcpy_d2h(dst_np: np.ndarray, src):
    cudart.cudaMemcpy(dst_np.ctypes.data, src, dst_np.nbytes, ctypes.c_int(0))

def cuda_memset(ptr, val: int, nbytes: int):
    cudart.cudaMemset(ptr, val, nbytes)

def verify_kernel(fn, label: str, rng: np.random.Generator) -> bool:
    n      = VERIFY_N
    nbytes = n * n * 2
    h_A = rng.random(n * n).astype(np.float16)
    h_B = rng.random(n * n).astype(np.float16)
    d_A = cuda_malloc(nbytes)
    d_B = cuda_malloc(nbytes)
    d_C = cuda_malloc(nbytes)

    cuda_memcpy_h2d(d_A, h_A)
    cuda_memcpy_h2d(d_B, h_B)
    cuda_memset(d_C, 0, nbytes)

    ms_val = ctypes.c_float(0.0)
    fn(d_A, d_B, d_C, n, n, n, ctypes.byref(ms_val), 0, 1)
    h_C = np.zeros(n * n, dtype=np.float16)
    cuda_memcpy_d2h(h_C, d_C)

    cuda_free(d_A)
    cuda_free(d_B)
    cuda_free(d_C)
    A     = h_A.reshape(n, n).astype(np.float32)
    B     = h_B.reshape(n, n).astype(np.float32)
    C_ref = (A @ B).astype(np.float16)
    C_got = h_C.reshape(n, n)
    max_err = float(np.max(np.abs(C_got.astype(np.float32) - C_ref.astype(np.float32))))
    rel_err = max_err / (float(np.max(np.abs(C_ref.astype(np.float32)))) + 1e-9)
    passed  = rel_err < VERIFY_TOL

    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] {label:30s}  max_abs_err={max_err:.4e}  rel_err={rel_err:.4e}")
    return passed

def benchmark(fn, d_A, d_B, d_C, n: int, m: int, k: int) -> float:
    ms_val = ctypes.c_float(0.0)
    fn(d_A, d_B, d_C, n, m, k, ctypes.byref(ms_val), WARMUP_RUNS, BENCH_RUNS)
    return ms_val.value

def main():
    print("=" * 60)
    print("CUDA Kernel Benchmark")
    print("=" * 60)

    loaded = []
    if not os.path.exists(SCRIPT_DIR+"lib"):
        os.makedirs(SCRIPT_DIR+"lib")

    print("\n[1/4] Compiling kernels …")
    for k in KERNELS:
        out = os.path.join(SCRIPT_DIR+"lib", f"{k['short']}.so")
        if not compile_kernel(k["src"], out):
            print(f"Skipping {k['label']} due to compilation error.")
            loaded.append(None)
            continue
        lib = load_library(out)
        fn  = setup_kernel(lib, k["fn"])
        loaded.append(fn)

    print("\n[2/4] Verifying correctness (n={}) …".format(VERIFY_N))
    verify_rng = np.random.default_rng(0)
    verify_passed = []
    for k, fn in zip(KERNELS, loaded):
        if fn is None:
            print(f"    [SKIP] {k['label'].replace(chr(10), ' ')}")
            verify_passed.append(False)
            continue
        ok = verify_kernel(fn, k["label"].replace("\n", " "), verify_rng)
        verify_passed.append(ok)

    failed = [k["label"].replace("\n", " ") for k, ok in zip(KERNELS, verify_passed) if not ok and loaded[KERNELS.index(k)] is not None]
    if failed:
        print(f"\n  WARNING: correctness check FAILED for: {', '.join(failed)}")
        print("  These kernels will still be benchmarked but results may be meaningless.\n")

    print("\n[3/4] Allocating GPU memory …")
    n_max  = max(SIZES)
    nbytes = n_max * n_max * 2
    rng = np.random.default_rng(42)
    h_A = rng.random(n_max * n_max).astype(np.float16)
    h_B = rng.random(n_max * n_max).astype(np.float16)
    d_A = cuda_malloc(nbytes)
    d_B = cuda_malloc(nbytes)
    d_C = cuda_malloc(nbytes)
    cuda_memcpy_h2d(d_A, h_A)
    cuda_memcpy_h2d(d_B, h_B)

    print("\n[4/4] Running benchmarks …\n")
    results = {k["short"]: [] for k in KERNELS}

    for n in SIZES:
        cuda_memset(d_C, 0, n * n * 2)
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