"""Part 1: GHZ statevector benchmark, CPU (qpp-cpu) vs GPU (nvidia, fp32).

Usage::

    python src/ghz_benchmark.py                 # n = 20..30, 1 warm-up + 5 timed runs
    python src/ghz_benchmark.py --quick         # small CPU-only smoke run (< 2 min)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

warnings.filterwarnings("ignore", category=FutureWarning)

import cudaq  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hw import (  # noqa: E402
    available_ram_bytes,
    bytes_to_gib,
    detect_gpu,
    estimate_statevector_bytes,
    fits_in_memory,
    gpu_target_usable,
    hardware_summary,
)

CSV_FIELDS = ["target", "n", "median_s", "std_s", "est_mem_gb"]


@cudaq.kernel
def ghz(n: int):
    """Prepare an n-qubit GHZ state (|0..0> + |1..1>)/sqrt(2) and measure all qubits."""
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])
    mz(q)


def check_ghz_correctness(n: int = 8, shots: int = 4000, tol: float = 0.05) -> dict[str, int]:
    """Sample a GHZ state and assert only all-0 / all-1 appear, each near 50%.

    Returns the measured counts. Raises ``AssertionError`` on failure.
    """
    counts = dict(cudaq.sample(ghz, n, shots_count=shots).items())
    expected = {"0" * n, "1" * n}
    assert set(counts) <= expected, f"unexpected bitstrings: {set(counts) - expected}"
    for bits in expected:
        frac = counts.get(bits, 0) / shots
        assert abs(frac - 0.5) <= tol, f"{bits} observed {frac:.3f}, expected 0.5 +/- {tol}"
    return counts


def time_ghz(n: int, warmup: int, runs: int) -> tuple[float, float]:
    """Time ``cudaq.sample(ghz, n)`` (1 shot). Returns (median_s, std_s)."""
    for _ in range(warmup):
        cudaq.sample(ghz, n, shots_count=1)
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        cudaq.sample(ghz, n, shots_count=1)
        times.append(time.perf_counter() - t0)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return statistics.median(times), std


def benchmark_target(
    target: str,
    precision: str,
    qubit_range: range,
    available_bytes: Optional[int],
    warmup: int,
    runs: int,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Benchmark one target over ``qubit_range``, skipping sizes that do not fit."""
    rows: list[dict] = []
    for n in qubit_range:
        est = estimate_statevector_bytes(n, precision)
        if not fits_in_memory(n, precision, available_bytes):
            log(f"  [{target}] n={n}: SKIP (needs {bytes_to_gib(est):.1f} GiB, "
                f"have {bytes_to_gib(available_bytes or 0):.1f} GiB)")
            continue
        median, std = time_ghz(n, warmup, runs)
        log(f"  [{target}] n={n}: median {median:.4f}s  std {std:.4f}s  ({bytes_to_gib(est):.2f} GiB)")
        rows.append({"target": target, "n": n, "median_s": median, "std_s": std,
                     "est_mem_gb": bytes_to_gib(est)})
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Write benchmark rows to CSV."""
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    """Read a benchmark CSV back into typed rows."""
    with path.open() as f:
        return [{"target": r["target"], "n": int(r["n"]), "median_s": float(r["median_s"]),
                 "std_s": float(r["std_s"]), "est_mem_gb": float(r["est_mem_gb"])}
                for r in csv.DictReader(f)]


def speedups(rows: list[dict]) -> list[tuple[int, float]]:
    """CPU/GPU median-time ratio for every n measured on both targets."""
    cpu = {r["n"]: r["median_s"] for r in rows if r["target"] == "qpp-cpu"}
    gpu = {r["n"]: r["median_s"] for r in rows if r["target"] == "nvidia"}
    return [(n, cpu[n] / gpu[n]) for n in sorted(cpu.keys() & gpu.keys())]


def plot_runtime(rows: list[dict], path: Path) -> None:
    """Runtime vs qubit count (log y-axis), one line per target."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    styles = {"qpp-cpu": ("CPU (qpp-cpu, fp64)", "#1f77b4", "o"),
              "nvidia": ("GPU (nvidia, fp32)", "#76b900", "s")}
    for target, (label, color, marker) in styles.items():
        sub = sorted((r for r in rows if r["target"] == target), key=lambda r: r["n"])
        if not sub:
            continue
        ax.errorbar([r["n"] for r in sub], [r["median_s"] for r in sub],
                    yerr=[r["std_s"] for r in sub], label=label, color=color,
                    marker=marker, capsize=3, lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits $n$")
    ax.set_ylabel("Median runtime per run (s)")
    ax.set_title("GHZ circuit statevector simulation: runtime vs. qubits")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_speedup(rows: list[dict], path: Path) -> bool:
    """Plot CPU/GPU speedup vs n. Returns False (and writes nothing) if no GPU data."""
    sp = speedups(rows)
    if not sp:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    ax.plot([n for n, _ in sp], [s for _, s in sp], marker="o", color="#76b900", lw=1.8)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits $n$")
    ax.set_ylabel("Speedup (CPU time / GPU time)")
    ax.set_title("GPU speedup over CPU: GHZ statevector simulation")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--min-qubits", type=int, default=20)
    p.add_argument("--max-qubits", type=int, default=30)
    p.add_argument("--cpu-max-qubits", type=int, default=None,
                   help="optional lower cap for the (slow) CPU target")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--skip-gpu", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="n=10..16 step 2, 3 timed runs (CPU, < 2 min)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: results/ (results/quick with --quick)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.min_qubits, args.max_qubits, args.step, args.runs = 10, 16, 2, 3
    out_dir = args.out_dir or Path("results/quick" if args.quick else "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    hw = hardware_summary()
    (out_dir / "hardware_ghz.json").write_text(json.dumps(hw, indent=2))
    print("Hardware:", json.dumps(hw))

    plan = [("qpp-cpu", "fp64", None, available_ram_bytes(),
             args.cpu_max_qubits or args.max_qubits)]
    gpu = detect_gpu()
    if args.skip_gpu:
        print("GPU target skipped (--skip-gpu).")
    elif gpu is None:
        print("NOTE: no NVIDIA GPU detected -> skipping the 'nvidia' target. "
              "CPU results and plots are still produced. For the GPU half, run "
              "notebooks/colab_quickstart.ipynb on a free Colab T4.")
    elif not gpu_target_usable():
        print("NOTE: GPU found but this CUDA-Q build has no 'nvidia' target -> skipping GPU.")
    else:
        plan.append(("nvidia", "fp32", "fp32", gpu.vram_free_bytes, args.max_qubits))

    rows: list[dict] = []
    for target, precision, option, avail, max_n in plan:
        print(f"\n== Target {target} ({precision}) ==")
        cudaq.set_target(target, option=option) if option else cudaq.set_target(target)
        counts = check_ghz_correctness()
        print(f"  correctness OK (n=8 counts: {counts})")
        rows += benchmark_target(target, precision, range(args.min_qubits, max_n + 1, args.step),
                                 avail, args.warmup, args.runs)

    if not rows:
        print("No benchmark rows produced (everything skipped for memory?).")
        return 1
    write_csv(rows, out_dir / "ghz_benchmark.csv")
    plot_runtime(rows, out_dir / "ghz_benchmark.png")
    if plot_speedup(rows, out_dir / "ghz_speedup.png"):
        print("Wrote ghz_speedup.png")
    else:
        print("ghz_speedup.png NOT written: needs both CPU and GPU data.")
    print(f"\nResults written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
