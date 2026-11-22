"""Render README.md from README.template.md using the numbers in results/*.csv|json.

Every figure in the README comes from the result files, never typed by hand.

Usage::  python src/report.py [--results-dir results] [--out README.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ghz_benchmark import read_csv, speedups
from h2_vqe import CHEM_ACC_MHA, read_pes_csv

ROOT = Path(__file__).resolve().parents[1]


def fmt_mha(x: float) -> str:
    """Format an error in mHa with sensible precision (tiny values in scientific notation)."""
    return f"{x:.1e}" if x < 0.01 else f"{x:.3f}"


def ghz_summary(rows: list[dict]) -> dict:
    """Headline GHZ numbers: largest n per target and speedup at the largest shared n."""
    out: dict = {}
    for target in ("qpp-cpu", "nvidia"):
        ns = [r["n"] for r in rows if r["target"] == target]
        out[target] = max(ns) if ns else None
    sp = speedups(rows)
    out["speedup_n"], out["speedup"] = (sp[-1] if sp else (None, None))
    out["max_qubits"] = max((n for n in (out["qpp-cpu"], out["nvidia"]) if n), default=None)
    return out


def ghz_table(rows: list[dict]) -> str:
    """Markdown table of median runtimes (and speedup where both targets exist)."""
    cpu = {r["n"]: r for r in rows if r["target"] == "qpp-cpu"}
    gpu = {r["n"]: r for r in rows if r["target"] == "nvidia"}
    has_gpu = bool(gpu)
    head = "| n | CPU median (s) | " + ("GPU median (s) | Speedup |" if has_gpu else "Est. memory (GiB) |")
    lines = [head, "|---|---|" + ("---|---|" if has_gpu else "---|")]
    for n in sorted(cpu.keys() | gpu.keys()):
        c = f"{cpu[n]['median_s']:.4f}" if n in cpu else "–"
        if has_gpu:
            g = f"{gpu[n]['median_s']:.4f}" if n in gpu else "–"
            s = f"{cpu[n]['median_s'] / gpu[n]['median_s']:.1f}×" if n in cpu and n in gpu else "–"
            lines.append(f"| {n} | {c} | {g} | {s} |")
        else:
            lines.append(f"| {n} | {c} | {cpu[n]['est_mem_gb']:.3f} |")
    return "\n".join(lines)


def hardware_md(hw: dict) -> str:
    gpu = hw.get("gpu")
    gpu_s = (f"{gpu['name']} ({gpu['vram_total_bytes'] / 1024**3:.0f} GiB VRAM)"
             if gpu else "none (CPU-only run)")
    return (f"- **CPU:** {hw['cpu']} ({hw['cpu_logical_cores']} logical cores)\n"
            f"- **RAM:** {hw['ram_total_gib']} GiB\n- **GPU:** {gpu_s}\n"
            f"- **Software:** {hw['cudaq']}, Python {hw['python']}, {hw['os']}")


def render(results_dir: Path, template: str) -> str:
    rows = read_csv(results_dir / "ghz_benchmark.csv")
    pes = read_pes_csv(results_dir / "h2_pes.csv")
    hw = json.loads((results_dir / "hardware_ghz.json").read_text())
    vqe = json.loads((results_dir / "h2_summary.json").read_text())
    g = ghz_summary(rows)
    max_err = max(p["abs_error_mHa"] for p in pes)
    ok = max_err < CHEM_ACC_MHA
    sp_n, sp = g["speedup_n"], g["speedup"]
    rel = results_dir.relative_to(ROOT) if results_dir.is_absolute() else results_dir

    if sp is not None:
        ghz_sentence = (f"GPU statevector simulation of GHZ circuits reaches {g['nvidia']} qubits, "
                        f"with a {sp:.1f}× speedup over the CPU at n = {sp_n}")
        resume_ghz = (f"GPU-accelerated statevector simulation up to {g['nvidia']} qubits with NVIDIA "
                      f"CUDA-Q/cuStateVec ({sp:.1f}× speedup over CPU at n = {sp_n})")
        gpu_note = ""
        speed_img = f"![GPU speedup]({rel}/ghz_speedup.png)\n"
    else:
        ghz_sentence = (f"CPU statevector simulation of GHZ circuits was measured up to {g['qpp-cpu']} "
                        "qubits on this machine (no NVIDIA GPU was available, so the GPU comparison "
                        "has not been measured yet)")
        resume_ghz = (f"Benchmarked CUDA-Q statevector simulation of GHZ circuits on CPU up to "
                      f"{g['qpp-cpu']} qubits (GPU speedup: not yet measured, see note below)")
        gpu_note = ("> **GPU results pending.** This run had no NVIDIA GPU, so the `nvidia` target and "
                    "the speedup plot were skipped. Run `notebooks/colab_quickstart.ipynb` on a free "
                    "Colab T4, copy its `results/` files here and run `make report` to fill in the GPU "
                    "numbers.\n")
        speed_img = ""

    vqe_sentence = (f"VQE with a UCCSD ansatz reproduces the H₂/STO-3G potential energy curve "
                    f"({vqe['n_points']} bond lengths) with a maximum error of {fmt_mha(max_err)} mHa "
                    f"versus exact FCI, " + ("inside" if ok else "OUTSIDE") +
                    f" chemical accuracy ({CHEM_ACC_MHA} mHa)")
    resume_vqe = (f"VQE with a UCCSD ansatz reproduces the H₂ potential energy curve within "
                  f"{fmt_mha(max_err)} mHa of FCI ({'chemical accuracy' if ok else 'not chemical accuracy'})")

    fit_v, fit_f = vqe["fit_min_vqe"], vqe["fit_min_fci"]
    values = {
        "SUMMARY": f"{ghz_sentence}. {vqe_sentence}.",
        "GPU_NOTE": gpu_note,
        "SPEEDUP_IMG": speed_img,
        "RESULTS_DIR": str(rel),
        "GHZ_TABLE": ghz_table(rows),
        "HARDWARE": hardware_md(hw),
        "VQE_RESULTS": (
            f"- Hamiltonian: {vqe['n_pauli_terms']} Pauli terms (incl. identity) on 4 qubits; "
            f"UCCSD has {vqe['n_parameters']} parameters; optimizer: {vqe['optimizer']}.\n"
            f"- At R = 0.735 Å: E_VQE = {vqe['at_0735']['E_vqe']:.6f} Ha, E_FCI = "
            f"{vqe['at_0735']['E_fci']:.6f} Ha, error {fmt_mha(vqe['at_0735']['abs_error_mHa'])} mHa "
            f"({vqe['at_0735']['n_evals']} energy evaluations); Hartree-Fock: "
            f"{vqe['at_0735']['E_hf']:.6f} Ha.\n"
            f"- Equilibrium from a 1D minimisation of the VQE energy: "
            f"R = {fit_v['R']:.3f} Å, E = {fit_v['E']:.6f} Ha (same search on FCI: "
            f"R = {fit_f['R']:.3f} Å, E = {fit_f['E']:.6f} Ha). Lowest grid point: "
            f"R = {vqe['grid_min_vqe']['R']:.2f} Å.\n"
            f"- Max |E_VQE − E_FCI| over the sweep: {fmt_mha(max_err)} mHa."),
        "RESUME": f"- {resume_ghz}; {resume_vqe}.",
    }
    text = template
    for key, val in values.items():
        text = text.replace("{{" + key + "}}", val)
    return text


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("README.md"))
    p.add_argument("--template", type=Path, default=ROOT / "README.template.md")
    args = p.parse_args(argv)
    args.out.write_text(render(args.results_dir, args.template.read_text()))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
