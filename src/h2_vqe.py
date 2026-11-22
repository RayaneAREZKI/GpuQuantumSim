"""Part 2: VQE with a UCCSD ansatz on H2 (STO-3G), bond-length sweep vs exact FCI.

Usage::

    python src/h2_vqe.py            # full sweep R = 0.30..2.50 A, step 0.05
    python src/h2_vqe.py --quick    # reduced sweep (< 2 min on CPU)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

warnings.filterwarnings("ignore", category=FutureWarning)

import cudaq  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from cudaq import spin  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHEM_ACC_MHA = 1.6
HARTREE_TO_MHA = 1000.0
N_ELECTRONS, N_QUBITS = 2, 4
PES_FIELDS = ["R", "E_vqe", "E_fci", "E_hf", "abs_error_mHa"]


# --------------------------------------------------------------------------- Hamiltonian
def geometry(r_angstrom: float) -> list[tuple[str, tuple[float, float, float]]]:
    """H2 geometry along the z axis, bond length in Angstrom."""
    return [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, float(r_angstrom)))]


def qubit_operator_to_spin_operator(qop) -> "cudaq.SpinOperator":
    """Convert an OpenFermion ``QubitOperator`` into a ``cudaq.SpinOperator``."""
    paulis = {"X": spin.x, "Y": spin.y, "Z": spin.z}
    total = None
    for term, coeff in qop.terms.items():
        c = complex(coeff)
        if abs(c.imag) > 1e-10:
            raise ValueError(f"non-Hermitian term {term}: {c}")
        op = c.real * spin.i(0)
        for index, label in term:
            op = op * paulis[label](index)
        total = op if total is None else total + op
    return total


def hamiltonian_from_openfermion(r_angstrom: float) -> "cudaq.SpinOperator":
    """PySCF -> OpenFermion molecular Hamiltonian -> Jordan-Wigner -> SpinOperator."""
    from openfermion import get_fermion_operator, jordan_wigner
    from openfermion.chem import MolecularData
    from openfermionpyscf import run_pyscf

    mol = MolecularData(geometry(r_angstrom), "sto-3g", multiplicity=1, charge=0)
    mol = run_pyscf(mol, run_scf=True, run_fci=False)
    qop = jordan_wigner(get_fermion_operator(mol.get_molecular_hamiltonian()))
    return qubit_operator_to_spin_operator(qop)


def build_hamiltonian(r_angstrom: float, backend: str = "auto") -> "cudaq.SpinOperator":
    """Jordan-Wigner H2/STO-3G Hamiltonian (4 qubits) including nuclear repulsion.

    ``backend='cudaq'`` uses ``cudaq.chemistry.create_molecular_hamiltonian``;
    ``'openfermion'`` uses the PySCF+OpenFermion route; ``'auto'`` tries CUDA-Q
    first and falls back. (The CUDA-Q helper writes an .hdf5 cache file to the
    current directory, so it is run inside a temporary directory.)
    """
    has_helper = hasattr(getattr(cudaq, "chemistry", None), "create_molecular_hamiltonian")
    if backend in ("auto", "cudaq") and has_helper:
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                result = cudaq.chemistry.create_molecular_hamiltonian(
                    geometry(r_angstrom), "sto-3g", 1, 0)
            finally:
                os.chdir(cwd)
        return result[0] if isinstance(result, tuple) else result
    if backend == "cudaq":
        raise RuntimeError("cudaq.chemistry.create_molecular_hamiltonian not available")
    return hamiltonian_from_openfermion(r_angstrom)


# --------------------------------------------------------------------------- Ansatz + VQE
@cudaq.kernel
def uccsd_ansatz(thetas: list[float]):
    """UCCSD circuit for 2 electrons in 4 spin-orbitals, from the Hartree-Fock state."""
    q = cudaq.qvector(4)
    x(q[0])  # Hartree-Fock reference |1100>
    x(q[1])
    cudaq.kernels.uccsd(q, thetas, 2, 4)


def n_parameters() -> int:
    """Number of UCCSD variational parameters for (2 electrons, 4 qubits): 3."""
    return cudaq.kernels.uccsd_num_parameters(N_ELECTRONS, N_QUBITS)


def energy(hamiltonian, theta: Sequence[float]) -> float:
    """Exact (shot-noise-free) expectation value <psi(theta)|H|psi(theta)>."""
    return float(cudaq.observe(uccsd_ansatz, hamiltonian, [float(t) for t in theta]).expectation())


@dataclass
class VQEResult:
    energy: float
    params: np.ndarray
    history: list[float] = field(default_factory=list)  # energy at every evaluation

    @property
    def n_evals(self) -> int:
        return len(self.history)


def run_vqe(hamiltonian, x0: Optional[Sequence[float]] = None, optimizer: str = "COBYLA",
            maxiter: int = 200, rhobeg: float = 0.1, tol: float = 1e-6) -> VQEResult:
    """Minimise the UCCSD energy with a SciPy optimizer (``COBYLA`` or ``L-BFGS-B``)."""
    x0 = np.zeros(n_parameters()) if x0 is None else np.asarray(x0, dtype=float)
    history: list[float] = []

    def objective(theta: np.ndarray) -> float:
        e = energy(hamiltonian, theta)
        history.append(e)
        return e

    if optimizer.upper() == "COBYLA":
        res = minimize(objective, x0, method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": rhobeg, "tol": tol})
    elif optimizer.upper() == "L-BFGS-B":
        res = minimize(objective, x0, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": tol, "gtol": 1e-7})
    else:
        raise ValueError("optimizer must be COBYLA or L-BFGS-B")
    return VQEResult(float(res.fun), np.asarray(res.x), history)


# --------------------------------------------------------------------------- Exact references
def exact_energies(r_angstrom: float) -> tuple[float, float]:
    """(E_fci, E_hf) for H2/STO-3G at bond length R, from PySCF (total energies, Ha)."""
    from pyscf import fci, gto, scf

    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r_angstrom}", basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    e_fci = float(fci.FCI(mf).kernel()[0])
    return e_fci, float(mf.e_tot)


def hydrogen_atom_energy() -> float:
    """Energy of an isolated H atom in STO-3G (UHF doublet), in Hartree."""
    from pyscf import gto, scf

    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    return float(scf.UHF(mol).kernel())


# --------------------------------------------------------------------------- Sweep
def sweep(radii: Sequence[float], optimizer: str = "COBYLA", log=print) -> list[dict]:
    """VQE over bond lengths with warm-started parameters; adds FCI/HF references."""
    rows: list[dict] = []
    theta = np.zeros(n_parameters())
    for r in radii:
        h = build_hamiltonian(r)
        res = run_vqe(h, x0=theta, optimizer=optimizer, rhobeg=0.05)
        theta = res.params  # warm start for the next bond length
        e_fci, e_hf = exact_energies(r)
        err = abs(res.energy - e_fci) * HARTREE_TO_MHA
        log(f"  R={r:.2f} A  E_vqe={res.energy:.8f}  E_fci={e_fci:.8f}  "
            f"err={err:.5f} mHa  ({res.n_evals} evals)")
        rows.append({"R": float(r), "E_vqe": res.energy, "E_fci": e_fci, "E_hf": e_hf,
                     "abs_error_mHa": err})
    return rows


def find_equilibrium(bounds: tuple[float, float] = (0.6, 0.9), use_vqe: bool = True,
                     theta0: Optional[Sequence[float]] = None) -> tuple[float, float]:
    """Bond length and energy at the minimum of the curve (bounded 1D search).

    ``use_vqe=True`` minimises the VQE energy (re-running VQE at each trial R,
    warm-started); otherwise the exact FCI energy. Returns (R_eq in A, E_min in Ha).
    """
    from scipy.optimize import minimize_scalar

    state = {"theta": np.zeros(n_parameters()) if theta0 is None else np.asarray(theta0, float)}

    def f(r: float) -> float:
        if not use_vqe:
            return exact_energies(r)[0]
        res = run_vqe(build_hamiltonian(r), x0=state["theta"], rhobeg=0.02)
        state["theta"] = res.params
        return res.energy

    out = minimize_scalar(f, bounds=bounds, method="bounded", options={"xatol": 1e-4})
    return float(out.x), float(out.fun)


def write_pes_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PES_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_pes_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


# --------------------------------------------------------------------------- Plots
def plot_convergence(history: list[float], e_fci: float, path: Path, r: float = 0.735) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    ax.plot(range(1, len(history) + 1), history, marker="o", ms=3, lw=1.4, label="VQE (UCCSD, COBYLA)")
    ax.axhline(e_fci, color="k", ls="--", lw=1.2, label=f"FCI = {e_fci:.6f} Ha")
    ax.set_xlabel("Energy evaluation")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(f"VQE convergence, H$_2$ at R = {r} Å")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pes(rows: list[dict], e_atom: float, path: Path) -> None:
    r = np.array([x["R"] for x in rows])
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    ax.plot(r, [x["E_fci"] for x in rows], color="k", lw=1.8, label="FCI (exact)")
    ax.plot(r, [x["E_hf"] for x in rows], color="#d62728", ls="--", lw=1.5, label="Hartree-Fock")
    ax.plot(r, [x["E_vqe"] for x in rows], "o", color="#1f77b4", ms=5, label="VQE (UCCSD)")
    ax.axhline(2 * e_atom, color="gray", ls="--", lw=1.1, label="2 × E(H atom)")
    ax.set_xlabel("H–H bond length R (Å)")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("H$_2$ potential energy curve (STO-3G)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_error(rows: list[dict], path: Path) -> None:
    r = [x["R"] for x in rows]
    err = np.maximum([x["abs_error_mHa"] for x in rows], 1e-9)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    ax.plot(r, err, "o-", color="#1f77b4", ms=4, lw=1.3, label="|E$_{VQE}$ − E$_{FCI}$|")
    ymin = min(err.min() / 3, 1e-6)
    ax.axhspan(ymin, CHEM_ACC_MHA, color="#76b900", alpha=0.18,
               label=f"chemical accuracy ({CHEM_ACC_MHA} mHa)")
    ax.set_yscale("log")
    ax.set_ylim(ymin, max(err.max() * 3, 3.0))
    ax.set_xlabel("H–H bond length R (Å)")
    ax.set_ylabel("|E$_{VQE}$ − E$_{FCI}$| (mHa)")
    ax.set_title("VQE error vs. exact FCI")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- CLI
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--r-min", type=float, default=0.30)
    p.add_argument("--r-max", type=float, default=2.50)
    p.add_argument("--r-step", type=float, default=0.05)
    p.add_argument("--optimizer", default="COBYLA", choices=["COBYLA", "L-BFGS-B"])
    p.add_argument("--quick", action="store_true", help="coarse sweep 0.5..2.0 A step 0.25")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: results/ (results/quick with --quick)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.r_min, args.r_max, args.r_step = 0.50, 2.00, 0.25
    out_dir = args.out_dir or Path("results/quick" if args.quick else "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    cudaq.set_target("qpp-cpu")
    t0 = time.perf_counter()

    h0 = build_hamiltonian(0.735)
    n_terms = h0.term_count
    print(f"Hamiltonian at 0.735 A: {n_terms} Pauli terms (incl. identity), "
          f"{n_parameters()} UCCSD parameters")

    print("\n== Convergence run at R = 0.735 A ==")
    conv = run_vqe(h0, optimizer=args.optimizer)
    e_fci_0735, e_hf_0735 = exact_energies(0.735)
    err_0735 = abs(conv.energy - e_fci_0735) * HARTREE_TO_MHA
    print(f"  E_vqe={conv.energy:.8f}  E_fci={e_fci_0735:.8f}  err={err_0735:.5f} mHa "
          f"({conv.n_evals} evals)")
    plot_convergence(conv.history, e_fci_0735, out_dir / "vqe_convergence.png")

    print("\n== Bond-length sweep (warm-started) ==")
    radii = np.round(np.arange(args.r_min, args.r_max + 1e-9, args.r_step), 3)
    rows = sweep(radii, optimizer=args.optimizer)
    write_pes_csv(rows, out_dir / "h2_pes.csv")
    e_atom = hydrogen_atom_energy()
    plot_pes(rows, e_atom, out_dir / "h2_pes.png")
    plot_error(rows, out_dir / "h2_error.png")

    r_arr = np.array([x["R"] for x in rows])
    print("\n== Locating the equilibrium bond length (1D search) ==")
    r_eq_v, e_min_v = find_equilibrium(use_vqe=True, theta0=conv.params)
    r_eq_f, e_min_f = find_equilibrium(use_vqe=False)
    summary = {
        "n_pauli_terms": n_terms,
        "n_parameters": n_parameters(),
        "optimizer": args.optimizer,
        "E_h_atom_ha": e_atom,
        "at_0735": {"E_vqe": conv.energy, "E_fci": e_fci_0735, "E_hf": e_hf_0735,
                    "abs_error_mHa": err_0735, "n_evals": conv.n_evals},
        "grid_min_vqe": {"R": float(r_arr[np.argmin([x["E_vqe"] for x in rows])]),
                         "E": min(x["E_vqe"] for x in rows)},
        "fit_min_vqe": {"R": r_eq_v, "E": e_min_v},
        "fit_min_fci": {"R": r_eq_f, "E": e_min_f},
        "max_abs_error_mHa": max(x["abs_error_mHa"] for x in rows),
        "n_points": len(rows),
        "runtime_s": time.perf_counter() - t0,
    }
    (out_dir / "h2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nVQE equilibrium (1D search): R = {r_eq_v:.4f} A, E = {e_min_v:.6f} Ha "
          f"(FCI, same search: {r_eq_f:.4f} A, {e_min_f:.6f} Ha)")
    print(f"Max |E_vqe - E_fci| over sweep: {summary['max_abs_error_mHa']:.5f} mHa "
          f"(chemical accuracy = {CHEM_ACC_MHA} mHa)")
    print(f"Results written to {out_dir}/ in {summary['runtime_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
