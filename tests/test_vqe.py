"""H2 VQE tests: Hamiltonian size, accuracy at equilibrium, fallback builder."""
import cudaq
import pytest

import h2_vqe as v


@pytest.fixture(autouse=True)
def cpu_target():
    cudaq.set_target("qpp-cpu")


def test_pauli_term_count_is_15():
    assert v.build_hamiltonian(0.735).term_count == 15


def test_ansatz_has_three_parameters():
    assert v.n_parameters() == 3


def test_vqe_within_chemical_accuracy_at_equilibrium():
    h = v.build_hamiltonian(0.735)
    res = v.run_vqe(h)
    e_fci, e_hf = v.exact_energies(0.735)
    assert abs(res.energy - e_fci) * 1000 < v.CHEM_ACC_MHA
    assert res.energy < e_hf  # VQE recovers correlation energy beyond Hartree-Fock
    assert len(res.history) == res.n_evals > 3


def test_openfermion_fallback_matches_cudaq_helper():
    h_fallback = v.build_hamiltonian(0.735, backend="openfermion")
    assert h_fallback.term_count == 15
    e_fb = v.energy(h_fallback, [0.0, 0.0, 0.0])
    e_cq = v.energy(v.build_hamiltonian(0.735, backend="cudaq"), [0.0, 0.0, 0.0])
    assert e_fb == pytest.approx(e_cq, abs=1e-7)
    assert e_fb == pytest.approx(v.exact_energies(0.735)[1], abs=1e-6)  # = HF energy


def test_find_equilibrium_fci_matches_known_minimum():
    r_eq, e_min = v.find_equilibrium(use_vqe=False)
    assert abs(r_eq - 0.735) < 0.01
    assert abs(e_min - (-1.1373)) < 1e-3
