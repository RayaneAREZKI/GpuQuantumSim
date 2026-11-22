"""Tests for the statevector memory estimator."""
import pytest

from hw import GIB, estimate_statevector_bytes, fits_in_memory


def test_fp32_is_8_bytes_per_amplitude():
    assert estimate_statevector_bytes(30, "fp32") == 2**30 * 8
    assert estimate_statevector_bytes(30, "fp32") / GIB == 8.0


def test_fp64_is_16_bytes_per_amplitude():
    assert estimate_statevector_bytes(20, "fp64") == 2**20 * 16


def test_each_extra_qubit_doubles_memory():
    assert estimate_statevector_bytes(25, "fp32") == 2 * estimate_statevector_bytes(24, "fp32")


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        estimate_statevector_bytes(-1)
    with pytest.raises(ValueError):
        estimate_statevector_bytes(10, "fp16")


def test_fits_in_memory_respects_headroom():
    need = estimate_statevector_bytes(20, "fp64")  # 16 MiB
    assert fits_in_memory(20, "fp64", need * 2)
    assert not fits_in_memory(20, "fp64", need)          # 100% > 80% headroom
    assert not fits_in_memory(20, "fp64", None)          # unknown -> unsafe
