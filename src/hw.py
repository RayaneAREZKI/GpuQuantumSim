"""Hardware detection and statevector memory estimation helpers.

Kept free of CUDA-Q imports at module level so it can be unit-tested anywhere.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Optional

import psutil

#: Bytes per complex amplitude: complex64 (fp32) and complex128 (fp64).
BYTES_PER_AMPLITUDE = {"fp32": 8, "fp64": 16}
GIB = 1024**3


@dataclass
class GpuInfo:
    """Description of the first visible NVIDIA GPU."""

    name: str
    vram_total_bytes: int
    vram_free_bytes: int


def estimate_statevector_bytes(n_qubits: int, precision: str = "fp32") -> int:
    """Memory needed to hold a dense n-qubit statevector.

    A statevector has 2**n complex amplitudes: 8 bytes each for complex64
    (``fp32``) or 16 bytes each for complex128 (``fp64``).
    """
    if n_qubits < 0:
        raise ValueError("n_qubits must be non-negative")
    try:
        return (2**n_qubits) * BYTES_PER_AMPLITUDE[precision]
    except KeyError as exc:
        raise ValueError(f"precision must be one of {sorted(BYTES_PER_AMPLITUDE)}") from exc


def bytes_to_gib(n_bytes: float) -> float:
    """Convert bytes to GiB (2**30 bytes)."""
    return n_bytes / GIB


def fits_in_memory(
    n_qubits: int, precision: str, available_bytes: Optional[int], headroom: float = 0.8
) -> bool:
    """True if the statevector uses at most ``headroom`` of ``available_bytes``.

    ``headroom`` < 1 leaves room for the OS, the Python process and simulator
    scratch space. ``available_bytes=None`` means unknown -> not safe to run.
    """
    if available_bytes is None:
        return False
    return estimate_statevector_bytes(n_qubits, precision) <= headroom * available_bytes


def available_ram_bytes() -> int:
    """Currently available system RAM in bytes."""
    return int(psutil.virtual_memory().available)


def detect_gpu() -> Optional[GpuInfo]:
    """Return info on the first NVIDIA GPU via ``nvidia-smi``, or None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
        name, total_mib, free_mib = [s.strip() for s in out.splitlines()[0].split(",")]
        return GpuInfo(name, int(total_mib) * 1024**2, int(free_mib) * 1024**2)
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        return None


def gpu_target_usable() -> bool:
    """True if a GPU is present AND this CUDA-Q build lists the ``nvidia`` target."""
    if detect_gpu() is None:
        return False
    import cudaq

    return "nvidia" in {t.name for t in cudaq.get_targets()}


def hardware_summary() -> dict:
    """Collect a JSON-serialisable description of the machine."""
    gpu = detect_gpu()
    try:
        import cudaq

        cudaq_version = cudaq.__version__.splitlines()[0]
    except Exception:  # pragma: no cover - cudaq missing
        cudaq_version = "not installed"
    return {
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gib": round(bytes_to_gib(psutil.virtual_memory().total), 1),
        "gpu": asdict(gpu) if gpu else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "cudaq": cudaq_version,
    }
