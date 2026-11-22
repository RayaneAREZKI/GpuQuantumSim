"""Make ``src/`` importable and force the CPU target for tests."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
