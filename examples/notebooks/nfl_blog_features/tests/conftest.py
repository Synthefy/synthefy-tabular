"""Make the notebook-local helper package importable during repository tests."""

import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NOTEBOOKS))
