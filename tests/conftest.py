"""Make source-layout imports work before installation during local test runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
