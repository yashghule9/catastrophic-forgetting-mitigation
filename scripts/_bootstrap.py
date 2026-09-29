"""Ensures `import cfm` works when scripts are run directly without an editable
install (`pip install -e .`). Every script imports this first."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
