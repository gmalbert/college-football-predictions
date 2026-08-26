"""Update closing-line value and results for recorded prospective shadow signals."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.prospective_ledger import settle_shadow_ledger  # noqa: E402


if __name__ == "__main__":
    print(f"Updated {settle_shadow_ledger()}")
