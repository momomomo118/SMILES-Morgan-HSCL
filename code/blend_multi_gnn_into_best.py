"""Build the current final Scheme A CCRL+GNN decision fusion.

This script is kept at the historical entry-point name for compatibility, but it
no longer builds the legacy residual-base multi-GNN artifact. The deployed final
model uses the same CCRL `pred_ensemble` definition on validation and test, then
fuses it with the three PureGIN branches.
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FINAL_SCRIPT = ROOT / "scripts" / "run_final_fusion_ccrl_gnn.py"


def main() -> None:
    subprocess.run([sys.executable, str(FINAL_SCRIPT)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
