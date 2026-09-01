"""Run the complete analysis-level reproduction in a fixed order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = (
    "reproduce_statistics.py",
    "plot_core_figure.py",
    "plot_window_robustness.py",
    "verify_outputs.py",
)


def main() -> None:
    for script in SCRIPTS:
        print(f"[run] {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    print("Analysis-level reproduction passed.")


if __name__ == "__main__":
    main()
