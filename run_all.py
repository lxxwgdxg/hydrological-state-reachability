"""Run the public repository's frozen reproduction layers."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-model-replay", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    subprocess.run(
        [sys.executable, str(ROOT / "analysis_reproduction" / "run_all.py")],
        cwd=ROOT / "analysis_reproduction",
        check=True,
    )
    if args.with_model_replay:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "model_replay" / "run_all.py"),
                "--device",
                args.device,
            ],
            cwd=ROOT / "model_replay",
            check=True,
        )
    print("Requested repository reproductions passed.")


if __name__ == "__main__":
    main()
