from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output = root / "outputs" / "model_replay.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    python_path = [str(root / "upstream_code"), str(root / "scripts")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    replay = [
        sys.executable,
        str(root / "scripts" / "audit_dpl_h4_sample_norm_checkpoint.py"),
        "--run-dir", str(root / "model" / "seed11"),
        "--epoch", "30",
        "--panel", str(root / "config" / "replay_basin.txt"),
        "--output", str(output),
        "--device", args.device,
        "--batch-size", "256",
        "--basin-limit", "1",
    ]
    subprocess.run(replay, cwd=root, env=environment, check=True)
    subprocess.run(
        [sys.executable, str(root / "scripts" / "verify_replay.py")],
        cwd=root,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
