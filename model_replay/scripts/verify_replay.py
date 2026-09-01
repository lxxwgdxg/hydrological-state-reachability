from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(expected: Any, actual: Any, path: str, mismatches: list[dict[str, Any]]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            mismatches.append({"path": path, "expected_type": "dict", "actual_type": type(actual).__name__})
            return
        for key, value in expected.items():
            if key not in actual:
                mismatches.append({"path": f"{path}.{key}", "reason": "missing"})
            else:
                compare(value, actual[key], f"{path}.{key}", mismatches)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            mismatches.append({"path": path, "expected_length": len(expected), "actual_length": len(actual) if isinstance(actual, list) else None})
            return
        for index, value in enumerate(expected):
            compare(value, actual[index], f"{path}[{index}]", mismatches)
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            mismatches.append({"path": path, "expected": expected, "actual": actual})
            return
        if not math.isclose(float(expected), float(actual), rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            mismatches.append({"path": path, "expected": expected, "actual": actual})
        return
    if expected != actual:
        mismatches.append({"path": path, "expected": expected, "actual": actual})


def main() -> None:
    replay_path = ROOT / "outputs" / "model_replay.json"
    expected_path = ROOT / "reference" / "expected_replay.json"
    provenance_path = ROOT / "reference" / "provenance.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    actual_rows = replay.get("per_basin", [])
    mismatches: list[dict[str, Any]] = []
    if len(actual_rows) != 1:
        mismatches.append({"path": "per_basin", "expected_length": 1, "actual_length": len(actual_rows)})
    else:
        compare(expected["per_basin"], actual_rows[0], "per_basin[0]", mismatches)

    parity = float(replay.get("official_original_path_max_abs_parity_error_mm_day", math.inf))
    parity_pass = parity <= 1e-6
    expected_days = int(expected["per_basin"]["performance"]["original"]["n"])
    actual_days = int(actual_rows[0]["performance"]["original"]["n"]) if actual_rows else -1
    day_count_pass = expected_days == actual_days == 3652

    hash_mismatches = []
    for item in provenance["hash_inventory"]:
        path = ROOT / item["path"]
        actual_hash = sha256(path) if path.is_file() else None
        if actual_hash != item["sha256"]:
            hash_mismatches.append({"path": item["path"], "expected": item["sha256"], "actual": actual_hash})

    absolute_path_hits = []
    credential_hits = []
    text_suffixes = {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".csv"}
    # The negative lookbehind prevents URL schemes such as ``https://`` from
    # being misclassified as a Windows drive path at the trailing ``s:/``.
    absolute_pattern = re.compile(r"(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/]")
    credential_pattern = re.compile(r"(?i)(?:token|secret|password|api[_-]?key)\s*[:=]\s*[^\s<]+")
    for path in ROOT.rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.is_file() and path.suffix.lower() in text_suffixes and "outputs" not in path.parts:
            text = path.read_text(encoding="utf-8", errors="replace")
            if absolute_pattern.search(text):
                absolute_path_hits.append(path.relative_to(ROOT).as_posix())
            if credential_pattern.search(text):
                credential_hits.append(path.relative_to(ROOT).as_posix())

    report = {
        "audit": "clean claim-specific DPL-H4 model replay verification",
        "numeric_tolerance": TOLERANCE,
        "frozen_reference_field_mismatch_count": len(mismatches),
        "frozen_reference_field_mismatches": mismatches,
        "official_forward_parity_error_mm_day": parity,
        "official_forward_parity_pass": parity_pass,
        "full_test_day_count": actual_days,
        "full_test_day_count_pass": day_count_pass,
        "provenance_hash_count": len(provenance["hash_inventory"]),
        "provenance_hash_mismatch_count": len(hash_mismatches),
        "provenance_hash_mismatches": hash_mismatches,
        "absolute_path_hit_count": len(absolute_path_hits),
        "absolute_path_hits": absolute_path_hits,
        "credential_assignment_hit_count": len(credential_hits),
        "credential_assignment_hits": credential_hits,
    }
    report["overall_pass"] = bool(
        not mismatches
        and parity_pass
        and day_count_pass
        and not hash_mismatches
        and not absolute_path_hits
        and not credential_hits
    )
    output = ROOT / "outputs" / "model_replay_verification.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
