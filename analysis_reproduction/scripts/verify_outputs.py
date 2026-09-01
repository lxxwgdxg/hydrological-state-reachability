"""Verify reproduced statistics, figures, and public-package boundaries."""

from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORT = OUTPUTS / "REPRODUCTION_VERIFICATION.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_integrity_audit() -> dict[str, Any]:
    provenance = json.loads(
        (ROOT / "reference" / "source_provenance.json").read_text(encoding="utf-8")
    )
    mismatches = []
    checked = 0
    for record in provenance["source_files"]:
        relative = record["package_relative_path"]
        path = ROOT / relative
        checked += 1
        observed = sha256(path) if path.is_file() else None
        if observed != record["package_sha256"]:
            mismatches.append(
                {
                    "package_relative_path": relative,
                    "reason": "missing" if observed is None else "sha256 mismatch",
                }
            )
    return {
        "checked_data_file_count": checked,
        "hash_mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "pass": checked == 12 and not mismatches,
    }


def compare(expected: Any, actual: Any, path: str, mismatches: list[dict[str, Any]]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            mismatches.append({"path": path, "reason": "type mismatch"})
            return
        for key, value in expected.items():
            if key not in actual:
                mismatches.append({"path": f"{path}.{key}", "reason": "missing"})
            else:
                compare(value, actual[key], f"{path}.{key}", mismatches)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            mismatches.append({"path": path, "reason": "list shape mismatch"})
            return
        for index, value in enumerate(expected):
            compare(value, actual[index], f"{path}[{index}]", mismatches)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append({"path": path, "expected": expected, "actual": actual})
    elif expected != actual:
        mismatches.append({"path": path, "expected": expected, "actual": actual})


def boundary_audit() -> dict[str, Any]:
    allowed_suffixes = {".md", ".yml", ".py", ".csv", ".json", ".png", ".pdf"}
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".nc", ".h5", ".hdf5", ".tar", ".zip"}
    text_suffixes = {".md", ".yml", ".py", ".csv", ".json"}
    absolute_path = re.compile(r"[A-Za-z]:[\\/]")
    secret_assignment = re.compile(r"(?i)(token|api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]+['\"]")
    absolute_hits = []
    secret_hits = []
    forbidden_files = []
    unexpected_suffixes = []
    for path in sorted(item for item in ROOT.rglob("*") if item.is_file()):
        relative = path.relative_to(ROOT).as_posix()
        suffix = path.suffix.lower()
        if suffix in forbidden_suffixes:
            forbidden_files.append(relative)
        if suffix not in allowed_suffixes:
            unexpected_suffixes.append(relative)
        if suffix in text_suffixes:
            content = path.read_text(encoding="utf-8", errors="replace")
            if absolute_path.search(content):
                absolute_hits.append(relative)
            if secret_assignment.search(content):
                secret_hits.append(relative)
    return {
        "absolute_windows_path_hits": absolute_hits,
        "credential_assignment_hits": secret_hits,
        "forbidden_binary_or_archive_files": forbidden_files,
        "unexpected_file_suffixes": unexpected_suffixes,
        "pass": not (absolute_hits or secret_hits or forbidden_files or unexpected_suffixes),
    }


def main() -> None:
    expected = json.loads((ROOT / "reference" / "expected_statistics.json").read_text(encoding="utf-8"))
    actual = json.loads((OUTPUTS / "reproduced_statistics.json").read_text(encoding="utf-8"))
    mismatches: list[dict[str, Any]] = []
    compare(expected, actual, "$", mismatches)
    figures = [
        "CORE_FIGURE_REPRODUCED.png",
        "CORE_FIGURE_REPRODUCED.pdf",
        "WINDOW_INITIALIZATION_ROBUSTNESS_REPRODUCED.png",
        "WINDOW_INITIALIZATION_ROBUSTNESS_REPRODUCED.pdf",
    ]
    missing_figures = [name for name in figures if not (OUTPUTS / name).is_file() or (OUTPUTS / name).stat().st_size == 0]
    integrity = data_integrity_audit()
    boundary = boundary_audit()
    passed = not mismatches and not missing_figures and integrity["pass"] and boundary["pass"]
    report = {
        "analysis_statistics_match_reference": not mismatches,
        "statistical_mismatch_count": len(mismatches),
        "statistical_mismatches": mismatches,
        "all_four_figure_files_exist": not missing_figures,
        "missing_figure_files": missing_figures,
        "data_integrity_audit": integrity,
        "package_boundary_audit": boundary,
        "overall_pass": passed,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
