#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTEGRITY_PATHS = [
    "docs/journal_reproducibility.md",
    "paper_run_manifest_index.json",
]

UNIEDIT_RUN = {
    "run_id": "uniedit_official_v1",
    "benchmark": "uniedit",
    "artifact_root": "artifacts/tmlr_official/uniedit_official_v1",
    "manifest": "artifacts/tmlr_official/manifests/uniedit_official_v1.json",
    "required_files": [
        "aggregate_table.md",
        "benchmark_metric_aggregates.json",
        "bootstrap.json",
        "effect_sizes.json",
        "manifest.json",
        "run_rows.jsonl",
    ],
}


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def _load_registry() -> dict:
    return json.loads((ROOT / "paper_run_manifest_index.json").read_text(encoding="utf-8"))


def _save_registry(registry: dict) -> None:
    (ROOT / "paper_run_manifest_index.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _official_rows(registry: dict) -> list[dict]:
    if "tmlr_official_reportable" in registry:
        return list(registry["tmlr_official_reportable"])
    return [
        row
        for row in registry.get("reportable_artifacts", [])
        if str(row.get("artifact_root", "")).startswith("artifacts/tmlr_official/")
    ]


def _is_registered(run_id: str) -> bool:
    registry = _load_registry()
    return any(row.get("run_id") == run_id for row in _official_rows(registry))


def _register_if_missing(run_id: str, benchmark: str, artifact_root: str) -> None:
    registry = _load_registry()
    if "tmlr_official_reportable" not in registry:
        registry["tmlr_official_reportable"] = _official_rows(registry)
    if "legacy_reportable_support" not in registry:
        registry["legacy_reportable_support"] = [
            row
            for row in registry.get("reportable_artifacts", [])
            if not str(row.get("artifact_root", "")).startswith("artifacts/tmlr_official/")
        ]
    if not any(row.get("run_id") == run_id for row in registry["tmlr_official_reportable"]):
        registry["tmlr_official_reportable"].append(
            {
                "run_id": run_id,
                "benchmark": benchmark,
                "artifact_root": artifact_root,
                "status": "frozen_real_run",
                "paper_scope": ["tmlr"],
            }
        )
    registry["reportable_artifacts"] = list(registry["tmlr_official_reportable"])
    if run_id == UNIEDIT_RUN["run_id"]:
        registry["pending_required_runs"] = [
            row for row in registry["pending_required_runs"] if row.get("run_id") != run_id
        ]
    _save_registry(registry)


def _validate_bundle(run: dict[str, object]) -> None:
    root = ROOT / str(run["artifact_root"])
    missing = [name for name in run["required_files"] if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"bundle {run['run_id']} missing files: {missing}")


def _wait_for_run(run: dict[str, object]) -> None:
    manifest_path = str(run["manifest"])
    bundle_root = ROOT / str(run["artifact_root"])
    required = [bundle_root / name for name in run["required_files"]]
    while True:
        if all(path.exists() for path in required):
            return
        proc = subprocess.run(["pgrep", "-f", manifest_path], cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{run['run_id']} is no longer running and the bundle is still incomplete")
        time.sleep(60)


def _refresh_outputs() -> None:
    _run([sys.executable, "scripts/build_official_stats_bundle.py"])
    _run([sys.executable, "scripts/render_official_status.py"])
    _run([sys.executable, "scripts/audit_submission_integrity.py", *sum([["--path", p] for p in INTEGRITY_PATHS], [])])


def main() -> None:
    if _is_registered(UNIEDIT_RUN["run_id"]):
        _refresh_outputs()
        return
    _wait_for_run(UNIEDIT_RUN)
    _validate_bundle(UNIEDIT_RUN)
    _register_if_missing("uniedit_reliability_note_v1", "uniedit", "artifacts/tmlr_official/uniedit_reliability_note_v1")
    _register_if_missing(UNIEDIT_RUN["run_id"], UNIEDIT_RUN["benchmark"], UNIEDIT_RUN["artifact_root"])
    _refresh_outputs()


if __name__ == "__main__":
    main()
