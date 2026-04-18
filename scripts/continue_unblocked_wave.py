#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTEGRITY_PATHS = [
    "paper/arxiv",
    "paper/tmlr",
    "paper/shared/claims_to_artifacts.md",
    "paper/shared/generated/README.md",
    "paper/shared/artifact_availability.tex",
    "docs/journal_reproducibility.md",
    "deliverables/PUBLICATION_STATUS.md",
    "deliverables/README.md",
    "paper_run_manifest_index.json",
]

RUNS = [
    {
        "run_id": "mquake_flan_small_v1",
        "benchmark": "mquake",
        "artifact_root": "artifacts/tmlr_official/mquake_flan_small_v1",
        "manifest": "artifacts/tmlr_official/manifests/mquake_flan_small_v1.json",
        "required_files": [
            "aggregate_table.md",
            "benchmark_metric_aggregates.json",
            "bootstrap.json",
            "effect_sizes.json",
            "manifest.json",
            "run_rows.jsonl",
        ],
    },
    {
        "run_id": "freshqa_public_main_qwen_1p5b_v1",
        "benchmark": "freshqa_public",
        "artifact_root": "artifacts/tmlr_official/freshqa_public_main_qwen_1p5b_v1",
        "manifest": "artifacts/tmlr_official/manifests/freshqa_public_main_qwen_1p5b_v1.json",
        "required_files": [
            "aggregate_table.md",
            "bootstrap.json",
            "effect_sizes.json",
            "manifest.json",
            "run_rows.jsonl",
        ],
    },
    {
        "run_id": "freshqa_public_sequence_qwen_1p5b_v1",
        "benchmark": "freshqa_public",
        "artifact_root": "artifacts/tmlr_official/freshqa_public_sequence_qwen_1p5b_v1",
        "manifest": "artifacts/tmlr_official/manifests/freshqa_public_sequence_qwen_1p5b_v1.json",
        "required_files": [
            "aggregate_table.md",
            "bootstrap.json",
            "effect_sizes.json",
            "manifest.json",
            "run_rows.jsonl",
        ],
    },
    {
        "run_id": "freshqa_public_main_smollm2_360m_v1",
        "benchmark": "freshqa_public",
        "artifact_root": "artifacts/tmlr_official/freshqa_public_main_smollm2_360m_v1",
        "manifest": "artifacts/tmlr_official/manifests/freshqa_public_main_smollm2_360m_v1.json",
        "required_files": [
            "aggregate_table.md",
            "bootstrap.json",
            "effect_sizes.json",
            "manifest.json",
            "run_rows.jsonl",
        ],
    },
    {
        "run_id": "freshqa_public_sequence_smollm2_360m_v1",
        "benchmark": "freshqa_public",
        "artifact_root": "artifacts/tmlr_official/freshqa_public_sequence_smollm2_360m_v1",
        "manifest": "artifacts/tmlr_official/manifests/freshqa_public_sequence_smollm2_360m_v1.json",
        "required_files": [
            "aggregate_table.md",
            "bootstrap.json",
            "effect_sizes.json",
            "manifest.json",
            "run_rows.jsonl",
        ],
    },
]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def _load_registry() -> dict:
    return json.loads((ROOT / "paper_run_manifest_index.json").read_text(encoding="utf-8"))


def _save_registry(registry: dict) -> None:
    (ROOT / "paper_run_manifest_index.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _is_registered(run_id: str) -> bool:
    registry = _load_registry()
    return any(row.get("run_id") == run_id for row in registry.get("reportable_artifacts", []))


def _register_run(run: dict[str, object]) -> None:
    registry = _load_registry()
    if not any(row.get("run_id") == run["run_id"] for row in registry["reportable_artifacts"]):
        registry["reportable_artifacts"].append(
            {
                "run_id": run["run_id"],
                "benchmark": run["benchmark"],
                "artifact_root": run["artifact_root"],
                "status": "frozen_real_run",
                "paper_scope": ["tmlr"],
            }
        )
    registry["pending_required_runs"] = [row for row in registry["pending_required_runs"] if row.get("run_id") != run["run_id"]]
    _save_registry(registry)


def _validate_bundle(run: dict[str, object]) -> None:
    root = ROOT / str(run["artifact_root"])
    missing = [name for name in run["required_files"] if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"bundle {run['run_id']} missing files: {missing}")


def _integrity_refresh() -> None:
    _run([sys.executable, "scripts/build_official_stats_bundle.py"])
    _run([sys.executable, "scripts/render_official_status.py"])
    _run([sys.executable, "scripts/audit_submission_integrity.py", *sum([["--path", p] for p in INTEGRITY_PATHS], [])])


def _wait_for_existing_run(run: dict[str, object]) -> None:
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


def main() -> None:
    # First wait for the already-running MQuAKE small bundle.
    first = RUNS[0]
    if not _is_registered(str(first["run_id"])):
        _wait_for_existing_run(first)
        _validate_bundle(first)
        _register_run(first)
        _integrity_refresh()

    # Then execute the remaining official FreshQA decoder-only runs sequentially.
    for run in RUNS[1:]:
        if _is_registered(str(run["run_id"])):
            continue
        _run([str(ROOT / ".venv-prod" / "bin" / "python"), "scripts/run_tmlr_campaign.py", "--manifest", str(run["manifest"])])
        _validate_bundle(run)
        _register_run(run)
        _integrity_refresh()


if __name__ == "__main__":
    main()
