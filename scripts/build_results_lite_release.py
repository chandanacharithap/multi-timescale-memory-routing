#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables" / "github_release"
STAGING = DELIVERABLES / "paper-v1.0.0-results-lite"
ZIP_BASE = DELIVERABLES / "paper-v1.0.0-results-lite"

OFFICIAL_BUNDLES = {
    "popqa_recurring_flan_base_v1": [
        "manifest.json",
        "aggregate_table.md",
        "run_rows.jsonl",
        "significance.json",
        "bootstrap.json",
    ],
    "freshqa_public_main_flan_small_v1": [
        "manifest.json",
        "aggregate_table.md",
        "run_rows.jsonl",
        "significance.json",
        "bootstrap.json",
    ],
    "freshqa_public_main_flan_base_v1": [
        "manifest.json",
        "aggregate_table.md",
        "run_rows.jsonl",
        "significance.json",
        "bootstrap.json",
    ],
    "freshqa_public_sequence_flan_small_v1": [
        "manifest.json",
        "aggregate_table.md",
        "run_rows.jsonl",
        "significance.json",
        "bootstrap.json",
    ],
    "freshqa_public_sequence_flan_base_v1": [
        "manifest.json",
        "aggregate_table.md",
        "run_rows.jsonl",
        "significance.json",
        "bootstrap.json",
    ],
    "freshqa_public_main_qwen_1p5b_v1": [
        "manifest.json",
        "aggregate_table.md",
        "run_rows.jsonl",
        "significance.json",
        "bootstrap.json",
    ],
    "freshqa_public_sequence_qwen_1p5b_v1": [
        "manifest.json",
        "aggregate_table.md",
        "run_rows.jsonl",
        "significance.json",
        "bootstrap.json",
    ],
    "freshqa_public_main_smollm2_360m_v1": [
        "manifest.json",
        "aggregate_table.md",
        "run_rows.jsonl",
        "significance.json",
        "bootstrap.json",
    ],
    "freshqa_public_sequence_smollm2_360m_v1": [
        "manifest.json",
        "aggregate_table.md",
        "run_rows.jsonl",
        "significance.json",
        "bootstrap.json",
    ],
    "mquake_flan_base_v1": [
        "manifest.json",
        "aggregate_table.md",
        "run_rows.jsonl",
        "significance.json",
        "bootstrap.json",
        "benchmark_metric_aggregates.json",
    ],
    "mquake_flan_small_v1": [
        "manifest.json",
        "aggregate_table.md",
        "run_rows.jsonl",
        "significance.json",
        "bootstrap.json",
        "benchmark_metric_aggregates.json",
    ],
    "uniedit_official_v1": [
        "manifest.json",
        "aggregate_table.md",
        "run_rows.jsonl",
        "significance.json",
        "bootstrap.json",
        "benchmark_metric_aggregates.json",
    ],
}

SUPPORT_BUNDLES = {
    "freshqa_leakage_audit_manual_v1": [
        "summary.json",
        "manual_audit_100.md",
        "manual_audit_100.json",
        "selection_manifest.json",
    ],
    "official_statistics_v1": [
        "README.md",
        "cross_benchmark_summary.md",
        "bootstrap_all.json",
        "effect_sizes_all.json",
        "significance_all.json",
        "run_index.json",
    ],
    "mquake_reliability_note_v1": [
        "reliability_note.json",
    ],
    "uniedit_reliability_note_v1": [
        "reliability_note.json",
    ],
}

SUPPLEMENTARY_BUNDLES = {
    "freshness_v1": [
        "README.md",
        "freshness_bundle_manifest.json",
        "baseline_comparison_table.md",
        "ablation_table.md",
        "router_summary.json",
        "rollback_report.md",
        "error_audit_summary.json",
        "error_audit_50.md",
    ],
}


def copy_bundle(source_root: Path, target_root: Path, filenames: list[str]) -> list[str]:
    copied: list[str] = []
    target_root.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        source = source_root / filename
        if not source.exists():
            continue
        shutil.copy2(source, target_root / filename)
        copied.append(str((target_root / filename).relative_to(STAGING)))
    return copied


def main() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "release_tag": "paper-v1.0.0",
        "description": "Lightweight frozen results bundle for the public GitHub release.",
        "official": {},
        "support": {},
        "supplementary": {},
    }

    readme_lines = [
        "# Results Lite Release Bundle",
        "",
        "This bundle contains the small, public-facing frozen summaries and statistics for the GitHub release.",
        "",
        "It intentionally excludes large raw artifact trees, heavy benchmark data, environments, caches, and internal delivery outputs.",
        "",
        "## Included Surfaces",
        "",
        "- `official/`: benchmark run summaries and significance/bootstrap files",
        "- `support/`: audit, cross-benchmark statistics, and reliability notes",
        "- `supplementary/`: bundled controlled-update supplementary context",
        "",
    ]

    for bundle_id, filenames in OFFICIAL_BUNDLES.items():
        source_root = ROOT / "artifacts" / "tmlr_official" / bundle_id
        target_root = STAGING / "official" / bundle_id
        manifest["official"][bundle_id] = copy_bundle(source_root, target_root, filenames)

    for bundle_id, filenames in SUPPORT_BUNDLES.items():
        source_root = ROOT / "artifacts" / "tmlr_official" / bundle_id
        target_root = STAGING / "support" / bundle_id
        manifest["support"][bundle_id] = copy_bundle(source_root, target_root, filenames)

    for bundle_id, filenames in SUPPLEMENTARY_BUNDLES.items():
        source_root = ROOT / "artifacts" / bundle_id
        target_root = STAGING / "supplementary" / bundle_id
        manifest["supplementary"][bundle_id] = copy_bundle(source_root, target_root, filenames)

    (STAGING / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    (STAGING / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    archive_path = shutil.make_archive(str(ZIP_BASE), "zip", root_dir=DELIVERABLES, base_dir=STAGING.name)
    summary = {
        "staging_dir": str(STAGING.relative_to(ROOT)),
        "zip_path": str(Path(archive_path).relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
