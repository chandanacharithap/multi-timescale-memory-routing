#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multitimescale_memory.freshqa_export import (  # noqa: E402
    build_public_freshqa_traceability,
    public_freshqa_provenance_manifest,
)
from multitimescale_memory.reporting import (  # noqa: E402
    build_public_freshqa_error_audit,
    error_audit_markdown,
    error_audit_summary,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_provenance_bundle(
    output_dir: Path,
    snapshot_paths: list[Path],
    main_path: Path,
    sequence_path: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = public_freshqa_provenance_manifest(
        snapshot_paths,
        derived_paths=[main_path, sequence_path],
        derivation_script_path=ROOT / "multitimescale_memory" / "freshqa_export.py",
    )
    traceability = build_public_freshqa_traceability({"main": main_path, "sequence": sequence_path})
    appendix_lines = [
        "| Source Week | Slice | Question | Update Type | Rollback Probe | Confirmation Count Before | Change Count Before |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in traceability:
        appendix_lines.append(
            "| {source_week} | {slice} | {question} | {update_type} | {rollback_probe} | {confirmation_count_before} | {change_count_before} |".format(
                **row
            )
        )
    compatibility_note = "\n".join(
        [
            "# Public FreshQA Compatibility Note",
            "",
            "This benchmark track is derived from the weekly public FreshQA spreadsheets released in the official `freshllms/freshqa` repository.",
            "The same repository also provides `fresheval_strict.ipynb` and `fresheval_relaxed.ipynb` as its reference evaluation notebooks.",
            "Our derived track preserves weekly ordering, carries all `answer_0..answer_n` aliases into `possible_answers`, and evaluates strict answer membership against those accepted aliases.",
            "We therefore describe the benchmark honestly as a public-data-derived track aligned with the public FreshQA format, not as an official packaged split.",
            "",
            "Official repository: https://github.com/freshllms/freshqa",
        ]
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output_dir / "source_to_derived_traceability.jsonl").open("w", encoding="utf-8") as handle:
        for row in traceability:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "appendix_mapping_table.md").write_text("\n".join(appendix_lines), encoding="utf-8")
    (output_dir / "compatibility_note.md").write_text(compatibility_note, encoding="utf-8")
    return {
        "manifest": str(output_dir / "manifest.json"),
        "traceability": str(output_dir / "source_to_derived_traceability.jsonl"),
        "appendix_mapping_table": str(output_dir / "appendix_mapping_table.md"),
        "compatibility_note": str(output_dir / "compatibility_note.md"),
    }


def build_audit_bundle(
    output_dir: Path,
    router_trace_path: Path,
    calibrated_trace_path: Path,
    limit: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    router_traces = _load_jsonl(router_trace_path)
    calibrated_traces = _load_jsonl(calibrated_trace_path)
    audit_rows = build_public_freshqa_error_audit(router_traces, calibrated_traces, limit=limit)
    summary = error_audit_summary(audit_rows)
    snippets = []
    for row in audit_rows[:5]:
        snippets.append(
            "\n".join(
                [
                    f"## {row['episode_id']}",
                    f"- label: {row['label']}",
                    f"- router action: {row['router_action']}",
                    f"- calibrated action: {row['calibrated_action']}",
                    f"- rationale: {row['rationale']}",
                ]
            )
        )
    (output_dir / "error_audit_100.json").write_text(json.dumps(audit_rows, indent=2), encoding="utf-8")
    (output_dir / "error_audit_100.md").write_text(error_audit_markdown(audit_rows), encoding="utf-8")
    (output_dir / "error_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "representative_trace_snippets.md").write_text("\n\n".join(snippets), encoding="utf-8")
    return {
        "audit_json": str(output_dir / "error_audit_100.json"),
        "audit_markdown": str(output_dir / "error_audit_100.md"),
        "audit_summary": str(output_dir / "error_audit_summary.json"),
        "snippets": str(output_dir / "representative_trace_snippets.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public FreshQA provenance and audit bundles for the TMLR boundary paper.")
    parser.add_argument("--snapshot", action="append", required=True, help="Public FreshQA weekly CSV snapshot path. Repeat this flag for each snapshot.")
    parser.add_argument("--main-jsonl", required=True, help="Path to the derived public FreshQA main JSONL.")
    parser.add_argument("--sequence-jsonl", required=True, help="Path to the derived public FreshQA sequence JSONL.")
    parser.add_argument("--router-trace", required=True, help="Trace JSONL for the full router on the public track.")
    parser.add_argument("--calibrated-trace", required=True, help="Trace JSONL for the calibrated router on the public track.")
    parser.add_argument("--provenance-dir", required=True, help="Output directory for the public-track provenance bundle.")
    parser.add_argument("--audit-dir", required=True, help="Output directory for the public-track audit bundle.")
    parser.add_argument("--audit-limit", type=int, default=100)
    args = parser.parse_args()

    provenance = build_provenance_bundle(
        Path(args.provenance_dir),
        [Path(item) for item in args.snapshot],
        Path(args.main_jsonl),
        Path(args.sequence_jsonl),
    )
    audit = build_audit_bundle(
        Path(args.audit_dir),
        Path(args.router_trace),
        Path(args.calibrated_trace),
        args.audit_limit,
    )
    print(json.dumps({"provenance": provenance, "audit": audit}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
