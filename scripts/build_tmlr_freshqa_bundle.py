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
    build_public_freshqa_export,
    build_public_freshqa_traceability,
    public_freshqa_provenance_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full FreshQA public benchmark bundle for the TMLR campaign.")
    parser.add_argument("--snapshot-glob", default="data/freshqa_*.csv")
    parser.add_argument("--output-dir", default="artifacts/tmlr_freshqa_full_v1")
    parser.add_argument("--min-snapshots-per-question", type=int, default=3)
    args = parser.parse_args()

    snapshot_paths = sorted(Path(ROOT).glob(args.snapshot_glob))
    if not snapshot_paths:
        raise SystemExit(f"no snapshot files matched {args.snapshot_glob}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = Path("data/freshqa_public_full_main_v3.jsonl")
    sequence_path = Path("data/freshqa_public_full_sequence_v3.jsonl")

    main_manifest = build_public_freshqa_export(
        snapshot_paths,
        main_path,
        min_snapshots_per_question=args.min_snapshots_per_question,
        max_questions=0,
        test_split_only=True,
        slice_mode="main",
    )
    sequence_manifest = build_public_freshqa_export(
        snapshot_paths,
        sequence_path,
        min_snapshots_per_question=args.min_snapshots_per_question,
        max_questions=0,
        test_split_only=True,
        slice_mode="sequence",
    )
    provenance = public_freshqa_provenance_manifest(
        snapshot_paths,
        derived_paths=[main_path, sequence_path],
        derivation_script_path=Path("multitimescale_memory/freshqa_export.py"),
    )
    traceability = build_public_freshqa_traceability({"main": main_path, "sequence": sequence_path})

    (output_dir / "main_manifest.json").write_text(json.dumps(main_manifest, indent=2), encoding="utf-8")
    (output_dir / "sequence_manifest.json").write_text(json.dumps(sequence_manifest, indent=2), encoding="utf-8")
    (output_dir / "provenance_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output_dir / "source_to_derived_traceability.json").write_text(json.dumps(traceability, indent=2), encoding="utf-8")
    (output_dir / "snapshot_coverage.json").write_text(
        json.dumps(
            {
                "snapshot_paths": [str(path) for path in snapshot_paths],
                "snapshot_count": len(snapshot_paths),
                "main_output": str(main_path),
                "sequence_output": str(sequence_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "main_output": str(main_path),
                "sequence_output": str(sequence_path),
                "artifact_dir": str(output_dir),
                "snapshots": [str(path) for path in snapshot_paths],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
