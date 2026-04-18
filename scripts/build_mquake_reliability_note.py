#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from multitimescale_memory.mquake import build_mquake_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an official MQuAKE reliability note bundle.")
    parser.add_argument("--data-path", default="data/benchmarks/mquake/MQuAKE-CF-3k.json")
    parser.add_argument("--output-dir", default="artifacts/tmlr_official/mquake_reliability_note_v1")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    raw = data_path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    cases = json.loads(raw.decode("utf-8"))
    episodes, _docs, manifest = build_mquake_benchmark(data_path)
    required_fields_ok = all(
        "requested_rewrite" in row and "new_single_hops" in row and "questions" in row
        for row in cases
    )
    note = {
        "source_file": str(data_path),
        "source_sha256": sha256,
        "case_count": len(cases),
        "episode_count": len(episodes),
        "update_episodes": manifest["update_episodes"],
        "single_hop_probes": manifest["single_hop_probes"],
        "multi_hop_probes": manifest["multi_hop_probes"],
        "required_fields_ok": required_fields_ok,
        "builder_manifest": manifest,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reliability_note.json").write_text(json.dumps(note, indent=2, sort_keys=True), encoding="utf-8")
    md = "\n".join(
        [
            "# MQuAKE Reliability Note",
            "",
            f"- source file: `{data_path}`",
            f"- source sha256: `{sha256}`",
            f"- case count: {len(cases)}",
            f"- episode count: {len(episodes)}",
            f"- update episodes: {manifest['update_episodes']}",
            f"- single-hop probes: {manifest['single_hop_probes']}",
            f"- multi-hop probes: {manifest['multi_hop_probes']}",
            f"- required field coverage: {'ok' if required_fields_ok else 'failed'}",
            "",
        ]
    )
    (output_dir / "reliability_note.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
