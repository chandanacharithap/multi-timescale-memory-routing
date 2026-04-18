#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_table(rows: list[dict]) -> str:
    header = "| Run ID | Benchmark | Model | Best Mode | Best Quality | Reference Retrieve | Reference No-V-Future |"
    sep = "| --- | --- | --- | --- | ---: | ---: | ---: |"
    body: list[str] = []
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["run_id"], row["benchmark"], row["model_name"]), []).append(row)
    for (run_id, benchmark, model_name), items in sorted(grouped.items()):
        best = max(items, key=lambda r: r["answer_quality"]["mean"])
        retrieve = next((r for r in items if r["mode"] == "always_retrieve"), None)
        nov = next((r for r in items if r["mode"] == "router_no_v_future"), None)
        body.append(
            f"| {run_id} | {benchmark} | {model_name} | {best['mode']} | {best['answer_quality']['mean']:.3f} | "
            f"{(retrieve['answer_quality']['mean'] if retrieve else 0.0):.3f} | "
            f"{(nov['answer_quality']['mean'] if nov else 0.0):.3f} |"
        )
    return "\n".join([header, sep, *body]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an official cross-benchmark stats bundle from registry-listed runs.")
    parser.add_argument("--registry", default="paper_run_manifest_index.json")
    parser.add_argument("--output-dir", default="artifacts/tmlr_official/official_statistics_v1")
    args = parser.parse_args()

    registry = _load_registry(Path(args.registry))
    official = [
        row
        for row in registry["reportable_artifacts"]
        if str(row.get("artifact_root", "")).startswith("artifacts/tmlr_official/")
    ]
    aggregate_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    effect_rows: list[dict] = []
    indexed: list[dict] = []
    for item in official:
        root = Path(item["artifact_root"])
        agg_path = root / "aggregate_rows.json"
        if not agg_path.exists():
            continue
        current_rows = _load_json(agg_path)
        for row in current_rows:
            row["run_id"] = item["run_id"]
            row["artifact_root"] = item["artifact_root"]
        aggregate_rows.extend(current_rows)
        indexed.append({"run_id": item["run_id"], "artifact_root": item["artifact_root"], "aggregate_rows": len(current_rows)})
        boot = root / "bootstrap.json"
        if boot.exists():
            bootstrap_rows.extend(_load_json(boot))
        eff = root / "effect_sizes.json"
        if eff.exists():
            effect_rows.extend(_load_json(eff))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate_rows_all.json").write_text(json.dumps(aggregate_rows, indent=2), encoding="utf-8")
    (output_dir / "bootstrap_all.json").write_text(json.dumps(bootstrap_rows, indent=2), encoding="utf-8")
    (output_dir / "effect_sizes_all.json").write_text(json.dumps(effect_rows, indent=2), encoding="utf-8")
    (output_dir / "run_index.json").write_text(json.dumps(indexed, indent=2), encoding="utf-8")
    (output_dir / "cross_benchmark_summary.md").write_text(_summary_table(aggregate_rows), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Official Statistics Bundle",
                "",
                "- source of truth: registry-listed official TMLR artifacts only",
                "- formal significance layer: pending implementation",
                "- do not imply p-values from this bundle until a `significance.json` surface exists",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
