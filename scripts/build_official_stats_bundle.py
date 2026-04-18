#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multitimescale_memory.stats import paired_sign_test


def _load_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _official_rows(registry: dict) -> list[dict]:
    if "tmlr_official_reportable" in registry:
        return list(registry["tmlr_official_reportable"])
    return [
        row
        for row in registry.get("reportable_artifacts", [])
        if str(row.get("artifact_root", "")).startswith("artifacts/tmlr_official/")
    ]


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


def _load_trace_qualities(trace_path: Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            rows[payload["episode_id"]] = float(payload["reward"]["quality"])
    return rows


def _bundle_significance(bundle_root: Path) -> list[dict]:
    manifest_path = bundle_root / "manifest.json"
    trace_root = bundle_root / "traces"
    if not manifest_path.exists() or not trace_root.exists():
        return []
    manifest = _load_json(manifest_path)
    model_names = list(manifest.get("model_names", []))
    seeds = list(manifest.get("seeds", []))
    modes = list(manifest.get("modes", []))
    references = [mode for mode in ["always_retrieve", "router_no_v_future"] if mode in modes]
    rows: list[dict] = []
    for model_name in model_names:
        model_slug = model_name.lower().replace("/", "_")
        for reference in references:
            for mode in modes:
                if mode == reference:
                    continue
                pooled_diffs: list[float] = []
                for seed in seeds:
                    ref_path = trace_root / model_slug / f"seed_{seed}" / f"{reference}.jsonl"
                    cand_path = trace_root / model_slug / f"seed_{seed}" / f"{mode}.jsonl"
                    if not ref_path.exists() or not cand_path.exists():
                        continue
                    ref_rows = _load_trace_qualities(ref_path)
                    cand_rows = _load_trace_qualities(cand_path)
                    ids = sorted(set(ref_rows).intersection(cand_rows))
                    pooled_diffs.extend(cand_rows[item] - ref_rows[item] for item in ids)
                if not pooled_diffs:
                    continue
                rows.append(
                    {
                        "benchmark": manifest.get("benchmark"),
                        "model_name": model_name,
                        "reference_mode": reference,
                        "candidate_mode": mode,
                        **paired_sign_test(pooled_diffs),
                    }
                )
    return rows


def _ensure_bundle_significance(bundle_root: Path) -> list[dict]:
    significance_path = bundle_root / "significance.json"
    if significance_path.exists():
        return _load_json(significance_path)
    rows = _bundle_significance(bundle_root)
    significance_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an official cross-benchmark stats bundle from registry-listed runs.")
    parser.add_argument("--registry", default="paper_run_manifest_index.json")
    parser.add_argument("--output-dir", default="artifacts/tmlr_official/official_statistics_v1")
    args = parser.parse_args()

    registry = _load_registry(Path(args.registry))
    official = _official_rows(registry)
    aggregate_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    effect_rows: list[dict] = []
    significance_rows: list[dict] = []
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
        significance_rows.extend(_ensure_bundle_significance(root))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate_rows_all.json").write_text(json.dumps(aggregate_rows, indent=2), encoding="utf-8")
    (output_dir / "bootstrap_all.json").write_text(json.dumps(bootstrap_rows, indent=2), encoding="utf-8")
    (output_dir / "effect_sizes_all.json").write_text(json.dumps(effect_rows, indent=2), encoding="utf-8")
    (output_dir / "significance_all.json").write_text(json.dumps(significance_rows, indent=2), encoding="utf-8")
    (output_dir / "run_index.json").write_text(json.dumps(indexed, indent=2), encoding="utf-8")
    (output_dir / "cross_benchmark_summary.md").write_text(_summary_table(aggregate_rows), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Official Statistics Bundle",
                "",
                "- source of truth: TMLR official registry-listed artifacts only",
                "- effect estimation: paired bootstrap confidence intervals",
                "- significance surface: paired sign test over per-example quality differences",
                "- comparisons included: versus `always_retrieve` and `router_no_v_future` when those references exist",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
