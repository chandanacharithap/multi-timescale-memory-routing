from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .journal import aggregate_journal_rows, export_freshqa_future_scale_sweep, export_journal_bundle, run_freshqa_future_scale_sweep
from .reporting import build_public_freshqa_leakage_audit, error_audit_markdown, error_audit_summary
from .runner import ExperimentRunner, RunnerConfig
from .stats import effect_size_dz, paired_bootstrap_mean_diff, paired_sign_test
from .types import JournalRunRow, ResultRow


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", text.lower())


def _result_row_from_run(mode: str, result: dict[str, object]) -> ResultRow:
    summary = result["summary"]
    recurring = result["subsets"]["recurring"]
    non_recurring = result["subsets"]["non_recurring"]
    return ResultRow(
        mode=mode,
        answer_quality=float(summary["accuracy"]),
        recurring_quality=float(recurring["accuracy"]),
        non_recurring_quality=float(non_recurring["accuracy"]),
        latency=float(summary["latency"]),
        retrieval_calls=int(summary["retrieval_calls"]),
        memory_reads=int(summary["memory_reads"]),
        memory_writes=int(summary["writes"]),
        adaptation_count=int(summary["adapt_steps"]),
        recurring_retrieval_calls=float(recurring["retrieval_calls"]),
        recurring_memory_reads=float(recurring["memory_reads"]),
        recurring_memory_writes=float(recurring["memory_writes"]),
        action_distribution=result["actions"],
        extra_metrics={
            "stale_answer_rate": float(summary.get("stale_answer_rate", 0.0)),
            "consolidation_count": float(summary.get("consolidations", 0.0)),
            "rollback_count": float(summary.get("rollback_count", 0.0)),
            "forgetting_delta": float(summary.get("forgetting_delta", 0.0)),
            "average_reward": float(summary.get("average_reward", 0.0)),
        },
    )


def _journal_run_row(benchmark: str, model_name: str, seed: int, row: ResultRow) -> JournalRunRow:
    return JournalRunRow(
        benchmark=benchmark,
        model_name=model_name,
        seed=seed,
        mode=row.mode,
        answer_quality=row.answer_quality,
        recurring_quality=row.recurring_quality,
        non_recurring_quality=row.non_recurring_quality,
        latency=row.latency,
        retrieval_calls=float(row.retrieval_calls),
        memory_reads=float(row.memory_reads),
        memory_writes=float(row.memory_writes),
        adaptation_count=float(row.adaptation_count),
        stale_answer_rate=float(row.extra_metrics.get("stale_answer_rate", 0.0)),
        consolidation_count=float(row.extra_metrics.get("consolidation_count", 0.0)),
        rollback_count=float(row.extra_metrics.get("rollback_count", 0.0)),
        forgetting_delta=float(row.extra_metrics.get("forgetting_delta", 0.0)),
        average_reward=float(row.extra_metrics.get("average_reward", 0.0)),
        action_distribution=row.action_distribution,
        extra_metrics=dict(row.extra_metrics),
    )


def _resolve_mode(mode: str) -> tuple[str, bool]:
    if mode == "router_no_v_future":
        return "router", False
    return mode, True


def _benchmark_metrics(benchmark_name: str, traces: list[dict[str, Any]]) -> dict[str, float]:
    def quality_for(role: str) -> float:
        items = [float(trace["reward"]["quality"]) for trace in traces if trace["metrics"].get("probe_role") == role or trace["metrics"].get("probe_family") == role or trace["metrics"].get("probe_role") == role]
        return sum(items) / len(items) if items else 0.0

    if benchmark_name == "mquake":
        expected_new = [trace for trace in traces if trace["metrics"].get("probe_role") in {"update", "single_hop", "multi_hop"}]
        forgetting = [
            1.0
            for trace in expected_new
            if trace["answer"].strip().lower() in {str(item).strip().lower() for item in trace["metrics"].get("original_answers", [])}
        ]
        total = len(expected_new)
        return {
            "edit_success": quality_for("update"),
            "single_hop_accuracy": quality_for("single_hop"),
            "multi_hop_accuracy": quality_for("multi_hop"),
            "forgetting_rate": (sum(forgetting) / total) if total else 0.0,
        }
    if benchmark_name == "knowedit":
        forgetfulness_items = [trace for trace in traces if trace["metrics"].get("probe_role") == "forgetfulness"]
        if forgetfulness_items:
            forgetfulness_rate = 1.0 - (sum(float(trace["reward"]["quality"]) for trace in forgetfulness_items) / len(forgetfulness_items))
        else:
            update_items = [trace for trace in traces if trace["metrics"].get("probe_role") == "update"]
            leaked = [
                1.0
                for trace in update_items
                if trace["answer"].strip().lower() in {str(item).strip().lower() for item in trace["metrics"].get("original_answers", [])}
            ]
            forgetfulness_rate = (sum(leaked) / len(update_items)) if update_items else 0.0
        return {
            "edit_success": quality_for("update"),
            "locality_retention": quality_for("locality"),
            "portability_success": quality_for("portability"),
            "forgetfulness_rate": forgetfulness_rate,
        }
    if benchmark_name == "uniedit":
        return {
            "edit_success": quality_for("update"),
            "generality_success": quality_for("generality"),
            "locality_retention": quality_for("locality"),
        }
    return {}


def _augment_trace_metrics(trace: dict[str, Any]) -> dict[str, Any]:
    metadata = trace.get("metadata") or {}
    metrics = dict(trace.get("metrics", {}))
    if "probe_role" in metadata:
        metrics["probe_role"] = metadata["probe_role"]
    if "probe_family" in metadata:
        metrics["probe_family"] = metadata["probe_family"]
    if "original_answers" in metadata:
        metrics["original_answers"] = metadata["original_answers"]
    return metrics


def _write_trace_bundle(path: Path, traces: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")


def _paired_quality_vectors(
    mode_traces: list[dict[str, Any]],
    reference_traces: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float]]:
    mode_by_id = {trace["episode_id"]: float(trace["reward"]["quality"]) for trace in mode_traces}
    reference_by_id = {trace["episode_id"]: float(trace["reward"]["quality"]) for trace in reference_traces}
    ids = sorted(set(mode_by_id).intersection(reference_by_id))
    baseline = [reference_by_id[item] for item in ids]
    candidate = [mode_by_id[item] for item in ids]
    diffs = [cand - base for base, cand in zip(baseline, candidate)]
    return baseline, candidate, diffs


def run_manifest_campaign(manifest: dict[str, Any]) -> dict[str, Any]:
    benchmark_name = str(manifest["benchmark"])
    model_names = list(manifest["model_names"])
    seeds = list(manifest["seeds"])
    modes = list(manifest["modes"])
    output_dir = Path(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = RunnerConfig(
        benchmark_name=benchmark_name,
        model_name=model_names[0],
        public_freshness_path=manifest.get("dataset_path"),
        freshness_limit=manifest.get("freshness_limit"),
        benchmark_limit=manifest.get("benchmark_limit"),
        benchmark_sample_seed=int(manifest.get("benchmark_sample_seed", 2026)),
        sequence_repeats=int(manifest.get("sequence_repeats", 1)),
        future_value_scale=float(manifest.get("future_value_scale", 1.0)),
        cache_dir=str(manifest.get("cache_dir", ".cache")),
        popqa_limit=int(manifest.get("popqa_limit", 100)),
        stochastic_router=True,
    )

    run_rows: list[JournalRunRow] = []
    benchmark_rows: list[dict[str, Any]] = []
    traces_by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = {}

    for model_name in model_names:
        for seed in seeds:
            for mode in modes:
                baseline_mode, use_v_future = _resolve_mode(mode)
                runner = ExperimentRunner(
                    RunnerConfig(
                        reward_weights=base_config.reward_weights,
                        alpha=base_config.alpha,
                        baseline_mode=baseline_mode,
                        benchmark_name=benchmark_name,
                        model_name=model_name,
                        cache_dir=base_config.cache_dir,
                        popqa_limit=base_config.popqa_limit,
                        use_v_future=use_v_future,
                        freshness_limit=base_config.freshness_limit,
                        benchmark_limit=base_config.benchmark_limit,
                        benchmark_sample_seed=base_config.benchmark_sample_seed,
                        router_seed=seed,
                        stochastic_router=True,
                        public_freshness_path=base_config.public_freshness_path,
                        sequence_repeats=base_config.sequence_repeats,
                        future_value_scale=base_config.future_value_scale,
                    )
                )
                result = runner.run()
                row = _result_row_from_run(mode, result)
                run_rows.append(_journal_run_row(benchmark_name, model_name, seed, row))
                augmented_traces = []
                for trace in result["traces"]:
                    trace["metrics"] = _augment_trace_metrics(trace)
                    augmented_traces.append(trace)
                traces_by_key[(model_name, seed, mode)] = augmented_traces
                trace_path = output_dir / "traces" / _slug(model_name) / f"seed_{seed}" / f"{_slug(mode)}.jsonl"
                _write_trace_bundle(trace_path, augmented_traces)
                benchmark_metric_row = {
                    "benchmark": benchmark_name,
                    "model_name": model_name,
                    "seed": seed,
                    "mode": mode,
                    **_benchmark_metrics(benchmark_name, augmented_traces),
                }
                if len(benchmark_metric_row) > 4:
                    benchmark_rows.append(benchmark_metric_row)

    aggregate_rows = aggregate_journal_rows(run_rows)
    bundle_paths = export_journal_bundle(output_dir, manifest, run_rows, aggregate_rows)

    benchmark_metric_aggregates: dict[str, list[dict[str, Any]]] = {}
    if benchmark_rows:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in benchmark_rows:
            grouped[(row["model_name"], row["mode"])].append(row)
        for (model_name, mode), rows in grouped.items():
            keys = [key for key in rows[0].keys() if key not in {"benchmark", "model_name", "seed", "mode"}]
            benchmark_metric_aggregates.setdefault(model_name, [])
            aggregate = {"benchmark": benchmark_name, "model_name": model_name, "mode": mode}
            for key in keys:
                values = [float(item[key]) for item in rows]
                aggregate[f"{key}_mean"] = sum(values) / len(values) if values else 0.0
            benchmark_metric_aggregates[model_name].append(aggregate)
        (output_dir / "benchmark_metric_rows.json").write_text(json.dumps(benchmark_rows, indent=2), encoding="utf-8")
        (output_dir / "benchmark_metric_aggregates.json").write_text(json.dumps(benchmark_metric_aggregates, indent=2, sort_keys=True), encoding="utf-8")

    bootstrap_rows: list[dict[str, Any]] = []
    effect_size_rows: list[dict[str, Any]] = []
    significance_rows: list[dict[str, Any]] = []
    references = [mode for mode in ["always_retrieve", "router_no_v_future"] if mode in modes]
    for model_name in model_names:
        for reference in references:
            for mode in modes:
                if mode == reference:
                    continue
                pooled_baseline: list[float] = []
                pooled_candidate: list[float] = []
                seed_level_diffs: list[float] = []
                for seed in seeds:
                    mode_traces = traces_by_key.get((model_name, seed, mode), [])
                    reference_traces = traces_by_key.get((model_name, seed, reference), [])
                    baseline, candidate, diffs = _paired_quality_vectors(mode_traces, reference_traces)
                    pooled_baseline.extend(baseline)
                    pooled_candidate.extend(candidate)
                    if diffs:
                        seed_level_diffs.append(sum(diffs) / len(diffs))
                if not pooled_baseline:
                    continue
                bootstrap = paired_bootstrap_mean_diff(pooled_baseline, pooled_candidate, seed=2026 + len(bootstrap_rows))
                bootstrap_rows.append(
                    {
                        "benchmark": benchmark_name,
                        "model_name": model_name,
                        "reference_mode": reference,
                        "candidate_mode": mode,
                        **bootstrap,
                    }
                )
                significance_rows.append(
                    {
                        "benchmark": benchmark_name,
                        "model_name": model_name,
                        "reference_mode": reference,
                        "candidate_mode": mode,
                        **paired_sign_test([cand - base for base, cand in zip(pooled_baseline, pooled_candidate)]),
                    }
                )
                effect_size_rows.append(
                    {
                        "benchmark": benchmark_name,
                        "model_name": model_name,
                        "reference_mode": reference,
                        "candidate_mode": mode,
                        **effect_size_dz(seed_level_diffs),
                    }
                )

    (output_dir / "bootstrap.json").write_text(json.dumps(bootstrap_rows, indent=2), encoding="utf-8")
    (output_dir / "effect_sizes.json").write_text(json.dumps(effect_size_rows, indent=2), encoding="utf-8")
    (output_dir / "significance.json").write_text(json.dumps(significance_rows, indent=2), encoding="utf-8")

    if benchmark_name == "freshqa_public" and {"always_retrieve", "router", "router_calibrated"}.issubset(set(modes)):
        first_model = model_names[0]
        first_seed = seeds[0]
        leakage_rows = build_public_freshqa_leakage_audit(
            traces_by_key[(first_model, first_seed, "always_retrieve")],
            traces_by_key[(first_model, first_seed, "router")],
            traces_by_key[(first_model, first_seed, "router_calibrated")],
        )
        leakage_dir = output_dir.parent / "tmlr_freshqa_leakage_audit_v1"
        leakage_dir.mkdir(parents=True, exist_ok=True)
        (leakage_dir / "error_audit_100.json").write_text(json.dumps(leakage_rows, indent=2), encoding="utf-8")
        (leakage_dir / "error_audit_100.md").write_text(error_audit_markdown(leakage_rows), encoding="utf-8")
        (leakage_dir / "error_audit_summary.json").write_text(
            json.dumps(error_audit_summary(leakage_rows), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return {
        "bundle_paths": bundle_paths,
        "benchmark_metric_rows": str(output_dir / "benchmark_metric_rows.json") if benchmark_rows else None,
        "benchmark_metric_aggregates": str(output_dir / "benchmark_metric_aggregates.json") if benchmark_rows else None,
        "bootstrap": str(output_dir / "bootstrap.json"),
        "effect_sizes": str(output_dir / "effect_sizes.json"),
        "significance": str(output_dir / "significance.json"),
        "trace_root": str(output_dir / "traces"),
    }


def run_future_sweep_campaign(manifest: dict[str, Any]) -> dict[str, str]:
    config = RunnerConfig(
        benchmark_name=str(manifest["benchmark"]),
        public_freshness_path=manifest.get("dataset_path"),
        sequence_repeats=int(manifest.get("sequence_repeats", 1)),
        future_value_scale=1.0,
    )
    future_value_scales = [float(item) for item in manifest["future_value_scales"]]
    sweep_manifest, rows = run_freshqa_future_scale_sweep(
        base_config=config,
        model_names=list(manifest["model_names"]),
        seeds=list(manifest["seeds"]),
        scales=future_value_scales,
    )
    return export_freshqa_future_scale_sweep(Path(manifest["output_dir"]), sweep_manifest, rows)
