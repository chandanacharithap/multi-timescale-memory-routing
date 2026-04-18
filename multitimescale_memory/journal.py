from __future__ import annotations

from collections import Counter
import json
import math
from dataclasses import asdict
from pathlib import Path

from .runner import ExperimentRunner, RunnerConfig
from .types import JournalAggregateRow, JournalRunRow, MetricSummary, ResultRow


JOURNAL_BASELINE_MODES = {
    # Submission-facing defaults intentionally exclude proxy-family baselines.
    "popqa": ["always_retrieve", "retrieve_gate", "router", "router_calibrated"],
    "freshness": ["always_retrieve", "router", "router_calibrated"],
    "freshqa_public": ["always_retrieve", "retrieve_gate", "router", "router_calibrated"],
    "mquake": ["always_retrieve", "router", "router_calibrated"],
    "knowedit": ["always_retrieve", "router", "router_calibrated"],
    "uniedit": ["always_retrieve", "router", "router_calibrated"],
}


def _metric_summary(values: list[float]) -> MetricSummary:
    if not values:
        return MetricSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / max(1, count - 1) if count > 1 else 0.0
    std = math.sqrt(max(variance, 0.0))
    margin = 1.96 * std / math.sqrt(count) if count > 1 else 0.0
    return MetricSummary(
        mean=mean,
        std=std,
        min=min(values),
        max=max(values),
        ci95_low=mean - margin,
        ci95_high=mean + margin,
        count=count,
    )


def _row_from_result_row(benchmark: str, model_name: str, seed: int, row: ResultRow) -> JournalRunRow:
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


def aggregate_journal_rows(rows: list[JournalRunRow]) -> list[JournalAggregateRow]:
    grouped: dict[tuple[str, str, str], list[JournalRunRow]] = {}
    for row in rows:
        grouped.setdefault((row.benchmark, row.model_name, row.mode), []).append(row)
    aggregate_rows: list[JournalAggregateRow] = []
    for (benchmark, model_name, mode), group in sorted(grouped.items()):
        aggregate_rows.append(
            JournalAggregateRow(
                benchmark=benchmark,
                model_name=model_name,
                mode=mode,
                answer_quality=_metric_summary([item.answer_quality for item in group]),
                recurring_quality=_metric_summary([item.recurring_quality for item in group]),
                non_recurring_quality=_metric_summary([item.non_recurring_quality for item in group]),
                latency=_metric_summary([item.latency for item in group]),
                retrieval_calls=_metric_summary([item.retrieval_calls for item in group]),
                memory_reads=_metric_summary([item.memory_reads for item in group]),
                memory_writes=_metric_summary([item.memory_writes for item in group]),
                adaptation_count=_metric_summary([item.adaptation_count for item in group]),
                stale_answer_rate=_metric_summary([item.stale_answer_rate for item in group]),
                consolidation_count=_metric_summary([item.consolidation_count for item in group]),
                rollback_count=_metric_summary([item.rollback_count for item in group]),
                forgetting_delta=_metric_summary([item.forgetting_delta for item in group]),
                average_reward=_metric_summary([item.average_reward for item in group]),
            )
        )
    return aggregate_rows


def journal_aggregate_table(rows: list[JournalAggregateRow]) -> str:
    header = (
        "| Benchmark | Model | Mode | Quality Mean | Quality 95% CI | Latency Mean | "
        "Retrieve Mean | Read Mean | Write Mean | Adapt Mean | Stale Mean | Consolidate Mean | Reward Mean |"
    )
    divider = "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    body = []
    for row in rows:
        body.append(
            "| {benchmark} | {model} | {mode} | {quality:.3f} | [{low:.3f}, {high:.3f}] | {latency:.3f} | {retrieve:.3f} | {reads:.3f} | {writes:.3f} | {adapt:.3f} | {stale:.3f} | {consolidate:.3f} | {reward:.3f} |".format(
                benchmark=row.benchmark,
                model=row.model_name,
                mode=row.mode,
                quality=row.answer_quality.mean,
                low=row.answer_quality.ci95_low,
                high=row.answer_quality.ci95_high,
                latency=row.latency.mean,
                retrieve=row.retrieval_calls.mean,
                reads=row.memory_reads.mean,
                writes=row.memory_writes.mean,
                adapt=row.adaptation_count.mean,
                stale=row.stale_answer_rate.mean,
                consolidate=row.consolidation_count.mean,
                reward=row.average_reward.mean,
            )
        )
    return "\n".join([header, divider, *body])


def journal_frontier_table(rows: list[JournalAggregateRow]) -> str:
    header = "| Benchmark | Model | Mode | Quality | Latency | Retrieval Calls | Memory Writes | Adapt Count |"
    divider = "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"
    body = []
    for row in rows:
        body.append(
            "| {benchmark} | {model} | {mode} | {quality:.3f} | {latency:.3f} | {retrieve:.3f} | {writes:.3f} | {adapt:.3f} |".format(
                benchmark=row.benchmark,
                model=row.model_name,
                mode=row.mode,
                quality=row.answer_quality.mean,
                latency=row.latency.mean,
                retrieve=row.retrieval_calls.mean,
                writes=row.memory_writes.mean,
                adapt=row.adaptation_count.mean,
            )
        )
    return "\n".join([header, divider, *body])


def journal_stale_answer_table(rows: list[JournalAggregateRow]) -> str:
    header = "| Benchmark | Model | Mode | Stale-Answer Rate Mean | 95% CI |"
    divider = "| --- | --- | --- | ---: | --- |"
    body = []
    for row in rows:
        body.append(
            "| {benchmark} | {model} | {mode} | {mean:.3f} | [{low:.3f}, {high:.3f}] |".format(
                benchmark=row.benchmark,
                model=row.model_name,
                mode=row.mode,
                mean=row.stale_answer_rate.mean,
                low=row.stale_answer_rate.ci95_low,
                high=row.stale_answer_rate.ci95_high,
            )
        )
    return "\n".join([header, divider, *body])


def journal_rollback_report(rows: list[JournalAggregateRow]) -> str:
    header = "| Benchmark | Model | Mode | Consolidate Mean | Rollback Mean |"
    divider = "| --- | --- | --- | ---: | ---: |"
    body = []
    for row in rows:
        body.append(
            "| {benchmark} | {model} | {mode} | {consolidate:.3f} | {rollback:.3f} |".format(
                benchmark=row.benchmark,
                model=row.model_name,
                mode=row.mode,
                consolidate=row.consolidation_count.mean,
                rollback=row.rollback_count.mean,
            )
        )
    return "\n".join([header, divider, *body])


def journal_forgetting_report(rows: list[JournalAggregateRow]) -> str:
    header = "| Benchmark | Model | Mode | Forgetting Delta Mean | 95% CI |"
    divider = "| --- | --- | --- | ---: | --- |"
    body = []
    for row in rows:
        body.append(
            "| {benchmark} | {model} | {mode} | {mean:.3f} | [{low:.3f}, {high:.3f}] |".format(
                benchmark=row.benchmark,
                model=row.model_name,
                mode=row.mode,
                mean=row.forgetting_delta.mean,
                low=row.forgetting_delta.ci95_low,
                high=row.forgetting_delta.ci95_high,
            )
        )
    return "\n".join([header, divider, *body])


def journal_action_distribution(rows: list[JournalRunRow]) -> dict[str, dict[str, dict[str, int]]]:
    aggregated: dict[str, dict[str, Counter[str]]] = {}
    for row in rows:
        benchmark_group = aggregated.setdefault(row.benchmark, {})
        key = f"{row.model_name}::{row.mode}"
        bucket = benchmark_group.setdefault(key, Counter())
        bucket.update(row.action_distribution)
    return {
        benchmark: {key: dict(counter) for key, counter in group.items()}
        for benchmark, group in aggregated.items()
    }


def _frontier_svg(
    rows: list[JournalAggregateRow],
    *,
    title: str,
    x_accessor,
    y_accessor,
    x_label: str,
    y_label: str,
    width: int = 960,
    height: int = 540,
) -> str:
    if not rows:
        rows = []
    margin_left = 80
    margin_right = 30
    margin_top = 50
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    xs = [float(x_accessor(row)) for row in rows] or [0.0]
    ys = [float(y_accessor(row)) for row in rows] or [0.0]
    max_x = max(xs) if max(xs) > 0 else 1.0
    max_y = max(ys) if max(ys) > 0 else 1.0

    def scale_x(value: float) -> float:
        return margin_left + (value / max_x) * plot_width

    def scale_y(value: float) -> float:
        return margin_top + plot_height - (value / max_y) * plot_height

    points: list[str] = []
    palette = ["#2874a6", "#ca6f1e", "#229954", "#8e44ad", "#c0392b", "#5d6d7e"]
    for index, row in enumerate(rows):
        x = scale_x(float(x_accessor(row)))
        y = scale_y(float(y_accessor(row)))
        color = palette[index % len(palette)]
        label = f"{row.model_name}:{row.mode}"
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" />')
        points.append(f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="11" fill="#333">{label}</text>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#faf8f3"/>
  <text x="{width/2:.1f}" y="28" font-size="22" text-anchor="middle" fill="#222">{title}</text>
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#333" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" />
  <text x="{width/2:.1f}" y="{height - 18}" font-size="14" text-anchor="middle" fill="#333">{x_label}</text>
  <text x="18" y="{height/2:.1f}" font-size="14" text-anchor="middle" fill="#333" transform="rotate(-90,18,{height/2:.1f})">{y_label}</text>
  {''.join(points)}
</svg>"""


def _runner_result_to_row(mode: str, result: dict[str, object]) -> ResultRow:
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
        action_distribution=dict(result["actions"]),
        extra_metrics={
            "stale_answer_rate": float(summary.get("stale_answer_rate", 0.0)),
            "consolidation_count": float(summary.get("consolidations", 0.0)),
            "rollback_count": float(summary.get("rollback_count", 0.0)),
            "forgetting_delta": float(summary.get("forgetting_delta", 0.0)),
            "average_reward": float(summary.get("average_reward", 0.0)),
        },
    )


def run_journal_matrix(
    base_config: RunnerConfig,
    benchmark_name: str,
    model_names: list[str],
    seeds: list[int],
    include_ablations: bool = True,
) -> tuple[dict[str, object], list[JournalRunRow], list[JournalAggregateRow]]:
    return run_selected_modes_matrix(
        base_config=base_config,
        benchmark_name=benchmark_name,
        model_names=model_names,
        seeds=seeds,
        modes=JOURNAL_BASELINE_MODES.get(benchmark_name, JOURNAL_BASELINE_MODES["popqa"]),
        include_ablations=include_ablations,
    )


def run_selected_modes_matrix(
    base_config: RunnerConfig,
    benchmark_name: str,
    model_names: list[str],
    seeds: list[int],
    modes: list[str],
    include_ablations: bool = False,
) -> tuple[dict[str, object], list[JournalRunRow], list[JournalAggregateRow]]:
    run_rows: list[JournalRunRow] = []
    for model_name in model_names:
        for seed in seeds:
            for mode in modes:
                runner = ExperimentRunner(
                    RunnerConfig(
                        reward_weights=base_config.reward_weights,
                        alpha=base_config.alpha,
                        trace_path=None,
                        baseline_mode=mode,
                        benchmark_name=benchmark_name,
                        model_name=model_name,
                        cache_dir=base_config.cache_dir,
                        popqa_limit=base_config.popqa_limit,
                        sqlite_memory_path=base_config.sqlite_memory_path,
                        use_v_future=base_config.use_v_future,
                        disable_actions=base_config.disable_actions,
                        allow_popqa_network=base_config.allow_popqa_network,
                        popqa_cached_only=base_config.popqa_cached_only,
                        popqa_guardrail_mode=base_config.popqa_guardrail_mode,
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
                run_rows.append(_row_from_result_row(benchmark_name, model_name, seed, _runner_result_to_row(mode, runner.run())))
            if include_ablations:
                ablation_runner = ExperimentRunner(
                    RunnerConfig(
                        reward_weights=base_config.reward_weights,
                        alpha=base_config.alpha,
                        benchmark_name=benchmark_name,
                        model_name=model_name,
                        cache_dir=base_config.cache_dir,
                        popqa_limit=base_config.popqa_limit,
                        sqlite_memory_path=base_config.sqlite_memory_path,
                        use_v_future=base_config.use_v_future,
                        allow_popqa_network=base_config.allow_popqa_network,
                        popqa_cached_only=base_config.popqa_cached_only,
                        popqa_guardrail_mode=base_config.popqa_guardrail_mode,
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
                for row in ablation_runner.run_router_ablation_suite():
                    run_rows.append(_row_from_result_row(benchmark_name, model_name, seed, row))
    aggregate_rows = aggregate_journal_rows(run_rows)
    manifest = {
        "benchmark": benchmark_name,
        "models": model_names,
        "seeds": seeds,
        "modes": modes,
        "include_ablations": include_ablations,
        "router_alpha": base_config.alpha,
        "use_v_future": base_config.use_v_future,
        "public_freshness_path": base_config.public_freshness_path,
        "sequence_repeats": base_config.sequence_repeats,
        "popqa_limit": base_config.popqa_limit,
        "freshness_limit": base_config.freshness_limit,
        "benchmark_limit": base_config.benchmark_limit,
        "benchmark_sample_seed": base_config.benchmark_sample_seed,
        "future_value_scale": base_config.future_value_scale,
    }
    return manifest, run_rows, aggregate_rows


def run_freshqa_future_scale_sweep(
    base_config: RunnerConfig,
    model_names: list[str],
    seeds: list[int],
    scales: list[float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for scale in scales:
        config = RunnerConfig(
            reward_weights=base_config.reward_weights,
            alpha=base_config.alpha,
            benchmark_name=base_config.benchmark_name,
            model_name=base_config.model_name,
            cache_dir=base_config.cache_dir,
            popqa_limit=base_config.popqa_limit,
            sqlite_memory_path=base_config.sqlite_memory_path,
            use_v_future=base_config.use_v_future,
            disable_actions=base_config.disable_actions,
            allow_popqa_network=base_config.allow_popqa_network,
            popqa_cached_only=base_config.popqa_cached_only,
            popqa_guardrail_mode=base_config.popqa_guardrail_mode,
            freshness_limit=base_config.freshness_limit,
            public_freshness_path=base_config.public_freshness_path,
            sequence_repeats=base_config.sequence_repeats,
            future_value_scale=scale,
        )
        manifest, run_rows, aggregate_rows = run_journal_matrix(
            base_config=config,
            benchmark_name=base_config.benchmark_name,
            model_names=model_names,
            seeds=seeds,
            include_ablations=False,
        )
        router_rows = [row for row in aggregate_rows if row.mode == "router"]
        for row in router_rows:
            rows.append(
                {
                    "benchmark": row.benchmark,
                    "model_name": row.model_name,
                    "future_value_scale": scale,
                    "answer_quality_mean": row.answer_quality.mean,
                    "answer_quality_ci95_low": row.answer_quality.ci95_low,
                    "answer_quality_ci95_high": row.answer_quality.ci95_high,
                    "latency_mean": row.latency.mean,
                    "retrieval_calls_mean": row.retrieval_calls.mean,
                    "memory_reads_mean": row.memory_reads.mean,
                    "memory_writes_mean": row.memory_writes.mean,
                    "adaptation_count_mean": row.adaptation_count.mean,
                    "stale_answer_rate_mean": row.stale_answer_rate.mean,
                    "forgetting_delta_mean": row.forgetting_delta.mean,
                    "average_reward_mean": row.average_reward.mean,
                }
            )
    manifest = {
        "benchmark": base_config.benchmark_name,
        "models": model_names,
        "seeds": seeds,
        "future_value_scales": scales,
        "public_freshness_path": base_config.public_freshness_path,
        "sequence_repeats": base_config.sequence_repeats,
    }
    return manifest, rows


def export_freshqa_future_scale_sweep(output_dir: Path, manifest: dict[str, object], rows: list[dict[str, object]]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output_dir / "run_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "| Benchmark | Model | V_future Scale | Quality Mean | Quality 95% CI | Retrieve Mean | Adapt Mean | Stale Mean | Forgetting Delta Mean |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {benchmark} | {model_name} | {future_value_scale:.2f} | {answer_quality_mean:.3f} | [{answer_quality_ci95_low:.3f}, {answer_quality_ci95_high:.3f}] | {retrieval_calls_mean:.3f} | {adaptation_count_mean:.3f} | {stale_answer_rate_mean:.3f} | {forgetting_delta_mean:.3f} |".format(
                **row
            )
        )
    (output_dir / "aggregate_table.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "manifest": str(output_dir / "manifest.json"),
        "run_rows": str(output_dir / "run_rows.jsonl"),
        "rows": str(output_dir / "rows.json"),
        "aggregate_table": str(output_dir / "aggregate_table.md"),
    }


def export_journal_bundle(
    output_dir: Path,
    manifest: dict[str, object],
    run_rows: list[JournalRunRow],
    aggregate_rows: list[JournalAggregateRow],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_vs_retrieval = _frontier_svg(
        aggregate_rows,
        title="Quality vs Retrieval Cost",
        x_accessor=lambda row: row.retrieval_calls.mean,
        y_accessor=lambda row: row.answer_quality.mean,
        x_label="Mean retrieval calls",
        y_label="Mean quality",
    )
    quality_vs_forgetting = _frontier_svg(
        aggregate_rows,
        title="Quality vs Forgetting Risk",
        x_accessor=lambda row: abs(row.forgetting_delta.mean),
        y_accessor=lambda row: row.answer_quality.mean,
        x_label="Absolute forgetting delta",
        y_label="Mean quality",
    )
    quality_vs_adaptation = _frontier_svg(
        aggregate_rows,
        title="Quality vs Adaptation and Write Burden",
        x_accessor=lambda row: row.adaptation_count.mean + row.memory_writes.mean,
        y_accessor=lambda row: row.answer_quality.mean,
        x_label="Mean adapt + write burden",
        y_label="Mean quality",
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output_dir / "run_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in run_rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    (output_dir / "aggregate_rows.json").write_text(
        json.dumps([asdict(row) for row in aggregate_rows], indent=2),
        encoding="utf-8",
    )
    (output_dir / "aggregate_table.md").write_text(journal_aggregate_table(aggregate_rows), encoding="utf-8")
    (output_dir / "frontier_table.md").write_text(journal_frontier_table(aggregate_rows), encoding="utf-8")
    (output_dir / "stale_answer_table.md").write_text(journal_stale_answer_table(aggregate_rows), encoding="utf-8")
    (output_dir / "rollback_report.md").write_text(journal_rollback_report(aggregate_rows), encoding="utf-8")
    (output_dir / "forgetting_report.md").write_text(journal_forgetting_report(aggregate_rows), encoding="utf-8")
    (output_dir / "quality_vs_retrieval_cost.svg").write_text(quality_vs_retrieval, encoding="utf-8")
    (output_dir / "quality_vs_forgetting_risk.svg").write_text(quality_vs_forgetting, encoding="utf-8")
    (output_dir / "quality_vs_adaptation_burden.svg").write_text(quality_vs_adaptation, encoding="utf-8")
    (output_dir / "action_distribution.json").write_text(
        json.dumps(journal_action_distribution(run_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "manifest": str(output_dir / "manifest.json"),
        "run_rows": str(output_dir / "run_rows.jsonl"),
        "aggregate_rows": str(output_dir / "aggregate_rows.json"),
        "aggregate_table": str(output_dir / "aggregate_table.md"),
        "frontier_table": str(output_dir / "frontier_table.md"),
        "stale_answer_table": str(output_dir / "stale_answer_table.md"),
        "rollback_report": str(output_dir / "rollback_report.md"),
        "forgetting_report": str(output_dir / "forgetting_report.md"),
        "quality_vs_retrieval_cost": str(output_dir / "quality_vs_retrieval_cost.svg"),
        "quality_vs_forgetting_risk": str(output_dir / "quality_vs_forgetting_risk.svg"),
        "quality_vs_adaptation_burden": str(output_dir / "quality_vs_adaptation_burden.svg"),
        "action_distribution": str(output_dir / "action_distribution.json"),
    }
