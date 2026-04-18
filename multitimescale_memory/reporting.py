from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .types import ResultRow, RunTrace


NON_RECURRING_AUDIT_LABELS = {
    "should have retrieved but used memory",
    "should have retrieved but used parametric answer",
    "memory match was too loose",
    "write happened too early and polluted later routing",
    "recurrence estimate was wrong",
    "low-confidence case should have forced retrieval",
    "retrieval was still the right choice",
}

FRESHNESS_AUDIT_LABELS = {
    "should have retrieved fresh evidence",
    "should have adapted temporarily",
    "consolidated unstable knowledge",
    "stale memory used",
    "rollback should have happened",
    "retrieval was still the right choice",
}

PUBLIC_FRESHQA_AUDIT_LABELS = {
    "premature memory reuse",
    "premature adaptation",
    "stale retrieval evidence",
    "missing alias handling",
    "contradiction under-capture",
    "delayed-utility overshoot",
    "retrieval was correctly dominant",
}

PUBLIC_LEAKAGE_AUDIT_LABELS = {
    "no_leakage_retrieval_legitimately_dominant",
    "timestamp_leakage",
    "alias_leniency",
    "cached_evidence_contamination",
    "evaluator_leniency",
    "derivation_bug",
}


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def overall_comparison_table(rows: list[ResultRow]) -> str:
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                row.mode,
                f"{row.answer_quality:.3f}",
                f"{row.latency:.3f}",
                str(row.retrieval_calls),
                str(row.memory_reads),
                str(row.memory_writes),
                str(row.adaptation_count),
            ]
        )
    return _markdown_table(
        ["Mode", "Overall Quality", "Latency", "Retrieval Calls", "Memory Reads", "Memory Writes", "Adaptation Count"],
        table_rows,
    )


def recurring_vs_nonrecurring_table(rows: list[ResultRow]) -> str:
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                row.mode,
                f"{row.answer_quality:.3f}",
                f"{row.recurring_quality:.3f}",
                f"{row.non_recurring_quality:.3f}",
                f"{row.recurring_retrieval_calls:.3f}",
                f"{row.recurring_memory_reads:.3f}",
                f"{row.recurring_memory_writes:.3f}",
            ]
        )
    return _markdown_table(
        [
            "Mode",
            "Overall Quality",
            "Recurring Quality",
            "Non-Recurring Quality",
            "Recurring Retrieval Calls",
            "Recurring Memory Reads",
            "Recurring Memory Writes",
        ],
        table_rows,
    )


def ablation_table(rows: list[ResultRow]) -> str:
    return recurring_vs_nonrecurring_table(rows)


def _svg_bar_chart(title: str, labels: list[str], values: list[float], color: str, width: int = 960, height: int = 480) -> str:
    if not labels:
        labels = ["none"]
        values = [0.0]
    margin_left = 140
    margin_right = 40
    margin_top = 60
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1e-9)
    bar_gap = 14
    bar_width = max(20, (plot_width - bar_gap * (len(values) - 1)) / max(len(values), 1))
    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        bar_height = (value / max_value) * (plot_height - 20)
        x = margin_left + index * (bar_width + bar_gap)
        y = margin_top + plot_height - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="6" />'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{margin_top + plot_height + 22:.1f}" font-size="12" text-anchor="middle" fill="#333">{label}</text>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" font-size="12" text-anchor="middle" fill="#333">{value:.3f}</text>'
        )
    y_axis = []
    for tick in range(5):
        tick_value = max_value * tick / 4
        y = margin_top + plot_height - (tick / 4) * (plot_height - 20)
        y_axis.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#ddd" />')
        y_axis.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="#555">{tick_value:.3f}</text>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#faf8f3"/>
  <text x="{width/2:.1f}" y="30" font-size="22" text-anchor="middle" fill="#222">{title}</text>
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#333" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" />
  {''.join(y_axis)}
  {''.join(bars)}
</svg>"""


def retrieval_reduction_chart(rows: list[ResultRow]) -> str:
    labels = [row.mode for row in rows]
    values = [row.recurring_retrieval_calls for row in rows]
    return _svg_bar_chart("Recurring Retrieval Calls Per Example", labels, values, color="#ca6f1e")


def action_distribution_chart(actions: dict[str, int]) -> str:
    labels = list(actions.keys())
    values = [float(actions[label]) for label in labels]
    return _svg_bar_chart("Router Action Distribution", labels, values, color="#2874a6")


def stale_answer_chart(rows: list[ResultRow]) -> str:
    labels = [row.mode for row in rows]
    values = [row.extra_metrics.get("stale_answer_rate", 0.0) for row in rows]
    return _svg_bar_chart("Stale Answer Rate", labels, values, color="#b03a2e")


def classify_non_recurring_failure(trace: dict) -> tuple[str, str]:
    action = trace["action"]
    features = trace.get("features", {})
    metrics = trace.get("metrics", {})
    model_confidence = float(features.get("model_confidence", 0.0))
    retrieval_quality = float(features.get("retrieval_quality_estimate", 0.0))
    memory_alignment = float(features.get("memory_alignment_score", 0.0))
    recurrence = float(features.get("recurrence_estimate", 0.0))
    had_memory = bool(metrics.get("had_memory_before", 0))

    if action == "read_memory":
        if memory_alignment < 0.8:
            return (
                "memory match was too loose",
                f"read_memory was chosen with weak memory alignment {memory_alignment:.2f} and recurrence {recurrence:.2f}",
            )
        return (
            "should have retrieved but used memory",
            f"read_memory was chosen on a non-recurring case despite retrieval quality {retrieval_quality:.2f}",
        )

    if action == "write_memory":
        if recurrence < 0.72:
            return (
                "write happened too early and polluted later routing",
                f"write_memory was chosen with recurrence estimate {recurrence:.2f} on a non-recurring failure",
            )
        if memory_alignment < 0.8:
            return (
                "memory match was too loose",
                f"write_memory depended on weak memory alignment {memory_alignment:.2f} after retrieval",
            )
        return (
            "should have retrieved but used memory",
            f"write_memory was chosen instead of a pure retrieval answer despite retrieval quality {retrieval_quality:.2f}",
        )

    if action == "param_only":
        if model_confidence < 0.45 and retrieval_quality >= 0.7:
            return (
                "low-confidence case should have forced retrieval",
                f"param_only stayed active with model confidence {model_confidence:.2f} and retrieval quality {retrieval_quality:.2f}",
            )
        if recurrence >= 0.72:
            return (
                "recurrence estimate was wrong",
                f"param_only won even though the recurrence estimate was elevated at {recurrence:.2f}",
            )
        return (
            "should have retrieved but used parametric answer",
            f"param_only was used on a non-recurring failure with retrieval quality {retrieval_quality:.2f}",
        )

    if action == "retrieve":
        return (
            "retrieval was still the right choice",
            f"retrieve was chosen and still missed; evidence ids={len(trace.get('evidence_ids', []))}",
        )

    if had_memory:
        return (
            "memory match was too loose",
            f"memory existed before the episode, but the chosen action {action} still failed on a non-recurring case",
        )
    return (
        "should have retrieved but used parametric answer",
        f"the chosen action {action} failed without a strong recurring signal or reliable memory reuse",
    )


def build_non_recurring_error_audit(traces: list[dict], limit: int = 50) -> list[dict]:
    selected: list[dict] = []
    for trace in traces:
        if trace["reward"]["quality"] >= 1.0:
            continue
        if bool(trace["metrics"].get("is_recurring_case", 0)):
            continue
        label, rationale = classify_non_recurring_failure(trace)
        if label not in NON_RECURRING_AUDIT_LABELS:
            raise ValueError(f"unexpected audit label: {label}")
        selected.append(
            {
                "episode_id": trace["episode_id"],
                "subject": trace["subject"],
                "relation": trace["relation"],
                "action": trace["action"],
                "answer": trace["answer"],
                "gold_answer": trace["gold_answer"],
                "label": label,
                "rationale": rationale,
                "recurring_case": False,
                "had_memory_before": bool(trace["metrics"].get("had_memory_before", 0)),
                "evidence_ids": trace.get("evidence_ids", []),
            }
        )
        if len(selected) >= limit:
            break
    return selected


def error_audit_markdown(audit_rows: list[dict]) -> str:
    headers = ["Episode", "Subject", "Relation", "Action", "Predicted", "Gold", "Label", "Rationale"]
    rows = []
    for row in audit_rows:
        rows.append(
            [
                row["episode_id"],
                row["subject"],
                row["relation"],
                row["action"],
                row["answer"],
                row["gold_answer"],
                row["label"],
                row["rationale"],
            ]
        )
    return _markdown_table(headers, rows)


def error_audit_summary(audit_rows: list[dict]) -> dict[str, dict[str, float]]:
    counts = Counter(row["label"] for row in audit_rows)
    total = max(len(audit_rows), 1)
    summary: dict[str, dict[str, float]] = {}
    for label, count in sorted(counts.items()):
        summary[label] = {"count": count, "share": count / total}
    return summary


def freshness_overall_table(rows: list[ResultRow]) -> str:
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                row.mode,
                f"{row.answer_quality:.3f}",
                f"{row.extra_metrics.get('stale_answer_rate', 0.0):.3f}",
                f"{row.latency:.3f}",
                str(row.retrieval_calls),
                str(row.memory_reads),
                str(row.memory_writes),
                str(row.adaptation_count),
                str(int(row.extra_metrics.get("consolidation_count", 0.0))),
                str(int(row.extra_metrics.get("rollback_count", 0.0))),
                f"{row.extra_metrics.get('forgetting_delta', 0.0):.3f}",
            ]
        )
    return _markdown_table(
        [
            "Mode",
            "Answer Quality",
            "Stale Answer Rate",
            "Latency",
            "Retrieval Calls",
            "Memory Reads",
            "Memory Writes",
            "Adapt Count",
            "Consolidations",
            "Rollbacks",
            "Forgetting Delta",
        ],
        table_rows,
    )


def freshness_ablation_table(rows: list[ResultRow]) -> str:
    return freshness_overall_table(rows)


def freshness_update_type_table(router_result: dict) -> str:
    headers = [
        "Subset",
        "Episodes",
        "Accuracy",
        "Stale Answer Rate",
        "Retrieval Calls",
        "Adapt Count",
        "Consolidation Count",
        "Rollback Count",
    ]
    table_rows: list[list[str]] = []
    for subset in ["stable_update", "volatile_update", "confirmation", "rollback_probe", "forgetting_probe"]:
        stats = router_result["subsets"].get(subset, {})
        table_rows.append(
            [
                subset,
                str(int(stats.get("episodes", 0))),
                f"{stats.get('accuracy', 0.0):.3f}",
                f"{stats.get('stale_answer_rate', 0.0):.3f}",
                f"{stats.get('retrieval_calls', 0.0):.3f}",
                f"{stats.get('adaptation_count', 0.0):.3f}",
                f"{stats.get('consolidation_count', 0.0):.3f}",
                f"{stats.get('rollback_count', 0.0):.3f}",
            ]
        )
    return _markdown_table(headers, table_rows)


def classify_freshness_failure(trace: dict) -> tuple[str, str]:
    action = trace["action"]
    metrics = trace.get("metrics", {})
    update_type = metrics.get("update_type", "unknown")
    stale_answer = bool(metrics.get("stale_answer", 0))
    rollback_triggered = bool(metrics.get("rollback_triggered", 0))
    if stale_answer and action in {"read_memory", "write_memory"}:
        return "stale memory used", f"{action} returned a stale answer during {update_type}"
    if action == "retrieve" and trace["reward"]["quality"] < 1.0:
        return "retrieval was still the right choice", f"retrieve was used on {update_type} but the evidence/model answer still missed"
    if update_type == "volatile_update" and action != "fast_adapt":
        return "should have adapted temporarily", f"{action} was chosen for a volatile update instead of fast_adapt"
    if update_type == "rollback_probe" and not rollback_triggered:
        return "rollback should have happened", "a rollback probe failed without triggering durable patch rollback"
    if action == "consolidate" and metrics.get("volatility_score", 0) > 0.3:
        return "consolidated unstable knowledge", "consolidate was used on an unstable or corrective update"
    return "should have retrieved fresh evidence", f"{action} missed an updated answer on {update_type}"


def build_freshness_error_audit(traces: list[dict], limit: int = 50) -> list[dict]:
    rows: list[dict] = []
    for trace in traces:
        if trace["reward"]["quality"] >= 1.0:
            continue
        label, rationale = classify_freshness_failure(trace)
        if label not in FRESHNESS_AUDIT_LABELS:
            raise ValueError(f"unexpected freshness audit label: {label}")
        rows.append(
            {
                "episode_id": trace["episode_id"],
                "subject": trace["subject"],
                "relation": trace["relation"],
                "action": trace["action"],
                "answer": trace["answer"],
                "gold_answer": trace["gold_answer"],
                "label": label,
                "rationale": rationale,
                "update_type": trace["metrics"].get("update_type", "unknown"),
                "evidence_ids": trace.get("evidence_ids", []),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def classify_public_freshqa_failure(trace: dict, calibrated_trace: dict | None = None) -> tuple[str, str]:
    action = trace["action"]
    features = trace.get("features", {})
    metrics = trace.get("metrics", {})
    update_type = str(metrics.get("update_type", "unknown"))
    answer_changed = bool(features.get("answer_changed_since_last_seen", 0.0))
    question_seen_before = bool(features.get("question_seen_before", 0.0))
    contradiction_risk = float(features.get("contradiction_risk", 0.0))
    retrieval_quality = float(features.get("retrieval_quality_estimate", 0.0))
    has_aliases = bool(features.get("has_aliases", 0.0))
    effective_scale = float(metrics.get("effective_future_value_scale", 0.0))
    stale_answer = bool(metrics.get("stale_answer", 0))
    calibrated_quality = None if calibrated_trace is None else float(calibrated_trace["reward"]["quality"])

    if action in {"read_memory", "write_memory"}:
        return (
            "premature memory reuse",
            f"{action} failed on public {update_type} with retrieval quality {retrieval_quality:.2f}",
        )
    if action in {"fast_adapt", "consolidate"}:
        if contradiction_risk >= 0.45 or update_type == "rollback_probe":
            return (
                "contradiction under-capture",
                f"{action} was used on a corrective public case with contradiction risk {contradiction_risk:.2f}",
            )
        return (
            "premature adaptation",
            f"{action} was chosen before retrieval-dominant evidence was exhausted on {update_type}",
        )
    if action == "retrieve":
        if stale_answer:
            return ("stale retrieval evidence", "retrieval stayed dominant, but the visible evidence still returned a stale answer")
        if has_aliases:
            return ("missing alias handling", "retrieval missed on a question with multiple accepted aliases")
        return ("retrieval was correctly dominant", "retrieve remained the correct action class even though the answer still missed")
    if calibrated_quality is not None and calibrated_quality > float(trace["reward"]["quality"]):
        return (
            "delayed-utility overshoot",
            f"the full router failed while the calibrated router recovered the example; effective future scale {effective_scale:.2f}",
        )
    if not question_seen_before or answer_changed or contradiction_risk >= 0.35:
        return (
            "delayed-utility overshoot",
            f"non-retrieval behavior was over-weighted on a changing-answer public case with future scale {effective_scale:.2f}",
        )
    return (
        "retrieval was correctly dominant",
        "the failure still looks retrieval-dominant rather than a reusable-memory success case",
    )


def build_public_freshqa_error_audit(
    router_traces: list[dict],
    calibrated_traces: list[dict] | None = None,
    limit: int = 100,
) -> list[dict]:
    calibrated_by_id = {trace["episode_id"]: trace for trace in (calibrated_traces or [])}
    rows: list[dict] = []
    for trace in router_traces:
        if trace["reward"]["quality"] >= 1.0:
            continue
        calibrated_trace = calibrated_by_id.get(trace["episode_id"])
        label, rationale = classify_public_freshqa_failure(trace, calibrated_trace)
        if label not in PUBLIC_FRESHQA_AUDIT_LABELS:
            raise ValueError(f"unexpected public FreshQA audit label: {label}")
        rows.append(
            {
                "episode_id": trace["episode_id"],
                "subject": trace["subject"],
                "relation": trace["relation"],
                "action": trace["action"],
                "answer": trace["answer"],
                "gold_answer": trace["gold_answer"],
                "label": label,
                "rationale": rationale,
                "update_type": trace["metrics"].get("update_type", "unknown"),
                "had_memory_before": bool(trace["metrics"].get("had_memory_before", 0)),
                "evidence_ids": trace.get("evidence_ids", []),
                "router_quality": float(trace["reward"]["quality"]),
                "calibrated_quality": None if calibrated_trace is None else float(calibrated_trace["reward"]["quality"]),
                "router_action": trace["action"],
                "calibrated_action": None if calibrated_trace is None else calibrated_trace["action"],
                "effective_future_value_scale": float(trace["metrics"].get("effective_future_value_scale", 0.0)),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def classify_public_freshqa_leakage_case(trace: dict) -> tuple[str, str]:
    metrics = trace.get("metrics", {})
    features = trace.get("features", {})
    evidence_ids = trace.get("evidence_ids", [])
    if any("future" in str(doc_id).lower() for doc_id in evidence_ids):
        return ("timestamp_leakage", "evidence ids suggest future-dated or invalid retrieval context")
    if float(features.get("has_aliases", 0.0)) >= 1.0:
        return ("alias_leniency", "the example has multiple accepted aliases and retrieval may benefit from permissive matching")
    if float(metrics.get("stale_answer", 0)) > 0:
        return ("evaluator_leniency", "retrieval still matched despite a stale-answer flag")
    if not evidence_ids:
        return ("derivation_bug", "retrieval was marked correct without visible evidence ids")
    return ("no_leakage_retrieval_legitimately_dominant", "retrieval appears legitimately dominant with visible timestamped support")


def build_public_freshqa_leakage_audit(
    always_retrieve_traces: list[dict],
    router_traces: list[dict],
    calibrated_traces: list[dict],
    *,
    limit: int = 100,
) -> list[dict]:
    always_by_id = {trace["episode_id"]: trace for trace in always_retrieve_traces}
    router_by_id = {trace["episode_id"]: trace for trace in router_traces}
    calibrated_by_id = {trace["episode_id"]: trace for trace in calibrated_traces}
    router_wrong = [
        trace for trace in router_traces
        if float(trace["reward"]["quality"]) < 1.0 and float(always_by_id.get(trace["episode_id"], {}).get("reward", {}).get("quality", 0.0)) >= 1.0
    ]
    calibrated_wrong = [
        trace for trace in calibrated_traces
        if float(trace["reward"]["quality"]) < 1.0 and float(always_by_id.get(trace["episode_id"], {}).get("reward", {}).get("quality", 0.0)) >= 1.0
    ]
    alias_cases = [
        trace for trace in always_retrieve_traces
        if float(trace["features"].get("has_aliases", 0.0)) >= 1.0
    ]
    rollback_cases = [
        trace for trace in always_retrieve_traces
        if trace["metrics"].get("update_type") in {"rollback_probe", "forgetting_probe"} or float(trace["features"].get("answer_changed_since_last_seen", 0.0)) >= 1.0
    ]

    selected: list[dict] = []
    used_ids: set[str] = set()

    def take(rows: list[dict], count: int, bucket: str) -> None:
        for trace in rows:
            if trace["episode_id"] in used_ids:
                continue
            label, rationale = classify_public_freshqa_leakage_case(trace)
            if label not in PUBLIC_LEAKAGE_AUDIT_LABELS:
                raise ValueError(f"unexpected public leakage audit label: {label}")
            calibrated_trace = calibrated_by_id.get(trace["episode_id"])
            selected.append(
                {
                    "episode_id": trace["episode_id"],
                    "bucket": bucket,
                    "label": label,
                    "rationale": rationale,
                    "question": trace.get("question"),
                    "subject": trace["subject"],
                    "relation": trace["relation"],
                    "action": trace["action"],
                    "answer": trace["answer"],
                    "gold_answer": trace["gold_answer"],
                    "always_retrieve_quality": float(always_by_id.get(trace["episode_id"], {}).get("reward", {}).get("quality", 0.0)),
                    "router_quality": None if bucket == "always_retrieve_correct_router_calibrated_wrong" else float(router_by_id.get(trace["episode_id"], {}).get("reward", {}).get("quality", 0.0)),
                    "router_calibrated_quality": None if calibrated_trace is None else float(calibrated_trace["reward"]["quality"]),
                    "update_type": trace["metrics"].get("update_type", "unknown"),
                    "has_aliases": bool(trace["features"].get("has_aliases", 0.0)),
                    "answer_changed_since_last_seen": bool(trace["features"].get("answer_changed_since_last_seen", 0.0)),
                    "evidence_ids": trace.get("evidence_ids", []),
                }
            )
            used_ids.add(trace["episode_id"])
            if len([item for item in selected if item["bucket"] == bucket]) >= count:
                break

    take(router_wrong, 40, "always_retrieve_correct_router_wrong")
    take(calibrated_wrong, 25, "always_retrieve_correct_router_calibrated_wrong")
    take(alias_cases, 20, "alias_heavy_or_multi_answer")
    take(rollback_cases, 15, "rollback_forgetting_or_answer_change")
    return selected[:limit]


def forgetting_report(router_result: dict) -> str:
    summary = router_result["summary"]
    return _markdown_table(
        ["Metric", "Value"],
        [
            ["Forgetting Probe Accuracy", f"{summary.get('forgetting_probe_accuracy', 0.0):.3f}"],
            ["Forgetting Delta", f"{summary.get('forgetting_delta', 0.0):.3f}"],
        ],
    )


def rollback_report(router_result: dict) -> str:
    summary = router_result["summary"]
    return _markdown_table(
        ["Metric", "Value"],
        [
            ["Rollback Count", str(int(summary.get("rollback_count", 0)))],
            ["Durable Patch Uses", str(int(summary.get("durable_patch_uses", 0)))],
        ],
    )


def export_popqa_bundle(
    output_dir: Path,
    suite_rows: list[ResultRow],
    ablation_rows: list[ResultRow],
    router_result: dict,
    router_trace_path: Path,
    audit_limit: int = 50,
    run_label: str = "post_fix",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_md = overall_comparison_table(suite_rows)
    recurring_md = recurring_vs_nonrecurring_table(suite_rows)
    ablation_md = ablation_table(ablation_rows)
    retrieval_svg = retrieval_reduction_chart(suite_rows)
    action_svg = action_distribution_chart(router_result["actions"])
    audit_rows = build_non_recurring_error_audit(router_result["traces"], limit=audit_limit)
    audit_md = error_audit_markdown(audit_rows)
    audit_json = json.dumps(audit_rows, indent=2)
    audit_summary_json = json.dumps(error_audit_summary(audit_rows), indent=2)

    (output_dir / "overall_comparison_table.md").write_text(overall_md, encoding="utf-8")
    (output_dir / "recurring_vs_nonrecurring_table.md").write_text(recurring_md, encoding="utf-8")
    (output_dir / "ablation_table.md").write_text(ablation_md, encoding="utf-8")
    (output_dir / "retrieval_calls_reduction.svg").write_text(retrieval_svg, encoding="utf-8")
    (output_dir / "action_distribution.svg").write_text(action_svg, encoding="utf-8")
    (output_dir / "error_audit_50.md").write_text(audit_md, encoding="utf-8")
    (output_dir / "error_audit_50.json").write_text(audit_json, encoding="utf-8")
    (output_dir / "error_audit_summary.json").write_text(audit_summary_json, encoding="utf-8")
    (output_dir / "router_summary.json").write_text(json.dumps(router_result, indent=2), encoding="utf-8")
    (output_dir / "router_trace_source.txt").write_text(str(router_trace_path), encoding="utf-8")
    (output_dir / "run_label.txt").write_text(run_label, encoding="utf-8")

    return {
        "overall_table": str(output_dir / "overall_comparison_table.md"),
        "recurring_table": str(output_dir / "recurring_vs_nonrecurring_table.md"),
        "ablation_table": str(output_dir / "ablation_table.md"),
        "retrieval_chart": str(output_dir / "retrieval_calls_reduction.svg"),
        "action_chart": str(output_dir / "action_distribution.svg"),
        "error_audit_markdown": str(output_dir / "error_audit_50.md"),
        "error_audit_json": str(output_dir / "error_audit_50.json"),
        "error_audit_summary": str(output_dir / "error_audit_summary.json"),
    }


def export_freshness_bundle(
    output_dir: Path,
    suite_rows: list[ResultRow],
    ablation_rows: list[ResultRow],
    router_result: dict,
    router_trace_path: Path,
    manifest: dict[str, object],
    audit_limit: int = 50,
    run_label: str = "freshness_v1",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_md = freshness_overall_table(suite_rows)
    update_type_md = freshness_update_type_table(router_result)
    ablation_md = freshness_ablation_table(ablation_rows)
    stale_svg = stale_answer_chart(suite_rows)
    action_svg = action_distribution_chart(router_result["actions"])
    audit_rows = build_freshness_error_audit(router_result["traces"], limit=audit_limit)
    audit_md = error_audit_markdown(audit_rows)
    audit_json = json.dumps(audit_rows, indent=2)
    audit_summary_json = json.dumps(error_audit_summary(audit_rows), indent=2)
    forgetting_md = forgetting_report(router_result)
    rollback_md = rollback_report(router_result)

    (output_dir / "baseline_comparison_table.md").write_text(overall_md, encoding="utf-8")
    (output_dir / "update_type_table.md").write_text(update_type_md, encoding="utf-8")
    (output_dir / "ablation_table.md").write_text(ablation_md, encoding="utf-8")
    (output_dir / "stale_answer_rate.svg").write_text(stale_svg, encoding="utf-8")
    (output_dir / "action_distribution.svg").write_text(action_svg, encoding="utf-8")
    (output_dir / "error_audit_50.md").write_text(audit_md, encoding="utf-8")
    (output_dir / "error_audit_50.json").write_text(audit_json, encoding="utf-8")
    (output_dir / "error_audit_summary.json").write_text(audit_summary_json, encoding="utf-8")
    (output_dir / "forgetting_report.md").write_text(forgetting_md, encoding="utf-8")
    (output_dir / "rollback_report.md").write_text(rollback_md, encoding="utf-8")
    (output_dir / "router_summary.json").write_text(json.dumps(router_result, indent=2), encoding="utf-8")
    (output_dir / "router_trace_source.txt").write_text(str(router_trace_path), encoding="utf-8")
    (output_dir / "run_label.txt").write_text(run_label, encoding="utf-8")
    (output_dir / "freshness_bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "baseline_table": str(output_dir / "baseline_comparison_table.md"),
        "update_type_table": str(output_dir / "update_type_table.md"),
        "ablation_table": str(output_dir / "ablation_table.md"),
        "stale_answer_chart": str(output_dir / "stale_answer_rate.svg"),
        "action_chart": str(output_dir / "action_distribution.svg"),
        "error_audit_markdown": str(output_dir / "error_audit_50.md"),
        "error_audit_json": str(output_dir / "error_audit_50.json"),
        "error_audit_summary": str(output_dir / "error_audit_summary.json"),
        "forgetting_report": str(output_dir / "forgetting_report.md"),
        "rollback_report": str(output_dir / "rollback_report.md"),
    }
