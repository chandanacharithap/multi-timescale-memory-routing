#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from multitimescale_memory.reporting import build_public_freshqa_leakage_audit  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _parse_doc_date(doc_id: str) -> str | None:
    parts = str(doc_id).split(":")
    if len(parts) >= 2 and len(parts[1]) == 10 and parts[1][4] == "-" and parts[1][7] == "-":
        return parts[1]
    return None


def _bool_label(flag: bool) -> str:
    return "yes" if flag else "no"


def _manual_verdict(row: dict[str, Any], dataset_row: dict[str, Any], always_trace: dict[str, Any]) -> tuple[str, str]:
    support_docs = dataset_row.get("support_docs", [])
    evidence_doc = support_docs[0] if support_docs else {}
    evidence_doc_id = str(evidence_doc.get("doc_id", ""))
    evidence_date = _parse_doc_date(evidence_doc_id)
    allowed_cutoff = str(dataset_row.get("metadata", {}).get("snapshot_date", ""))
    has_aliases = bool(dataset_row.get("metadata", {}).get("has_aliases", False))
    answer = str(always_trace.get("answer", "")).strip().lower()
    evidence_answer = str(evidence_doc.get("answer", "")).strip().lower()
    trivial_copy = bool(answer and evidence_answer and answer == evidence_answer)

    if not support_docs or not evidence_doc.get("source"):
        return ("leakage_risk", "retrieval was counted correct without a fully inspectable support record")
    if evidence_date and allowed_cutoff and evidence_date > allowed_cutoff:
        return ("leakage_risk", "retrieved evidence appears to post-date the allowed snapshot cutoff")
    if has_aliases:
        return ("suspect", "accepted answers are alias-heavy, so retrieval correctness may benefit from permissive matching")
    if trivial_copy:
        return ("clean", "retrieval answer matches visible snapshot support and the support date is within the allowed cutoff")
    return ("clean", "retrieval answer is supported by the frozen snapshot without an obvious leakage or alias issue")


def _audit_row(
    base_row: dict[str, Any],
    dataset_row: dict[str, Any],
    always_trace: dict[str, Any],
    router_trace: dict[str, Any] | None,
    calibrated_trace: dict[str, Any] | None,
) -> dict[str, Any]:
    support_docs = dataset_row.get("support_docs", [])
    evidence_doc = support_docs[0] if support_docs else {}
    evidence_doc_id = str(evidence_doc.get("doc_id", ""))
    evidence_date = _parse_doc_date(evidence_doc_id)
    allowed_cutoff = str(dataset_row.get("metadata", {}).get("snapshot_date", ""))
    has_aliases = bool(dataset_row.get("metadata", {}).get("has_aliases", False))
    answer = str(always_trace.get("answer", "")).strip().lower()
    evidence_answer = str(evidence_doc.get("answer", "")).strip().lower()
    trivial_copy = bool(answer and evidence_answer and answer == evidence_answer)
    stale_cache_risk = "low" if support_docs and evidence_date and evidence_date <= allowed_cutoff else "elevated"
    verdict, rationale = _manual_verdict(base_row, dataset_row, always_trace)
    return {
        "episode_id": base_row["episode_id"],
        "bucket": base_row["bucket"],
        "question": dataset_row.get("question"),
        "retrieved_evidence": {
            "doc_id": evidence_doc.get("doc_id"),
            "source": evidence_doc.get("source"),
            "answer": evidence_doc.get("answer"),
        },
        "evidence_timestamp": evidence_date,
        "allowed_cutoff": allowed_cutoff,
        "alias_permissiveness": "multi_answer" if has_aliases else "single_answer",
        "trivial_answer_copying_risk": _bool_label(trivial_copy),
        "stale_cache_risk": stale_cache_risk,
        "always_retrieve_quality": base_row.get("always_retrieve_quality"),
        "router_quality": base_row.get("router_quality"),
        "router_calibrated_quality": base_row.get("router_calibrated_quality"),
        "router_action": None if router_trace is None else router_trace.get("action"),
        "router_calibrated_action": None if calibrated_trace is None else calibrated_trace.get("action"),
        "manual_verdict": verdict,
        "manual_rationale": rationale,
    }


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# FreshQA Official Manual Leakage Audit",
        "",
        f"- audited rows: {len(rows)}",
        f"- clean: {summary['verdict_counts'].get('clean', 0)}",
        f"- suspect: {summary['verdict_counts'].get('suspect', 0)}",
        f"- leakage_risk: {summary['verdict_counts'].get('leakage_risk', 0)}",
        "",
        "| Episode | Bucket | Verdict | Evidence Date | Cutoff | Alias | Copy Risk | Cache Risk | Rationale |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['episode_id']}` | {row['bucket']} | {row['manual_verdict']} | "
            f"{row['evidence_timestamp'] or 'n/a'} | {row['allowed_cutoff'] or 'n/a'} | {row['alias_permissiveness']} | "
            f"{row['trivial_answer_copying_risk']} | {row['stale_cache_risk']} | {row['manual_rationale']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the official FreshQA manual leakage audit bundle.")
    parser.add_argument("--dataset-path", default="data/freshqa_public_full_main_v3.jsonl")
    parser.add_argument("--trace-root", default="artifacts/tmlr_official/freshqa_public_main_flan_base_v1/traces/google_flan-t5-base/seed_0")
    parser.add_argument("--output-dir", default="artifacts/tmlr_official/freshqa_leakage_audit_manual_v1")
    args = parser.parse_args()

    dataset_rows = {row["episode_id"]: row for row in _load_jsonl(Path(args.dataset_path))}
    trace_root = Path(args.trace_root)
    always = _load_jsonl(trace_root / "always_retrieve.jsonl")
    router = _load_jsonl(trace_root / "router.jsonl")
    calibrated = _load_jsonl(trace_root / "router_calibrated.jsonl")
    audit_seed_rows = build_public_freshqa_leakage_audit(always, router, calibrated, limit=100)
    always_by_id = {row["episode_id"]: row for row in always}
    router_by_id = {row["episode_id"]: row for row in router}
    calibrated_by_id = {row["episode_id"]: row for row in calibrated}

    reviewed_rows = [
        _audit_row(
            row,
            dataset_rows[row["episode_id"]],
            always_by_id[row["episode_id"]],
            router_by_id.get(row["episode_id"]),
            calibrated_by_id.get(row["episode_id"]),
        )
        for row in audit_seed_rows
    ]
    verdict_counts = Counter(row["manual_verdict"] for row in reviewed_rows)
    bucket_counts = Counter(row["bucket"] for row in reviewed_rows)
    summary = {
        "dataset_path": args.dataset_path,
        "trace_root": args.trace_root,
        "audited_rows": len(reviewed_rows),
        "bucket_counts": dict(bucket_counts),
        "verdict_counts": dict(verdict_counts),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manual_audit_100.json").write_text(json.dumps(reviewed_rows, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manual_audit_100.md").write_text(_markdown(reviewed_rows, summary), encoding="utf-8")
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(
            {
                "selection_source": "build_public_freshqa_leakage_audit",
                "selection_count": len(reviewed_rows),
                "dataset_path": args.dataset_path,
                "trace_root": args.trace_root,
                "review_policy": "case-by-case review of frozen support docs, snapshot cutoff, alias burden, and copying/cache risk",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
