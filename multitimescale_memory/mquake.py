from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import EpisodeInput, SupportDoc


DEFAULT_MQUAKE_PATH = Path("data/benchmarks/mquake/MQuAKE-CF-3k.json")
MQUAKE_BENCHMARK_SOURCE = "princeton_nlp_mquake_cf"


def _normalize_answers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    answers: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            answers.extend(_normalize_answers(item))
    elif isinstance(raw, dict):
        for value in raw.values():
            answers.extend(_normalize_answers(value))
    return [item for item in dict.fromkeys(str(item).strip() for item in answers) if item]


def _support_doc(doc_id: str, question: str, answer: str, timestamp: int, source: str) -> SupportDoc:
    return SupportDoc(
        doc_id=doc_id,
        text=f"{question} Answer: {answer}",
        answer=answer,
        source=source,
        timestamp=timestamp,
        trust=0.95,
        relevance=0.95,
    )


def load_mquake_cases(path: str | Path = DEFAULT_MQUAKE_PATH, limit: int | None = None) -> list[dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def build_mquake_benchmark(
    data_path: str | Path = DEFAULT_MQUAKE_PATH,
    *,
    limit: int | None = None,
) -> tuple[list[EpisodeInput], list[SupportDoc], dict[str, object]]:
    cases = load_mquake_cases(data_path, limit=limit)
    episodes: list[EpisodeInput] = []
    corpus_docs: dict[str, SupportDoc] = {}
    update_count = 0
    single_hop_count = 0
    multi_hop_count = 0

    for case in cases:
        case_id = int(case["case_id"])
        rewrite = case["requested_rewrite"][0]
        relation = str(rewrite.get("relation_id") or rewrite.get("prompt") or "edited_fact")
        rewrite_subject = str(rewrite["subject"])
        new_answer = str(case.get("new_answer") or rewrite.get("target_new", {}).get("str", "unknown"))
        old_answer = str(case.get("answer") or rewrite.get("target_true", {}).get("str", "unknown"))
        new_aliases = _normalize_answers(case.get("new_answer_alias")) or [new_answer]
        old_aliases = _normalize_answers(case.get("answer_alias")) or [old_answer]

        timestamp = len(episodes) + 1
        update_doc = _support_doc(
            f"mquake:{case_id}:update",
            str(rewrite.get("question") or rewrite.get("prompt", "")).replace("{}", rewrite_subject),
            new_answer,
            timestamp,
            "mquake",
        )
        corpus_docs[update_doc.doc_id] = update_doc
        episodes.append(
            EpisodeInput(
                episode_id=f"mquake::{case_id}::update",
                question=str(rewrite.get("question") or rewrite.get("prompt", "")).replace("{}", rewrite_subject),
                gold_answer=new_answer,
                dataset_id="mquake",
                timestamp=timestamp,
                subject=rewrite_subject,
                relation=relation,
                domain="mquake",
                support_docs=[update_doc],
                parametric_answer=old_answer,
                parametric_confidence=0.82,
                popularity_bin=0.5,
                recurrence_hint=0.92,
                stability_hint=0.82,
                volatility_hint=0.2,
                contradiction_hint=0.55,
                freshness=True,
                update_expected=True,
                metadata={
                    "benchmark_source": MQUAKE_BENCHMARK_SOURCE,
                    "case_id": case_id,
                    "probe_role": "update",
                    "update_type": "stable_update",
                    "source_support_count": 1,
                    "domain_change_rate": 0.2,
                    "question_seen_before": True,
                    "answer_changed_since_last_seen": True,
                    "weeks_since_last_change": 0,
                    "has_aliases": len(new_aliases) > 1,
                    "prior_stale_answer_available": True,
                    "possible_answers": new_aliases,
                    "stale_answers": old_aliases,
                    "original_answers": old_aliases,
                    "expected_new_answer": True,
                    "mquake_relation_id": relation,
                },
            )
        )
        update_count += 1

        for hop_index, hop in enumerate(case.get("new_single_hops", []), start=1):
            hop_answers = _normalize_answers(hop.get("answer_alias")) or [str(hop.get("answer", "unknown"))]
            hop_answer = hop_answers[0]
            hop_question = str(hop.get("question") or hop.get("cloze") or f"single-hop-{hop_index}")
            uses_rewrite_relation = hop_answer in new_aliases or rewrite_subject.lower() in hop_question.lower()
            hop_subject = rewrite_subject if uses_rewrite_relation else f"mquake_context::{case_id}::{hop_index}"
            hop_relation = relation if uses_rewrite_relation else f"single_hop::{hop_index}"
            timestamp = len(episodes) + 1
            doc = _support_doc(f"mquake:{case_id}:single:{hop_index}", hop_question, hop_answer, timestamp, "mquake")
            corpus_docs[doc.doc_id] = doc
            episodes.append(
                EpisodeInput(
                    episode_id=f"mquake::{case_id}::single_hop::{hop_index}",
                    question=hop_question,
                    gold_answer=hop_answer,
                    dataset_id="mquake",
                    timestamp=timestamp,
                    subject=hop_subject,
                    relation=hop_relation,
                    domain="mquake",
                    support_docs=[doc],
                    parametric_answer=old_answer if uses_rewrite_relation else None,
                    parametric_confidence=0.7 if uses_rewrite_relation else 0.25,
                    popularity_bin=0.5,
                    recurrence_hint=0.88 if uses_rewrite_relation else 0.35,
                    stability_hint=0.8 if uses_rewrite_relation else 0.65,
                    volatility_hint=0.2 if uses_rewrite_relation else 0.1,
                    contradiction_hint=0.35 if uses_rewrite_relation else 0.05,
                    freshness=True,
                    update_expected=uses_rewrite_relation,
                    metadata={
                        "benchmark_source": MQUAKE_BENCHMARK_SOURCE,
                        "case_id": case_id,
                        "probe_role": "single_hop",
                        "update_type": "confirmation" if uses_rewrite_relation else "stable_update",
                        "source_support_count": 1,
                        "domain_change_rate": 0.2 if uses_rewrite_relation else 0.05,
                        "question_seen_before": uses_rewrite_relation,
                        "answer_changed_since_last_seen": uses_rewrite_relation,
                        "weeks_since_last_change": 1 if uses_rewrite_relation else 0,
                        "has_aliases": len(hop_answers) > 1,
                        "prior_stale_answer_available": uses_rewrite_relation,
                        "possible_answers": hop_answers,
                        "stale_answers": old_aliases if uses_rewrite_relation else [],
                        "original_answers": old_aliases,
                        "expected_new_answer": uses_rewrite_relation,
                        "probe_family": "single_hop",
                    },
                )
            )
            single_hop_count += 1

        for probe_index, question in enumerate(case.get("questions", []), start=1):
            timestamp = len(episodes) + 1
            doc = _support_doc(f"mquake:{case_id}:multi:{probe_index}", str(question), new_answer, timestamp, "mquake")
            corpus_docs[doc.doc_id] = doc
            episodes.append(
                EpisodeInput(
                    episode_id=f"mquake::{case_id}::multi_hop::{probe_index}",
                    question=str(question),
                    gold_answer=new_answer,
                    dataset_id="mquake",
                    timestamp=timestamp,
                    subject=rewrite_subject,
                    relation=relation,
                    domain="mquake",
                    support_docs=[doc],
                    parametric_answer=old_answer,
                    parametric_confidence=0.68,
                    popularity_bin=0.5,
                    recurrence_hint=0.9,
                    stability_hint=0.8,
                    volatility_hint=0.25,
                    contradiction_hint=0.45,
                    freshness=True,
                    update_expected=True,
                    metadata={
                        "benchmark_source": MQUAKE_BENCHMARK_SOURCE,
                        "case_id": case_id,
                        "probe_role": "multi_hop",
                        "update_type": "rollback_probe",
                        "source_support_count": 1,
                        "domain_change_rate": 0.25,
                        "question_seen_before": True,
                        "answer_changed_since_last_seen": True,
                        "weeks_since_last_change": 1,
                        "has_aliases": len(new_aliases) > 1,
                        "prior_stale_answer_available": True,
                        "possible_answers": new_aliases,
                        "stale_answers": old_aliases,
                        "original_answers": old_aliases,
                        "expected_new_answer": True,
                        "probe_family": "multi_hop",
                    },
                )
            )
            multi_hop_count += 1

    manifest = {
        "benchmark_source": MQUAKE_BENCHMARK_SOURCE,
        "data_path": str(Path(data_path)),
        "cases": len(cases),
        "episodes": len(episodes),
        "update_episodes": update_count,
        "single_hop_probes": single_hop_count,
        "multi_hop_probes": multi_hop_count,
    }
    return episodes, list(corpus_docs.values()), manifest
