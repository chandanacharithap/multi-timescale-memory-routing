from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .types import EpisodeInput, SupportDoc


DEFAULT_UNIEDIT_ROOT = Path("data/benchmarks/uniedit")
UNIEDIT_BENCHMARK_SOURCE = "qizhou_uniedit"


def _flatten_answers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    answers: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            answers.extend(_flatten_answers(item))
    elif isinstance(raw, dict):
        for value in raw.values():
            answers.extend(_flatten_answers(value))
    return [item for item in dict.fromkeys(str(item).strip() for item in answers) if item]


def _tail_entity_answers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        value = raw.get("value")
        if isinstance(value, dict):
            label = value.get("label")
            if label is not None:
                return _flatten_answers(label)
        return _flatten_answers(raw)
    return _flatten_answers(raw)


def _support_doc(doc_id: str, text: str, answer: str, timestamp: int, source: str) -> SupportDoc:
    return SupportDoc(
        doc_id=doc_id,
        text=text,
        answer=answer,
        source=source,
        timestamp=timestamp,
        trust=0.95,
        relevance=0.92,
    )


def _load_domain(root: Path, split: str, domain: str) -> dict[str, Any]:
    return json.loads((root / split / f"{domain}.json").read_text(encoding="utf-8"))


def _sample_items(rows: dict[str, Any], sample_size: int, seed: int) -> list[tuple[str, dict[str, Any]]]:
    items = list(rows.items())
    if sample_size <= 0 or len(items) <= sample_size:
        return items
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(items)), sample_size))
    return [items[index] for index in indices]


def build_uniedit_benchmark(
    root_path: str | Path = DEFAULT_UNIEDIT_ROOT,
    *,
    split: str = "test",
    domains: list[str] | None = None,
    sample_per_domain: int = 200,
    sample_seed: int = 2026,
) -> tuple[list[EpisodeInput], list[SupportDoc], dict[str, object]]:
    root = Path(root_path)
    available_domains = sorted(path.stem for path in (root / split).glob("*.json"))
    selected_domains = domains or available_domains
    episodes: list[EpisodeInput] = []
    corpus_docs: dict[str, SupportDoc] = {}
    domain_counts: dict[str, int] = {}

    for domain_offset, domain in enumerate(selected_domains):
        rows = _sample_items(_load_domain(root, split, domain), sample_per_domain, sample_seed + domain_offset)
        domain_counts[domain] = len(rows)
        for row_index, (case_key, row) in enumerate(rows, start=1):
            edit = dict(row.get("edit", {}))
            generality = dict(row.get("generality", {}))
            locality = dict(row.get("locality", {}))
            case_id = f"{domain}::{case_key}"
            subject = str(edit.get("subject") or edit.get("head_entity", {}).get("label") or case_id)
            relation = str(edit.get("property", {}).get("id") or edit.get("property", {}).get("label") or f"{domain}::edit")
            new_answers = _flatten_answers(edit.get("target")) or ["unknown"]
            old_answers = _tail_entity_answers(edit.get("tail_entity"))
            update_question = str(edit.get("prompt") or f"Update the fact about {subject}.")
            timestamp = len(episodes) + 1

            update_doc = _support_doc(
                f"uniedit:{split}:{domain}:{case_key}:update",
                f"{update_question} Answer: {new_answers[0]}",
                new_answers[0],
                timestamp,
                f"uniedit:{split}:{domain}",
            )
            corpus_docs[update_doc.doc_id] = update_doc
            episodes.append(
                EpisodeInput(
                    episode_id=f"uniedit::{split}::{domain}::{case_key}::update",
                    question=update_question,
                    gold_answer=new_answers[0],
                    dataset_id="uniedit",
                    timestamp=timestamp,
                    subject=subject,
                    relation=relation,
                    domain=f"uniedit:{domain}",
                    support_docs=[update_doc],
                    parametric_answer=old_answers[0] if old_answers else None,
                    parametric_confidence=0.78 if old_answers else 0.25,
                    popularity_bin=0.5,
                    recurrence_hint=0.86,
                    stability_hint=0.78,
                    volatility_hint=0.2,
                    contradiction_hint=0.45,
                    freshness=True,
                    update_expected=True,
                    metadata={
                        "benchmark_source": UNIEDIT_BENCHMARK_SOURCE,
                        "split": split,
                        "domain_name": domain,
                        "case_id": case_id,
                        "probe_role": "update",
                        "update_type": "stable_update",
                        "source_support_count": 1,
                        "domain_change_rate": 0.2,
                        "question_seen_before": True,
                        "answer_changed_since_last_seen": bool(old_answers),
                        "weeks_since_last_change": 0,
                        "has_aliases": len(new_answers) > 1,
                        "prior_stale_answer_available": bool(old_answers),
                        "possible_answers": new_answers,
                        "stale_answers": old_answers,
                        "original_answers": old_answers,
                    },
                )
            )

            for probe_key, probe in generality.items():
                answers = _flatten_answers(probe.get("target")) or ["unknown"]
                prompt = str(probe.get("prompt") or f"{subject} generality probe")
                timestamp = len(episodes) + 1
                doc = _support_doc(
                    f"uniedit:{split}:{domain}:{case_key}:generality:{probe_key}",
                    f"{prompt} Answer: {answers[0]}",
                    answers[0],
                    timestamp,
                    f"uniedit:{split}:{domain}",
                )
                corpus_docs[doc.doc_id] = doc
                episodes.append(
                    EpisodeInput(
                        episode_id=f"uniedit::{split}::{domain}::{case_key}::generality::{probe_key}",
                        question=prompt,
                        gold_answer=answers[0],
                        dataset_id="uniedit",
                        timestamp=timestamp,
                        subject=subject,
                        relation=relation,
                        domain=f"uniedit:{domain}",
                        support_docs=[doc],
                        parametric_answer=old_answers[0] if old_answers else None,
                        parametric_confidence=0.55 if old_answers else 0.2,
                        popularity_bin=0.5,
                        recurrence_hint=0.8,
                        stability_hint=0.76,
                        volatility_hint=0.22,
                        contradiction_hint=0.28,
                        freshness=True,
                        update_expected=True,
                        metadata={
                            "benchmark_source": UNIEDIT_BENCHMARK_SOURCE,
                            "split": split,
                            "domain_name": domain,
                            "case_id": case_id,
                            "probe_role": "generality",
                            "probe_family": str(probe.get("path_type", "unknown")),
                            "update_type": "confirmation",
                            "source_support_count": 1,
                            "domain_change_rate": 0.22,
                            "question_seen_before": True,
                            "answer_changed_since_last_seen": True,
                            "weeks_since_last_change": 1,
                            "has_aliases": len(answers) > 1,
                            "prior_stale_answer_available": bool(old_answers),
                            "possible_answers": answers,
                            "stale_answers": old_answers,
                            "original_answers": old_answers,
                        },
                    )
                )

            for probe_key, probe in locality.items():
                answers = _flatten_answers(probe.get("target")) or ["unknown"]
                prompt = str(probe.get("prompt") or f"{subject} locality probe")
                timestamp = len(episodes) + 1
                doc = _support_doc(
                    f"uniedit:{split}:{domain}:{case_key}:locality:{probe_key}",
                    f"{prompt} Answer: {answers[0]}",
                    answers[0],
                    timestamp,
                    f"uniedit:{split}:{domain}",
                )
                corpus_docs[doc.doc_id] = doc
                episodes.append(
                    EpisodeInput(
                        episode_id=f"uniedit::{split}::{domain}::{case_key}::locality::{probe_key}",
                        question=prompt,
                        gold_answer=answers[0],
                        dataset_id="uniedit",
                        timestamp=timestamp,
                        subject=subject,
                        relation=f"{domain}::locality::{probe_key}",
                        domain=f"uniedit:{domain}",
                        support_docs=[doc],
                        parametric_answer=answers[0],
                        parametric_confidence=0.8,
                        popularity_bin=0.5,
                        recurrence_hint=0.62,
                        stability_hint=0.84,
                        volatility_hint=0.08,
                        contradiction_hint=0.05,
                        freshness=True,
                        update_expected=False,
                        metadata={
                            "benchmark_source": UNIEDIT_BENCHMARK_SOURCE,
                            "split": split,
                            "domain_name": domain,
                            "case_id": case_id,
                            "probe_role": "locality",
                            "probe_family": str(probe.get("loc_type", "unknown")),
                            "update_type": "confirmation",
                            "source_support_count": 1,
                            "domain_change_rate": 0.05,
                            "question_seen_before": True,
                            "answer_changed_since_last_seen": False,
                            "weeks_since_last_change": 1,
                            "has_aliases": len(answers) > 1,
                            "prior_stale_answer_available": False,
                            "possible_answers": answers,
                            "stale_answers": [],
                            "original_answers": answers,
                        },
                    )
                )

    manifest = {
        "benchmark_source": UNIEDIT_BENCHMARK_SOURCE,
        "root_path": str(root),
        "split": split,
        "domains": selected_domains,
        "sample_per_domain": sample_per_domain,
        "sample_seed": sample_seed,
        "domain_counts": domain_counts,
        "episodes": len(episodes),
        "corpus_docs": len(corpus_docs),
    }
    return episodes, list(corpus_docs.values()), manifest
