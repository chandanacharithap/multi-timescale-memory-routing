from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .types import EpisodeInput, SupportDoc


DEFAULT_KNOWEDIT_ROOT = Path("data/benchmarks/knowedit")
KNOWEDIT_BENCHMARK_SOURCE = "zjunlp_knowedit"
KNOWEDIT_SUBSET_PATHS = {
    "WikiBio": "WikiBio/wikibio-test-all.json",
    "ZsRE": "ZsRE/ZsRE-test-all.json",
    "wiki_counterfact": "wiki_counterfact/test_cf.json",
    "wiki_recent": "wiki_recent/recent_test.json",
}


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


def _load_subset(root: Path, subset: str) -> list[dict[str, Any]]:
    relative = KNOWEDIT_SUBSET_PATHS[subset]
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _sample_rows(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0 or len(rows) <= sample_size:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[index] for index in indices]


def build_knowedit_benchmark(
    root_path: str | Path = DEFAULT_KNOWEDIT_ROOT,
    *,
    subsets: list[str] | None = None,
    sample_per_subset: int = 200,
    sample_seed: int = 2026,
) -> tuple[list[EpisodeInput], list[SupportDoc], dict[str, object]]:
    root = Path(root_path)
    selected_subsets = subsets or list(KNOWEDIT_SUBSET_PATHS)
    episodes: list[EpisodeInput] = []
    corpus_docs: dict[str, SupportDoc] = {}
    subset_counts: dict[str, int] = {}

    for subset_offset, subset in enumerate(selected_subsets):
        rows = _sample_rows(_load_subset(root, subset), sample_per_subset, sample_seed + subset_offset)
        subset_counts[subset] = len(rows)
        for row_index, row in enumerate(rows, start=1):
            case_id = f"{subset}::{row_index}"
            subject = str(row.get("subject") or row.get("concept") or f"{subset}-subject-{row_index}")
            update_question = str(
                row.get("prompt")
                or row.get("rephrase_prompt")
                or f"Continue the updated biography for {subject}."
            )
            new_answers = _flatten_answers(row.get("target_new") or row.get("labels")) or ["unknown"]
            old_answers = _flatten_answers(row.get("ground_truth")) or []
            relation = f"{subset}::edit"
            timestamp = len(episodes) + 1
            if subset == "WikiBio":
                support_text = f"{row.get('text', '')}\nContinuation: {new_answers[0]}"
            else:
                support_text = f"{update_question} Answer: {new_answers[0]}"
            update_doc = _support_doc(
                f"knowedit:{subset}:{row_index}:update",
                support_text,
                new_answers[0],
                timestamp,
                f"knowedit:{subset}",
            )
            corpus_docs[update_doc.doc_id] = update_doc
            episodes.append(
                EpisodeInput(
                    episode_id=f"knowedit::{subset}::{row_index}::update",
                    question=update_question,
                    gold_answer=new_answers[0],
                    dataset_id="knowedit",
                    timestamp=timestamp,
                    subject=subject,
                    relation=relation,
                    domain=f"knowedit:{subset}",
                    support_docs=[update_doc],
                    parametric_answer=old_answers[0] if old_answers else None,
                    parametric_confidence=0.78 if old_answers else 0.25,
                    popularity_bin=0.5,
                    recurrence_hint=0.88,
                    stability_hint=0.8,
                    volatility_hint=0.2,
                    contradiction_hint=0.45,
                    freshness=True,
                    update_expected=True,
                    metadata={
                        "benchmark_source": KNOWEDIT_BENCHMARK_SOURCE,
                        "subset": subset,
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

            locality = row.get("locality", {})
            for family, probes in locality.items():
                for probe_index, probe in enumerate(probes, start=1):
                    answers = _flatten_answers(probe.get("ground_truth")) or ["unknown"]
                    timestamp = len(episodes) + 1
                    doc = _support_doc(
                        f"knowedit:{subset}:{row_index}:locality:{family}:{probe_index}",
                        f"{probe['prompt']} Answer: {answers[0]}",
                        answers[0],
                        timestamp,
                        f"knowedit:{subset}",
                    )
                    corpus_docs[doc.doc_id] = doc
                    episodes.append(
                        EpisodeInput(
                            episode_id=f"knowedit::{subset}::{row_index}::locality::{family}::{probe_index}",
                            question=str(probe["prompt"]),
                            gold_answer=answers[0],
                            dataset_id="knowedit",
                            timestamp=timestamp,
                            subject=subject,
                            relation=f"{subset}::locality::{family}::{probe_index}",
                            domain=f"knowedit:{subset}",
                            support_docs=[doc],
                            parametric_answer=answers[0],
                            parametric_confidence=0.8,
                            popularity_bin=0.5,
                            recurrence_hint=0.65,
                            stability_hint=0.82,
                            volatility_hint=0.08,
                            contradiction_hint=0.05,
                            freshness=True,
                            update_expected=False,
                            metadata={
                                "benchmark_source": KNOWEDIT_BENCHMARK_SOURCE,
                                "subset": subset,
                                "case_id": case_id,
                                "probe_role": "locality",
                                "probe_family": family,
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

            portability = row.get("portability", {})
            for family, probes in portability.items():
                for probe_index, probe in enumerate(probes, start=1):
                    answers = _flatten_answers(probe.get("ground_truth")) or ["unknown"]
                    timestamp = len(episodes) + 1
                    doc = _support_doc(
                        f"knowedit:{subset}:{row_index}:portability:{family}:{probe_index}",
                        f"{probe['prompt']} Answer: {answers[0]}",
                        answers[0],
                        timestamp,
                        f"knowedit:{subset}",
                    )
                    corpus_docs[doc.doc_id] = doc
                    episodes.append(
                        EpisodeInput(
                            episode_id=f"knowedit::{subset}::{row_index}::portability::{family}::{probe_index}",
                            question=str(probe["prompt"]),
                            gold_answer=answers[0],
                            dataset_id="knowedit",
                            timestamp=timestamp,
                            subject=subject,
                            relation=relation,
                            domain=f"knowedit:{subset}",
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
                                "benchmark_source": KNOWEDIT_BENCHMARK_SOURCE,
                                "subset": subset,
                                "case_id": case_id,
                                "probe_role": "portability" if family != "Forgetfulness" else "forgetfulness",
                                "probe_family": family,
                                "update_type": "rollback_probe" if family != "Forgetfulness" else "forgetting_probe",
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

    manifest = {
        "benchmark_source": KNOWEDIT_BENCHMARK_SOURCE,
        "root_path": str(root),
        "subsets": selected_subsets,
        "sample_per_subset": sample_per_subset,
        "sample_seed": sample_seed,
        "subset_counts": subset_counts,
        "episodes": len(episodes),
        "corpus_docs": len(corpus_docs),
    }
    return episodes, list(corpus_docs.values()), manifest
