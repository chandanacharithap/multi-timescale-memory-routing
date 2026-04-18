from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import EpisodeInput, SupportDoc


PUBLIC_FRESHNESS_BENCHMARK_SOURCE = "public_freshqa_track_v1"
DEFAULT_PUBLIC_FRESHNESS_CACHE = Path(".cache") / "freshqa_public.jsonl"
REQUIRED_TOP_LEVEL_FIELDS = ("episode_id", "question", "gold_answer", "timestamp", "subject", "relation", "support_docs")
REQUIRED_SUPPORT_DOC_FIELDS = ("doc_id", "text", "answer", "timestamp")


def _coerce_support_doc(raw: dict[str, Any]) -> SupportDoc:
    return SupportDoc(
        doc_id=str(raw["doc_id"]),
        text=str(raw["text"]),
        answer=str(raw.get("answer", "")),
        source=str(raw.get("source", "freshqa_public")),
        timestamp=int(raw["timestamp"]),
        trust=float(raw.get("trust", 1.0)),
        relevance=float(raw.get("relevance", 0.0)),
    )


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate_public_freshness_rows(raw_rows: list[dict[str, Any]]) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    previous_timestamp = -1
    repeated_pairs = 0
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(raw_rows):
        missing = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in row]
        if missing:
            errors.append(f"row {index} missing required fields: {', '.join(missing)}")
            continue
        try:
            timestamp = int(row["timestamp"])
        except (TypeError, ValueError):
            errors.append(f"row {index} has non-integer timestamp: {row.get('timestamp')!r}")
            continue
        if timestamp < previous_timestamp:
            errors.append(
                f"row {index} timestamp {timestamp} is earlier than the previous row timestamp {previous_timestamp}"
            )
        previous_timestamp = timestamp

        pair = (str(row["subject"]), str(row["relation"]))
        if pair in seen_pairs:
            repeated_pairs += 1
        else:
            seen_pairs.add(pair)

        support_docs = row.get("support_docs", [])
        if not isinstance(support_docs, list) or not support_docs:
            errors.append(f"row {index} must contain a non-empty support_docs list")
            continue
        for doc_index, doc in enumerate(support_docs):
            if not isinstance(doc, dict):
                errors.append(f"row {index} support_docs[{doc_index}] must be an object")
                continue
            missing_doc_fields = [field for field in REQUIRED_SUPPORT_DOC_FIELDS if field not in doc]
            if missing_doc_fields:
                errors.append(
                    f"row {index} support_docs[{doc_index}] missing fields: {', '.join(missing_doc_fields)}"
                )
                continue
            try:
                doc_timestamp = int(doc["timestamp"])
            except (TypeError, ValueError):
                errors.append(
                    f"row {index} support_docs[{doc_index}] has non-integer timestamp: {doc.get('timestamp')!r}"
                )
                continue
            if doc_timestamp > timestamp:
                warnings.append(
                    f"row {index} support_docs[{doc_index}] timestamp {doc_timestamp} exceeds episode timestamp {timestamp}"
                )
        metadata = row.get("metadata", {})
        if metadata and not isinstance(metadata, dict):
            errors.append(f"row {index} metadata must be an object when provided")
        elif isinstance(metadata, dict):
            if "update_type" not in metadata:
                warnings.append(f"row {index} metadata is missing update_type")
            if "source_support_count" not in metadata:
                warnings.append(f"row {index} metadata is missing source_support_count")

    if not raw_rows:
        errors.append("public freshness export is empty")

    return {
        "row_count": len(raw_rows),
        "errors": errors,
        "warnings": warnings,
        "repeated_subject_relation_pairs": repeated_pairs,
        "looks_like_sample_fixture": bool(
            raw_rows and all(str(row.get("dataset_id", "")).startswith("freshqa_public") for row in raw_rows) and len(raw_rows) <= 10
        ),
    }


def _expand_sequential_repeats(episodes: list[EpisodeInput], repeats: int) -> list[EpisodeInput]:
    if repeats <= 1 or not episodes:
        return episodes
    expanded: list[EpisodeInput] = []
    base_max_timestamp = max(episode.timestamp for episode in episodes)
    for repeat_index in range(repeats):
        offset = repeat_index * (base_max_timestamp + 1)
        for episode in episodes:
            metadata = dict(episode.metadata)
            metadata["sequence_repeat"] = repeat_index
            metadata["original_episode_id"] = episode.episode_id
            expanded.append(
                EpisodeInput(
                    episode_id=f"{episode.episode_id}::repeat{repeat_index}",
                    question=episode.question,
                    gold_answer=episode.gold_answer,
                    dataset_id=episode.dataset_id,
                    timestamp=episode.timestamp + offset,
                    subject=episode.subject,
                    relation=episode.relation,
                    domain=episode.domain,
                    support_docs=episode.support_docs,
                    parametric_answer=episode.parametric_answer,
                    parametric_confidence=episode.parametric_confidence,
                    popularity_bin=episode.popularity_bin,
                    recurrence_hint=min(1.0, episode.recurrence_hint + 0.05 * repeat_index),
                    stability_hint=episode.stability_hint,
                    volatility_hint=episode.volatility_hint,
                    contradiction_hint=episode.contradiction_hint,
                    freshness=episode.freshness,
                    update_expected=episode.update_expected,
                    metadata=metadata,
                )
            )
    return expanded


def build_public_freshness_benchmark(
    data_path: str | Path | None = None,
    limit: int | None = None,
    sequence_repeats: int = 1,
) -> tuple[list[EpisodeInput], list[SupportDoc], dict[str, str]]:
    path = Path(data_path) if data_path is not None else DEFAULT_PUBLIC_FRESHNESS_CACHE
    if not path.exists():
        raise FileNotFoundError(
            f"public freshness data not found at {path}. "
            "Provide --freshness-data-path pointing to a JSONL export of a public timestamped freshness benchmark."
        )

    raw_rows = _load_json_lines(path)
    if limit is not None and limit > 0:
        raw_rows = raw_rows[:limit]
    validation = validate_public_freshness_rows(raw_rows)
    if validation["errors"]:
        joined = "; ".join(validation["errors"])
        raise ValueError(f"public freshness export failed validation: {joined}")

    episodes: list[EpisodeInput] = []
    corpus_docs: dict[str, SupportDoc] = {}
    previous_timestamp = -1
    for row in raw_rows:
        timestamp = int(row["timestamp"])
        if timestamp < previous_timestamp:
            raise ValueError("public freshness rows must be sorted by non-decreasing timestamp")
        previous_timestamp = timestamp
        support_docs = [_coerce_support_doc(doc) for doc in row.get("support_docs", [])]
        for doc in support_docs:
            corpus_docs.setdefault(doc.doc_id, doc)
        metadata = dict(row.get("metadata", {}))
        metadata["benchmark_source"] = row.get("benchmark_source", PUBLIC_FRESHNESS_BENCHMARK_SOURCE)
        episodes.append(
            EpisodeInput(
                episode_id=str(row["episode_id"]),
                question=str(row["question"]),
                gold_answer=str(row["gold_answer"]),
                dataset_id=str(row.get("dataset_id", "freshqa_public")),
                timestamp=timestamp,
                subject=str(row["subject"]),
                relation=str(row["relation"]),
                domain=str(row.get("domain", "freshness")),
                support_docs=support_docs,
                parametric_answer=row.get("parametric_answer"),
                parametric_confidence=float(row.get("parametric_confidence", 0.0)),
                popularity_bin=float(row.get("popularity_bin", 0.0)),
                recurrence_hint=float(row.get("recurrence_hint", 0.0)),
                stability_hint=float(row.get("stability_hint", 0.0)),
                volatility_hint=float(row.get("volatility_hint", 0.0)),
                contradiction_hint=float(row.get("contradiction_hint", 0.0)),
                freshness=bool(row.get("freshness", True)),
                update_expected=bool(row.get("update_expected", True)),
                metadata=metadata,
            )
        )

    episodes = _expand_sequential_repeats(episodes, max(1, sequence_repeats))
    manifest = {
        "benchmark_source": PUBLIC_FRESHNESS_BENCHMARK_SOURCE,
        "data_path": str(path),
        "base_episodes": str(len(raw_rows)),
        "episodes": str(len(episodes)),
        "sequence_repeats": str(max(1, sequence_repeats)),
        "corpus_docs": str(len(corpus_docs)),
        "repeated_subject_relation_pairs": str(validation["repeated_subject_relation_pairs"]),
        "looks_like_sample_fixture": str(validation["looks_like_sample_fixture"]).lower(),
        "validation_warnings": str(len(validation["warnings"])),
    }
    return episodes, list(corpus_docs.values()), manifest
