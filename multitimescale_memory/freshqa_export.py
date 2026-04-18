from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal


DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
SliceMode = Literal["main", "sequence"]


@dataclass(frozen=True, slots=True)
class FreshQASnapshotRow:
    snapshot_date: date
    snapshot_label: str
    row_id: str
    split: str
    question: str
    effective_year: str
    next_review: str
    false_premise: str
    num_hops: str
    fact_type: str
    source: str
    answers: list[str]
    note: str


@dataclass(frozen=True, slots=True)
class FreshQAGroupStats:
    unique_primary_answers: int
    snapshot_count: int
    change_count: int
    has_forgetting_probe: bool


def infer_snapshot_date(path: Path) -> date:
    match = DATE_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"could not infer snapshot date from filename {path.name}")
    return date.fromisoformat(match.group(1))


def load_freshqa_snapshot_csv(path: str | Path) -> list[FreshQASnapshotRow]:
    csv_path = Path(path)
    snapshot_date = infer_snapshot_date(csv_path)
    text = csv_path.read_text(encoding="utf-8")
    rows = list(csv.reader(text.splitlines()))
    def _is_header_row(row: list[str]) -> bool:
        if not row:
            return False
        first = row[0].strip().lstrip("\ufeff").strip('"').strip()
        return first == "id"

    header_index = next((index for index, row in enumerate(rows) if _is_header_row(row)), None)
    if header_index is None:
        header_line = next(
            (index for index, line in enumerate(text.splitlines()) if line.lstrip("\ufeff").startswith("id,")),
            None,
        )
        if header_line is not None:
            rows = list(csv.reader(text.splitlines()[header_line:]))
            header_index = next((index for index, row in enumerate(rows) if _is_header_row(row)), None)
    if header_index is None:
        raise ValueError(f"{csv_path} does not contain a FreshQA header row")
    header = rows[header_index]
    records: list[FreshQASnapshotRow] = []
    for raw in rows[header_index + 1 :]:
        if not raw or not any(cell.strip() for cell in raw):
            continue
        padded = raw + [""] * max(0, len(header) - len(raw))
        row = dict(zip(header, padded))
        answers = [row.get(f"answer_{index}", "").strip() for index in range(10)]
        answers = [answer for answer in answers if answer]
        if not row.get("question", "").strip() or not answers:
            continue
        records.append(
            FreshQASnapshotRow(
                snapshot_date=snapshot_date,
                snapshot_label=csv_path.stem,
                row_id=row.get("id", "").strip(),
                split=row.get("split", "").strip() or "TEST",
                question=row["question"].strip(),
                effective_year=row.get("effective_year", "").strip(),
                next_review=row.get("next_review", "").strip(),
                false_premise=row.get("false_premise", "").strip(),
                num_hops=row.get("num_hops", "").strip(),
                fact_type=row.get("fact_type", "").strip(),
                source=row.get("source", "").strip(),
                answers=answers,
                note=row.get("note", "").strip(),
            )
        )
    return records


def _normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def _stability_and_volatility(next_review: str, fact_type: str) -> tuple[float, float, str, str]:
    review_key = next_review.strip().lower()
    fact_key = fact_type.strip().lower()
    if review_key in {"frequently", "weekly", "monthly"}:
        stability = 0.25
    elif review_key in {"occasionally", "quarterly"}:
        stability = 0.55
    elif review_key in {"yearly", "annually", "rarely"}:
        stability = 0.85
    else:
        stability = 0.65
    if "fast" in fact_key:
        stability = min(stability, 0.35)
    elif "slow" in fact_key or "never" in fact_key:
        stability = max(stability, 0.8)
    volatility = max(0.05, min(0.95, 1.0 - stability))
    stability_class = "stable" if stability >= 0.75 else "mixed" if stability >= 0.45 else "volatile"
    volatility_class = "low" if volatility <= 0.25 else "medium" if volatility <= 0.55 else "high"
    return stability, volatility, stability_class, volatility_class


def _source_urls(raw_source: str) -> list[str]:
    return [line.strip() for line in raw_source.splitlines() if line.strip()]


def _compute_group_stats(group: list[FreshQASnapshotRow]) -> FreshQAGroupStats:
    ordered = sorted(group, key=lambda item: item.snapshot_date)
    primary_answers = [row.answers[0] for row in ordered]
    change_positions = [
        index
        for index in range(1, len(primary_answers))
        if primary_answers[index] != primary_answers[index - 1]
    ]
    last_change_position = change_positions[-1] if change_positions else -1
    has_forgetting_probe = bool(change_positions) and len(primary_answers) - last_change_position >= 3
    return FreshQAGroupStats(
        unique_primary_answers=len(set(primary_answers)),
        snapshot_count=len(ordered),
        change_count=len(change_positions),
        has_forgetting_probe=has_forgetting_probe,
    )


def _group_priority(item: tuple[list[FreshQASnapshotRow], FreshQAGroupStats], slice_mode: SliceMode) -> tuple[int, int, int, int, str]:
    group, stats = item
    sequence_bonus = int(stats.has_forgetting_probe) if slice_mode == "sequence" else 0
    return (
        -sequence_bonus,
        -stats.change_count,
        -stats.unique_primary_answers,
        -stats.snapshot_count,
        group[0].question,
    )


def _select_question_groups(
    question_groups: dict[str, list[FreshQASnapshotRow]],
    *,
    min_snapshots_per_question: int,
    max_questions: int | None,
    slice_mode: SliceMode,
) -> list[tuple[list[FreshQASnapshotRow], FreshQAGroupStats]]:
    ranked = []
    for group in question_groups.values():
        if len(group) < min_snapshots_per_question:
            continue
        stats = _compute_group_stats(group)
        if stats.change_count <= 0:
            continue
        ranked.append((group, stats))
    ranked.sort(key=lambda item: _group_priority(item, slice_mode))
    if max_questions is not None and max_questions > 0:
        ranked = ranked[:max_questions]
    return ranked


def _build_group_episodes(
    group: list[FreshQASnapshotRow],
    *,
    question_index: int,
    snapshot_count: int,
) -> list[dict[str, object]]:
    ordered = sorted(group, key=lambda item: item.snapshot_date)
    history_answers: list[str] = []
    last_change_position = -1
    confirmation_count_since_last_change = 0
    change_count_before = 0
    episodes: list[dict[str, object]] = []
    relation = ordered[0].fact_type.strip().lower() or "freshqa_fact"
    recurrence_hint = min(1.0, len(ordered) / max(1, snapshot_count))

    for position, row in enumerate(ordered):
        primary_answer = row.answers[0]
        previous_answer = history_answers[-1] if history_answers else primary_answer
        stability, volatility, stability_class, volatility_class = _stability_and_volatility(row.next_review, row.fact_type)
        question_seen_before = position > 0
        answer_changed_since_last_seen = question_seen_before and primary_answer != previous_answer
        rollback_probe = answer_changed_since_last_seen and primary_answer in history_answers[:-1]

        if not question_seen_before:
            update_type = "stable_update" if volatility <= 0.55 else "volatile_update"
            weeks_since_last_change = 0
            confirmation_count_before = 0
        elif answer_changed_since_last_seen:
            change_count_before += 1
            confirmation_count_before = 0
            confirmation_count_since_last_change = 0
            weeks_since_last_change = 0
            last_change_position = position
            update_type = "rollback_probe" if rollback_probe else ("volatile_update" if volatility > 0.55 else "stable_update")
        else:
            confirmation_count_before = confirmation_count_since_last_change
            confirmation_count_since_last_change += 1
            weeks_since_last_change = position - last_change_position if last_change_position >= 0 else position
            if change_count_before > 0 and confirmation_count_before >= 1:
                update_type = "forgetting_probe"
            else:
                update_type = "confirmation"

        stale_answers = [previous_answer] if answer_changed_since_last_seen else []
        possible_answers = list(dict.fromkeys(row.answers))
        prior_stale_answer_available = bool(stale_answers)
        contradiction_hint = 0.85 if rollback_probe else (0.65 if answer_changed_since_last_seen else 0.1)
        urls = _source_urls(row.source)
        support_docs = []
        for answer_index, answer in enumerate(possible_answers):
            source_url = urls[min(answer_index, len(urls) - 1)] if urls else "https://freshqa.local/source"
            support_docs.append(
                {
                    "doc_id": f"freshqa:{row.snapshot_date.isoformat()}:{row.row_id}:{answer_index}",
                    "text": f"FreshQA snapshot {row.snapshot_date.isoformat()} source: {source_url}",
                    "answer": answer,
                    "source": source_url,
                    "timestamp": 0,
                    "trust": max(0.55, 0.92 - 0.04 * answer_index),
                    "relevance": max(0.5, 0.95 - 0.08 * answer_index),
                }
            )

        metadata = {
            "update_id": f"freshqa::{question_index}",
            "update_type": update_type,
            "snapshot_date": row.snapshot_date.isoformat(),
            "source_support_count": len(possible_answers),
            "source_freshness": 1.0,
            "domain_change_rate": volatility,
            "stability_class": stability_class,
            "volatility_class": volatility_class,
            "stale_answers": stale_answers,
            "possible_answers": possible_answers,
            "rollback_probe": rollback_probe,
            "temporary_patch_horizon": 1 if update_type in {"volatile_update", "rollback_probe"} else 0,
            "effective_year": row.effective_year,
            "false_premise": row.false_premise,
            "num_hops": row.num_hops,
            "fact_type": row.fact_type,
            "note": row.note,
            "repeated_subject": question_seen_before,
            "repeated_relation": question_seen_before,
            "near_duplicate_question": question_seen_before,
            "recurring_fact": question_seen_before,
            "is_recurring_case": question_seen_before,
            "question_seen_before": question_seen_before,
            "answer_changed_since_last_seen": answer_changed_since_last_seen,
            "weeks_since_last_change": weeks_since_last_change,
            "has_aliases": len(possible_answers) > 1,
            "prior_stale_answer_available": prior_stale_answer_available,
            "confirmation_count_before": confirmation_count_before,
            "change_count_before": change_count_before,
        }
        episodes.append(
            {
                "episode_id": f"freshqa::{question_index}::{row.snapshot_date.isoformat()}::{row.row_id or position}",
                "question": row.question,
                "gold_answer": primary_answer,
                "dataset_id": "freshqa_public",
                "timestamp": 0,
                "subject": row.question,
                "relation": relation,
                "domain": row.fact_type or "freshness",
                "parametric_answer": previous_answer,
                "parametric_confidence": 0.82 if question_seen_before else 0.25,
                "popularity_bin": 0.5,
                "recurrence_hint": recurrence_hint,
                "stability_hint": stability,
                "volatility_hint": volatility,
                "contradiction_hint": contradiction_hint,
                "freshness": True,
                "update_expected": update_type != "forgetting_probe",
                "metadata": metadata,
                "support_docs": support_docs,
            }
        )
        history_answers.append(primary_answer)
    return episodes


def _episode_priority(slice_mode: SliceMode, episode: dict[str, object]) -> tuple[int, str, str]:
    if slice_mode == "main":
        return (0, str(episode["question"]).lower(), str(episode["episode_id"]))
    update_type = str(episode["metadata"]["update_type"])
    priority = {
        "rollback_probe": 0,
        "volatile_update": 1,
        "stable_update": 2,
        "forgetting_probe": 3,
        "confirmation": 4,
    }.get(update_type, 5)
    return (priority, str(episode["question"]).lower(), str(episode["episode_id"]))


def _finalize_episodes(episodes: list[dict[str, object]], slice_mode: SliceMode) -> list[dict[str, object]]:
    by_snapshot: dict[str, list[dict[str, object]]] = defaultdict(list)
    for episode in episodes:
        by_snapshot[str(episode["metadata"]["snapshot_date"])].append(episode)
    ordered_snapshots = sorted(by_snapshot)
    finalized: list[dict[str, object]] = []
    for snapshot_date in ordered_snapshots:
        bucket = sorted(by_snapshot[snapshot_date], key=lambda item: _episode_priority(slice_mode, item))
        finalized.extend(bucket)
    for timestamp, episode in enumerate(finalized, start=1):
        episode["timestamp"] = timestamp
        for doc in episode["support_docs"]:
            doc["timestamp"] = timestamp
    return finalized


def build_public_freshqa_export(
    snapshot_paths: list[str | Path],
    output_path: str | Path,
    *,
    min_snapshots_per_question: int = 3,
    max_questions: int | None = 24,
    test_split_only: bool = True,
    slice_mode: SliceMode = "main",
) -> dict[str, object]:
    snapshots: list[FreshQASnapshotRow] = []
    for snapshot_path in snapshot_paths:
        snapshots.extend(load_freshqa_snapshot_csv(snapshot_path))
    snapshots.sort(key=lambda row: (row.snapshot_date, row.question, row.row_id))

    question_groups: dict[str, list[FreshQASnapshotRow]] = {}
    for row in snapshots:
        if test_split_only and row.split.upper() != "TEST":
            continue
        question_groups.setdefault(_normalize_question(row.question), []).append(row)

    selected = _select_question_groups(
        question_groups,
        min_snapshots_per_question=min_snapshots_per_question,
        max_questions=max_questions if (max_questions is None or max_questions > 0) else None,
        slice_mode=slice_mode,
    )

    episodes: list[dict[str, object]] = []
    changed_questions = 0
    forgetting_probe_count = 0
    rollback_probe_count = 0
    for question_index, (group, stats) in enumerate(selected, start=1):
        changed_questions += int(stats.change_count > 0)
        group_episodes = _build_group_episodes(group, question_index=question_index, snapshot_count=len(snapshot_paths))
        forgetting_probe_count += sum(
            1 for episode in group_episodes if episode["metadata"]["update_type"] == "forgetting_probe"
        )
        rollback_probe_count += sum(
            1 for episode in group_episodes if episode["metadata"]["update_type"] == "rollback_probe"
        )
        episodes.extend(group_episodes)

    finalized = _finalize_episodes(episodes, slice_mode)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for episode in finalized:
            handle.write(json.dumps(episode, sort_keys=True) + "\n")

    manifest = {
        "output_path": str(output),
        "slice_mode": slice_mode,
        "snapshots": [str(Path(path)) for path in snapshot_paths],
        "snapshot_count": len(snapshot_paths),
        "tracked_questions": len(selected),
        "episodes": len(finalized),
        "changed_questions": changed_questions,
        "forgetting_probe_count": forgetting_probe_count,
        "rollback_probe_count": rollback_probe_count,
        "min_snapshots_per_question": min_snapshots_per_question,
        "max_questions": max_questions,
        "test_split_only": test_split_only,
    }
    return manifest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_freshqa_provenance_manifest(
    snapshot_paths: list[str | Path],
    *,
    derived_paths: list[str | Path],
    derivation_script_path: str | Path,
) -> dict[str, object]:
    source_rows = []
    for snapshot_path in snapshot_paths:
        snapshot_path = Path(snapshot_path)
        rows = load_freshqa_snapshot_csv(snapshot_path)
        source_rows.append(
            {
                "path": str(snapshot_path),
                "snapshot_date": infer_snapshot_date(snapshot_path).isoformat(),
                "sha256": file_sha256(snapshot_path),
                "row_count": len(rows),
                "test_row_count": sum(1 for row in rows if row.split.upper() == "TEST"),
            }
        )
    derived_rows = []
    for derived_path in derived_paths:
        derived_path = Path(derived_path)
        line_count = 0
        with derived_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    line_count += 1
        derived_rows.append(
            {
                "path": str(derived_path),
                "sha256": file_sha256(derived_path),
                "row_count": line_count,
            }
        )
    return {
        "source_files": source_rows,
        "derived_files": derived_rows,
        "derivation_script": {
            "path": str(derivation_script_path),
            "sha256": file_sha256(derivation_script_path),
        },
    }


def build_public_freshqa_traceability(
    derived_paths: dict[str, str | Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slice_name, path in derived_paths.items():
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                episode = json.loads(line)
                metadata = episode.get("metadata", {})
                rows.append(
                    {
                        "slice": slice_name,
                        "episode_id": episode["episode_id"],
                        "question": episode["question"],
                        "source_week": metadata.get("snapshot_date"),
                        "question_id": episode["episode_id"].split("::")[-1],
                        "update_type": metadata.get("update_type"),
                        "stale_answers": metadata.get("stale_answers", []),
                        "possible_answers": metadata.get("possible_answers", []),
                        "rollback_probe": bool(metadata.get("rollback_probe", False)),
                        "confirmation_count_before": int(metadata.get("confirmation_count_before", 0)),
                        "change_count_before": int(metadata.get("change_count_before", 0)),
                    }
                )
    rows.sort(key=lambda row: (str(row["source_week"]), str(row["question"]).lower(), str(row["slice"]), str(row["episode_id"])))
    return rows
