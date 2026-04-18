from __future__ import annotations

import json
import math
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .types import EpisodeInput, SupportDoc

POPQA_ROWS_URL = "https://datasets-server.huggingface.co/rows"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
POPQA_TOTAL_ROWS = 14267


def _fetch_json(url: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(6):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "multitimescale-memory-controller/0.1 (research prototype)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 502, 503, 504}:
                raise
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(min(2**attempt, 20))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"failed to fetch JSON from {url}")


def parse_answer_list(raw: str) -> list[str]:
    answers = json.loads(raw)
    return [str(item).strip() for item in answers if str(item).strip()]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def question_skeleton(question: str, subject: str) -> str:
    normalized_question = normalize_text(question)
    normalized_subject = normalize_text(subject)
    return normalized_question.replace(normalized_subject, "<subj>")


def popularity_bin(value: int) -> float:
    return min(1.0, math.log10(max(value, 1)) / 7.0)


def _rows_cache_path(cache_dir: Path, limit: int) -> Path:
    return cache_dir / f"popqa_rows_{limit}.json"


def _subset_manifest_path(cache_dir: Path, limit: int) -> Path:
    return cache_dir / f"popqa_subset_{limit}.json"


def _corpus_cache_path(cache_dir: Path, limit: int) -> Path:
    return cache_dir / f"popqa_corpus_{limit}.json"


def _cached_wikipedia_titles(cache_dir: Path) -> set[str]:
    wiki_cache = cache_dir / "wiki"
    return {urllib.parse.unquote(path.stem).replace("_", " ") for path in wiki_cache.glob("*.json")}


def fetch_popqa_rows(limit: int, cache_dir: Path, allow_network: bool) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _rows_cache_path(cache_dir, limit)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if not allow_network:
        raise FileNotFoundError(
            f"cached PopQA rows not found at {cache_path}. Run the PopQA prefetch command first."
        )

    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 100
    while len(rows) < limit:
        query = urllib.parse.urlencode(
            {
                "dataset": "akariasai/PopQA",
                "config": "default",
                "split": "test",
                "offset": offset,
                "length": min(page_size, limit - len(rows)),
            }
        )
        data = _fetch_json(f"{POPQA_ROWS_URL}?{query}")
        page_rows = [item["row"] for item in data.get("rows", [])]
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)
    rows = rows[:limit]
    cache_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def fetch_wikipedia_summary(title: str, cache_dir: Path, allow_network: bool) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = urllib.parse.quote(title.replace(" ", "_"))
    cache_path = cache_dir / f"{slug}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return str(payload.get("extract") or payload.get("description") or "")
    if not allow_network:
        raise FileNotFoundError(
            f"cached Wikipedia summary not found for {title}. Run the PopQA prefetch command first."
        )
    payload = _fetch_json(WIKIPEDIA_SUMMARY_URL.format(title=slug))
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(payload.get("extract") or payload.get("description") or "")


def _annotate_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subject_counts: dict[str, int] = {}
    relation_counts: dict[str, int] = {}
    skeleton_counts: dict[str, int] = {}
    question_counts: dict[str, int] = {}
    subject_relation_counts: dict[tuple[str, str], int] = {}

    for row in rows:
        subj_key = normalize_text(row["subj"])
        relation_key = normalize_text(row["prop"])
        skeleton_key = question_skeleton(row["question"], row["subj"])
        question_key = normalize_text(row["question"])
        subject_relation_key = (subj_key, relation_key)
        subject_counts[subj_key] = subject_counts.get(subj_key, 0) + 1
        relation_counts[relation_key] = relation_counts.get(relation_key, 0) + 1
        skeleton_counts[skeleton_key] = skeleton_counts.get(skeleton_key, 0) + 1
        question_counts[question_key] = question_counts.get(question_key, 0) + 1
        subject_relation_counts[subject_relation_key] = subject_relation_counts.get(subject_relation_key, 0) + 1

    annotated: list[dict[str, Any]] = []
    for row in rows:
        annotated_row = dict(row)
        subj_key = normalize_text(row["subj"])
        relation_key = normalize_text(row["prop"])
        skeleton_key = question_skeleton(row["question"], row["subj"])
        question_key = normalize_text(row["question"])
        subject_relation_key = (subj_key, relation_key)
        repeated_subject = subject_counts[subj_key] > 1
        repeated_relation = relation_counts[relation_key] > 1
        recurring_fact = subject_relation_counts[subject_relation_key] > 1
        near_duplicate_question = question_counts[question_key] > 1 or (
            recurring_fact and skeleton_counts[skeleton_key] > 1
        )
        strong_recurring = repeated_subject or near_duplicate_question or recurring_fact
        annotated_row["repeated_subject"] = repeated_subject
        annotated_row["repeated_relation"] = repeated_relation
        annotated_row["near_duplicate_question"] = near_duplicate_question
        annotated_row["recurring_fact"] = recurring_fact
        annotated_row["is_recurring_case"] = strong_recurring
        pop_value = int(row["s_pop"])
        annotated_row["selection_score"] = [
            int(recurring_fact),
            int(repeated_subject),
            int(near_duplicate_question),
            int(repeated_relation),
            pop_value,
        ]
        annotated.append(annotated_row)
    return annotated


def _select_subset(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    annotated = _annotate_candidates(rows)
    strong_recurring = [row for row in annotated if row["is_recurring_case"]]
    moderate = [row for row in annotated if row["repeated_relation"] and not row["is_recurring_case"]]
    nonrecurring = [row for row in annotated if not row["repeated_relation"] and not row["is_recurring_case"]]

    strong_recurring.sort(key=lambda row: tuple(row["selection_score"]), reverse=True)
    moderate.sort(key=lambda row: tuple(row["selection_score"]), reverse=True)
    nonrecurring.sort(key=lambda row: (int(row["s_pop"]), int(row["id"])), reverse=True)

    selected: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    strong_target = min(len(strong_recurring), max(int(limit * 0.5), min(50, limit)))
    recurring_target = min(len(strong_recurring) + len(moderate), max(int(limit * 0.7), strong_target))

    def take(rows_to_take: list[dict[str, Any]], target_size: int) -> None:
        for row in rows_to_take:
            if len(selected) >= target_size:
                break
            row_id = int(row["id"])
            if row_id in seen_ids:
                continue
            selected.append(row)
            seen_ids.add(row_id)

    take(strong_recurring, strong_target)
    take(moderate, recurring_target)
    take(nonrecurring, limit)
    take(strong_recurring, limit)
    take(moderate, limit)
    return selected[:limit]


def _popqa_metadata(row: dict[str, Any]) -> dict[str, Any]:
    repeated_subject = bool(row["repeated_subject"])
    repeated_relation = bool(row["repeated_relation"])
    near_duplicate_question = bool(row["near_duplicate_question"])
    recurring_fact = bool(row["recurring_fact"])
    strong_recurring = bool(row["is_recurring_case"])
    recurrence_hint = 0.15
    if repeated_relation:
        recurrence_hint = 0.35
    if near_duplicate_question:
        recurrence_hint = 0.7
    if repeated_subject:
        recurrence_hint = 0.8
    if recurring_fact:
        recurrence_hint = 0.9
    recurrence_hint = max(recurrence_hint, popularity_bin(int(row["s_pop"])) * 0.3)

    return {
        "possible_answers": parse_answer_list(row["possible_answers"]),
        "source_support_count": 1,
        "domain_change_rate": 0.05,
        "disable_fast_adapt": True,
        "disable_consolidate": True,
        "wikipedia_doc_id": f"wiki:{row['s_wiki_title'].replace(' ', '_')}",
        "repeated_subject": repeated_subject,
        "repeated_relation": repeated_relation,
        "near_duplicate_question": near_duplicate_question,
        "recurring_fact": recurring_fact,
        "is_recurring_case": strong_recurring,
        "recurrence_hint": recurrence_hint,
    }


def _serialize_subset_manifest(rows: list[dict[str, Any]], cache_dir: Path, limit: int) -> None:
    manifest_rows = []
    for row in rows:
        manifest_rows.append(
            {
                "id": row["id"],
                "subj": row["subj"],
                "prop": row["prop"],
                "question": row["question"],
                "s_pop": row["s_pop"],
                "s_wiki_title": row["s_wiki_title"],
                "possible_answers": row["possible_answers"],
                "metadata": _popqa_metadata(row),
            }
        )
    _subset_manifest_path(cache_dir, limit).write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")


def _serialize_corpus(rows: list[dict[str, Any]], cache_dir: Path, limit: int, allow_network: bool) -> None:
    wiki_cache = cache_dir / "wiki"
    docs: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for row in rows:
        title = row["s_wiki_title"]
        doc_id = f"wiki:{title.replace(' ', '_')}"
        if doc_id in seen_doc_ids:
            continue
        summary = fetch_wikipedia_summary(title, wiki_cache, allow_network=allow_network)
        docs.append(
            {
                "doc_id": doc_id,
                "text": summary,
                "answer": "",
                "source": "wikipedia",
                "timestamp": 0,
                "trust": 0.8,
                "relevance": 0.0,
            }
        )
        seen_doc_ids.add(doc_id)
    _corpus_cache_path(cache_dir, limit).write_text(json.dumps(docs, indent=2), encoding="utf-8")


def prefetch_popqa_benchmark(
    limit: int,
    cache_dir: Path,
    candidate_limit: int | None = None,
    cached_only: bool = False,
) -> dict[str, int]:
    popqa_cache = cache_dir / "popqa"
    requested_candidate_limit = candidate_limit or min(max(limit * 20, 2000), POPQA_TOTAL_ROWS)
    rows = fetch_popqa_rows(
        limit=requested_candidate_limit,
        cache_dir=popqa_cache,
        allow_network=not cached_only,
    )
    if cached_only:
        cached_titles = _cached_wikipedia_titles(cache_dir)
        rows = [row for row in rows if row["s_wiki_title"] in cached_titles]
        if len(rows) < limit:
            raise FileNotFoundError(
                f"only {len(rows)} PopQA rows have cached local evidence, but limit={limit} was requested"
            )
    subset = _select_subset(rows, limit)
    _serialize_subset_manifest(subset, cache_dir, limit)
    _serialize_corpus(subset, cache_dir, limit, allow_network=not cached_only)

    recurring_count = sum(1 for row in subset if row["is_recurring_case"])
    repeated_subject_count = sum(1 for row in subset if row["repeated_subject"])
    recurring_fact_count = sum(1 for row in subset if row["recurring_fact"])
    near_duplicate_count = sum(1 for row in subset if row["near_duplicate_question"])
    return {
        "limit": limit,
        "candidate_limit": requested_candidate_limit,
        "recurring_examples": recurring_count,
        "repeated_subject_examples": repeated_subject_count,
        "recurring_fact_examples": recurring_fact_count,
        "near_duplicate_examples": near_duplicate_count,
        "cached_only": int(cached_only),
    }


def build_popqa_benchmark(limit: int, cache_dir: Path, allow_network: bool = False) -> tuple[list[EpisodeInput], list[SupportDoc]]:
    manifest_path = _subset_manifest_path(cache_dir, limit)
    corpus_path = _corpus_cache_path(cache_dir, limit)
    if not manifest_path.exists() or not corpus_path.exists():
        if not allow_network:
            raise FileNotFoundError(
                f"cached PopQA subset not found for limit={limit}. Run the PopQA prefetch command first."
            )
        prefetch_popqa_benchmark(limit=limit, cache_dir=cache_dir)

    manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_rows = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus_docs = [SupportDoc(**row) for row in corpus_rows]

    episodes: list[EpisodeInput] = []
    for index, row in enumerate(manifest_rows):
        metadata = dict(row["metadata"])
        possible_answers = metadata["possible_answers"]
        episodes.append(
            EpisodeInput(
                episode_id=f"popqa-{row['id']}",
                question=row["question"],
                gold_answer=possible_answers[0],
                dataset_id="popqa",
                timestamp=index + 1,
                subject=row["subj"],
                relation=row["prop"],
                domain="popqa",
                support_docs=[],
                parametric_answer=None,
                parametric_confidence=0.0,
                popularity_bin=popularity_bin(int(row["s_pop"])),
                recurrence_hint=float(metadata["recurrence_hint"]),
                stability_hint=0.9,
                volatility_hint=0.05,
                contradiction_hint=0.05,
                freshness=False,
                update_expected=False,
                metadata=metadata,
            )
        )
    return episodes, corpus_docs


class SqliteMemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                value TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                source_ids TEXT NOT NULL,
                support_count INTEGER NOT NULL,
                conflict_state TEXT NOT NULL,
                stability_score REAL NOT NULL,
                revision_history TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def clone(self) -> "SqliteMemoryStore":
        return self

    def close(self) -> None:
        self._connection.close()

    def query(self, subject: str, relation: str):
        row = self._connection.execute(
            """
            SELECT * FROM memory_records
            WHERE subject = ? AND relation = ?
            ORDER BY support_count DESC, stability_score DESC, timestamp DESC
            LIMIT 1
            """,
            (subject, relation),
        ).fetchone()
        if row is None:
            return None
        from .types import MemoryRecord

        return MemoryRecord(
            record_id=f"memory-{row['record_id']}",
            subject=row["subject"],
            relation=row["relation"],
            value=row["value"],
            timestamp=row["timestamp"],
            source_ids=json.loads(row["source_ids"]),
            support_count=row["support_count"],
            conflict_state=row["conflict_state"],
            stability_score=row["stability_score"],
            revision_history=json.loads(row["revision_history"]),
        )

    def write(
        self,
        subject: str,
        relation: str,
        value: str,
        timestamp: int,
        source_ids: list[str],
        support_count: int,
        stability_score: float,
    ):
        existing = self.query(subject, relation)
        conflict_state = "clean"
        revision_history: list[str] = []
        if existing:
            revision_history = list(existing.revision_history) + [existing.value]
            if existing.value != value:
                conflict_state = "contradiction"
        self._connection.execute(
            """
            INSERT INTO memory_records
            (subject, relation, value, timestamp, source_ids, support_count, conflict_state, stability_score, revision_history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject,
                relation,
                value,
                timestamp,
                json.dumps(source_ids),
                support_count,
                conflict_state,
                stability_score,
                json.dumps(revision_history),
            ),
        )
        self._connection.commit()
        return self.query(subject, relation)
