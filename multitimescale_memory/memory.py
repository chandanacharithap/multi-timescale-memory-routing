from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .types import MemoryRecord, SupportDoc


class PersistentMemoryStore:
    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._key_to_ids: dict[tuple[str, str], list[str]] = {}
        self._counter = 0

    def clone(self) -> "PersistentMemoryStore":
        other = PersistentMemoryStore()
        other._records = {key: replace(value) for key, value in self._records.items()}
        other._key_to_ids = {key: list(value) for key, value in self._key_to_ids.items()}
        other._counter = self._counter
        return other

    def query(self, subject: str, relation: str) -> MemoryRecord | None:
        record_ids = self._key_to_ids.get((subject, relation), [])
        records = [self._records[record_id] for record_id in record_ids]
        if not records:
            return None
        return max(records, key=lambda record: (record.support_count, record.stability_score, record.timestamp))

    def write(
        self,
        subject: str,
        relation: str,
        value: str,
        timestamp: int,
        source_ids: list[str],
        support_count: int,
        stability_score: float,
    ) -> MemoryRecord:
        self._counter += 1
        record_id = f"memory-{self._counter}"
        existing = self.query(subject, relation)
        conflict_state = "clean"
        revision_history: list[str] = []
        if existing:
            revision_history = list(existing.revision_history) + [existing.value]
            if existing.value != value:
                conflict_state = "contradiction"
        record = MemoryRecord(
            record_id=record_id,
            subject=subject,
            relation=relation,
            value=value,
            timestamp=timestamp,
            source_ids=source_ids,
            support_count=support_count,
            conflict_state=conflict_state,
            stability_score=stability_score,
            revision_history=revision_history,
        )
        self._records[record_id] = record
        self._key_to_ids.setdefault((subject, relation), []).append(record_id)
        return record

    def all_records(self) -> Iterable[MemoryRecord]:
        return self._records.values()


class CorpusRetriever:
    def __init__(self) -> None:
        self._docs: list[SupportDoc] = []

    def clone(self) -> "CorpusRetriever":
        other = CorpusRetriever()
        other._docs = list(self._docs)
        return other

    def ingest(self, docs: list[SupportDoc]) -> None:
        self._docs.extend(docs)

    def retrieve(self, query: str, subject: str, relation: str, limit: int = 3) -> list[SupportDoc]:
        query_terms = set(query.lower().split()) | {subject.lower(), relation.lower()}

        def score(doc: SupportDoc) -> tuple[float, float]:
            overlap = len(query_terms.intersection(doc.text.lower().split()))
            return (overlap + doc.relevance + doc.trust, doc.timestamp)

        return sorted(self._docs, key=score, reverse=True)[:limit]
