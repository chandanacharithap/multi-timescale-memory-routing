from __future__ import annotations

import math
import re
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

    def retrieve(self, query: str, subject: str, relation: str, limit: int = 3, max_timestamp: int | None = None) -> list[SupportDoc]:
        query_terms = set(query.lower().split()) | {subject.lower(), relation.lower()}
        docs = [doc for doc in self._docs if max_timestamp is None or doc.timestamp <= max_timestamp]

        def score(doc: SupportDoc) -> tuple[float, float]:
            overlap = len(query_terms.intersection(doc.text.lower().split()))
            return (overlap + doc.relevance + doc.trust, doc.timestamp)

        return sorted(docs, key=score, reverse=True)[:limit]


class BM25Retriever:
    def __init__(self, docs: list[SupportDoc] | None = None) -> None:
        self._docs: list[SupportDoc] = docs or []
        self._tokenized: list[list[str]] = []
        self._doc_freqs: dict[str, int] = {}
        self._avg_doc_length = 0.0
        self._rebuild()

    def clone(self) -> "BM25Retriever":
        return BM25Retriever(list(self._docs))

    def ingest(self, docs: list[SupportDoc]) -> None:
        self._docs.extend(docs)
        self._rebuild()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _rebuild(self) -> None:
        self._tokenized = [self._tokenize(doc.text) for doc in self._docs]
        self._doc_freqs = {}
        total_length = 0
        for tokens in self._tokenized:
            total_length += len(tokens)
            for token in set(tokens):
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
        self._avg_doc_length = total_length / len(self._tokenized) if self._tokenized else 0.0

    def retrieve(self, query: str, subject: str, relation: str, limit: int = 3, max_timestamp: int | None = None) -> list[SupportDoc]:
        docs = [doc for doc in self._docs if max_timestamp is None or doc.timestamp <= max_timestamp]
        if not docs:
            return []
        query_tokens = self._tokenize(query) + self._tokenize(subject) + self._tokenize(relation)
        unique_tokens = set(query_tokens)
        tokenized_docs = [self._tokenize(doc.text) for doc in docs]
        doc_freqs: dict[str, int] = {}
        total_length = 0
        for tokens in tokenized_docs:
            total_length += len(tokens)
            for token in set(tokens):
                doc_freqs[token] = doc_freqs.get(token, 0) + 1
        avg_doc_length = total_length / len(tokenized_docs) if tokenized_docs else 0.0
        n_docs = len(docs)
        k1 = 1.5
        b = 0.75
        scored: list[tuple[float, SupportDoc]] = []
        for doc, tokens in zip(docs, tokenized_docs):
            if not tokens:
                scored.append((0.0, doc))
                continue
            doc_len = len(tokens)
            score = 0.0
            for token in unique_tokens:
                tf = tokens.count(token)
                if tf == 0:
                    continue
                df = doc_freqs.get(token, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * doc_len / max(avg_doc_length, 1.0))
                score += idf * (tf * (k1 + 1)) / denom
            score += doc.trust
            scored.append((score, doc))
        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
        return [
            SupportDoc(
                doc_id=doc.doc_id,
                text=doc.text,
                answer=doc.answer,
                source=doc.source,
                timestamp=doc.timestamp,
                trust=doc.trust,
                relevance=score,
            )
            for score, doc in ranked
        ]
