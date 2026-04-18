from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    PARAM_ONLY = "param_only"
    READ_MEMORY = "read_memory"
    RETRIEVE = "retrieve"
    WRITE_MEMORY = "write_memory"
    FAST_ADAPT = "fast_adapt"
    CONSOLIDATE = "consolidate"


@dataclass(slots=True)
class SupportDoc:
    doc_id: str
    text: str
    answer: str
    source: str
    timestamp: int
    trust: float = 1.0
    relevance: float = 1.0


@dataclass(slots=True)
class EpisodeInput:
    episode_id: str
    question: str
    gold_answer: str
    dataset_id: str
    timestamp: int
    subject: str
    relation: str
    domain: str
    support_docs: list[SupportDoc] = field(default_factory=list)
    parametric_answer: str | None = None
    parametric_confidence: float = 0.0
    popularity_bin: float = 0.0
    recurrence_hint: float = 0.0
    stability_hint: float = 0.0
    volatility_hint: float = 0.0
    contradiction_hint: float = 0.0
    freshness: bool = False
    update_expected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouterFeatures:
    model_confidence: float
    retrieval_quality_estimate: float
    memory_hit_score: float
    memory_alignment_score: float
    recurrence_estimate: float
    stability_score: float
    volatility_score: float
    contradiction_risk: float
    source_agreement_count: float
    time_since_last_update: float
    domain_change_rate: float
    popularity_bin: float
    recent_action_repeat: float
    forgetting_risk: float
    question_seen_before: float = 0.0
    answer_changed_since_last_seen: float = 0.0
    weeks_since_last_change: float = 0.0
    has_aliases: float = 0.0
    prior_stale_answer_available: float = 0.0

    def to_vector(self) -> list[float]:
        return [
            1.0,
            self.model_confidence,
            self.retrieval_quality_estimate,
            self.memory_hit_score,
            self.memory_alignment_score,
            self.recurrence_estimate,
            self.stability_score,
            self.volatility_score,
            self.contradiction_risk,
            self.source_agreement_count,
            self.time_since_last_update,
            self.domain_change_rate,
            self.popularity_bin,
            self.recent_action_repeat,
            self.forgetting_risk,
            self.question_seen_before,
            self.answer_changed_since_last_seen,
            self.weeks_since_last_change,
            self.has_aliases,
            self.prior_stale_answer_available,
        ]


@dataclass(slots=True)
class RewardBreakdown:
    quality: float
    cost: float
    forgetting: float
    future_value: float
    total: float


@dataclass(slots=True)
class RouterDecision:
    action: ActionType
    action_scores: dict[str, float]
    action_mask: dict[str, bool]
    immediate_reward_estimate: float
    future_value_estimate: float
    rationale: str


@dataclass(slots=True)
class MemoryRecord:
    record_id: str
    subject: str
    relation: str
    value: str
    timestamp: int
    source_ids: list[str]
    support_count: int
    conflict_state: str
    stability_score: float
    revision_history: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PatchRecord:
    patch_id: str
    scope_key: str
    answer: str
    creation_trigger: str
    acceptance_score: float
    durability_status: str
    rollback_metadata: dict[str, Any]
    activation_policy: dict[str, Any]
    subject: str
    relation: str
    temporary: bool = True


@dataclass(slots=True)
class OperationResult:
    answer: str
    evidence_ids: list[str]
    touched_memory_ids: list[str]
    touched_patch_ids: list[str]
    latency: float
    side_effects: list[str]
    metrics: dict[str, Any]


@dataclass(slots=True)
class RunTrace:
    episode_id: str
    dataset_id: str
    question: str
    subject: str
    relation: str
    gold_answer: str
    action: str
    action_mask: dict[str, bool]
    action_scores: dict[str, float]
    features: dict[str, float]
    answer: str
    reward: dict[str, float]
    evidence_ids: list[str]
    touched_memory_ids: list[str]
    touched_patch_ids: list[str]
    side_effects: list[str]
    metrics: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResultRow:
    mode: str
    answer_quality: float
    recurring_quality: float
    non_recurring_quality: float
    latency: float
    retrieval_calls: int
    memory_reads: int
    memory_writes: int
    adaptation_count: int
    recurring_retrieval_calls: float
    recurring_memory_reads: float
    recurring_memory_writes: float
    action_distribution: dict[str, int]
    extra_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class MetricSummary:
    mean: float
    std: float
    min: float
    max: float
    ci95_low: float
    ci95_high: float
    count: int


@dataclass(slots=True)
class JournalRunRow:
    benchmark: str
    model_name: str
    seed: int
    mode: str
    answer_quality: float
    recurring_quality: float
    non_recurring_quality: float
    latency: float
    retrieval_calls: float
    memory_reads: float
    memory_writes: float
    adaptation_count: float
    stale_answer_rate: float
    consolidation_count: float
    rollback_count: float
    forgetting_delta: float
    average_reward: float
    action_distribution: dict[str, int] = field(default_factory=dict)
    extra_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class JournalAggregateRow:
    benchmark: str
    model_name: str
    mode: str
    answer_quality: MetricSummary
    recurring_quality: MetricSummary
    non_recurring_quality: MetricSummary
    latency: MetricSummary
    retrieval_calls: MetricSummary
    memory_reads: MetricSummary
    memory_writes: MetricSummary
    adaptation_count: MetricSummary
    stale_answer_rate: MetricSummary
    consolidation_count: MetricSummary
    rollback_count: MetricSummary
    forgetting_delta: MetricSummary
    average_reward: MetricSummary
