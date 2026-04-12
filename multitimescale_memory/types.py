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

    def to_vector(self) -> list[float]:
        return [
            1.0,
            self.model_confidence,
            self.retrieval_quality_estimate,
            self.memory_hit_score,
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
    metrics: dict[str, float]


@dataclass(slots=True)
class RunTrace:
    episode_id: str
    dataset_id: str
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
    metrics: dict[str, float]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
