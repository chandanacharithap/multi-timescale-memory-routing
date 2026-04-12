from __future__ import annotations

from dataclasses import dataclass

from .types import RewardBreakdown


@dataclass(slots=True)
class RewardWeights:
    latency: float = 0.4
    retrieval: float = 0.8
    write: float = 0.7
    adapt: float = 0.8
    consolidate: float = 1.1
    forgetting: float = 1.0
    beta_future: float = 0.9


def compute_reward(
    quality: float,
    latency: float,
    retrieval_calls: int,
    writes: int,
    adapt_steps: int,
    consolidations: int,
    forgetting_risk: float,
    future_value: float,
    weights: RewardWeights,
) -> RewardBreakdown:
    cost = (
        weights.latency * latency
        + weights.retrieval * retrieval_calls
        + weights.write * writes
        + weights.adapt * adapt_steps
        + weights.consolidate * consolidations
    )
    forgetting = weights.forgetting * forgetting_risk
    total = quality - cost - forgetting + weights.beta_future * future_value
    return RewardBreakdown(
        quality=quality,
        cost=cost,
        forgetting=forgetting,
        future_value=weights.beta_future * future_value,
        total=total,
    )
