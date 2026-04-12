from __future__ import annotations

from dataclasses import dataclass

from .memory import CorpusRetriever, PersistentMemoryStore
from .patches import PatchBank
from .types import (
    ActionType,
    EpisodeInput,
    MemoryRecord,
    OperationResult,
    PatchRecord,
    RouterFeatures,
    SupportDoc,
)


class FrozenParametricModel:
    def __init__(self) -> None:
        self._knowledge: dict[tuple[str, str], str] = {}

    def clone(self) -> "FrozenParametricModel":
        other = FrozenParametricModel()
        other._knowledge = dict(self._knowledge)
        return other

    def seed(self, subject: str, relation: str, answer: str) -> None:
        self._knowledge[(subject, relation)] = answer

    def answer(self, episode: EpisodeInput) -> tuple[str, float]:
        answer = episode.parametric_answer
        if answer is None:
            answer = self._knowledge.get((episode.subject, episode.relation), "unknown")
        return answer, episode.parametric_confidence


@dataclass(slots=True)
class WorldState:
    model: FrozenParametricModel
    memory: PersistentMemoryStore
    retriever: CorpusRetriever
    patches: PatchBank

    def clone(self) -> "WorldState":
        return WorldState(
            model=self.model.clone(),
            memory=self.memory.clone(),
            retriever=self.retriever.clone(),
            patches=self.patches.clone(),
        )


def best_doc(docs: list[SupportDoc]) -> SupportDoc | None:
    if not docs:
        return None
    return max(docs, key=lambda doc: (doc.trust + doc.relevance, doc.timestamp))


def retrieval_candidates(world: WorldState, episode: EpisodeInput) -> list[SupportDoc]:
    if episode.metadata.get("disable_retrieval", False):
        return []
    if episode.support_docs:
        return list(episode.support_docs)
    return world.retriever.retrieve(episode.question, episode.subject, episode.relation)


def answer_with_patches(world: WorldState, episode: EpisodeInput) -> str | None:
    temporary = world.patches.get_temporary(episode.subject, episode.relation)
    if temporary:
        return temporary.answer
    durable = world.patches.get_durable(episode.subject, episode.relation)
    if durable:
        return durable.answer
    return None


def execute_action(action: ActionType, episode: EpisodeInput, world: WorldState) -> OperationResult:
    if action == ActionType.PARAM_ONLY:
        patched = answer_with_patches(world, episode)
        if patched is not None:
            return OperationResult(
                answer=patched,
                evidence_ids=[],
                touched_memory_ids=[],
                touched_patch_ids=[world.patches.scope_key(episode.subject, episode.relation)],
                latency=0.2,
                side_effects=["durable_patch_read"],
                metrics={"retrieval_calls": 0, "writes": 0, "adapt_steps": 0, "consolidations": 0, "source": 0.0},
            )
        answer, _ = world.model.answer(episode)
        return OperationResult(
            answer=answer,
            evidence_ids=[],
            touched_memory_ids=[],
            touched_patch_ids=[],
            latency=0.2,
            side_effects=[],
            metrics={"retrieval_calls": 0, "writes": 0, "adapt_steps": 0, "consolidations": 0, "source": 0.0},
        )

    if action == ActionType.READ_MEMORY:
        record = world.memory.query(episode.subject, episode.relation)
        answer = record.value if record else "unknown"
        return OperationResult(
            answer=answer,
            evidence_ids=[],
            touched_memory_ids=[record.record_id] if record else [],
            touched_patch_ids=[],
            latency=0.3,
            side_effects=["memory_read"] if record else ["memory_miss"],
            metrics={"retrieval_calls": 0, "writes": 0, "adapt_steps": 0, "consolidations": 0, "source": 1.0 if record else 0.0},
        )

    if action == ActionType.RETRIEVE:
        docs = retrieval_candidates(world, episode)
        doc = best_doc(docs)
        answer = doc.answer if doc else "unknown"
        return OperationResult(
            answer=answer,
            evidence_ids=[doc.doc_id] if doc else [],
            touched_memory_ids=[],
            touched_patch_ids=[],
            latency=0.7,
            side_effects=["retrieved"] if doc else ["retrieval_miss"],
            metrics={"retrieval_calls": 1, "writes": 0, "adapt_steps": 0, "consolidations": 0, "source": 2.0 if doc else 0.0},
        )

    if action == ActionType.WRITE_MEMORY:
        docs = retrieval_candidates(world, episode)
        doc = best_doc(docs)
        if not doc:
            return OperationResult(
                answer="unknown",
                evidence_ids=[],
                touched_memory_ids=[],
                touched_patch_ids=[],
                latency=0.8,
                side_effects=["write_skipped_no_doc"],
                metrics={"retrieval_calls": 1, "writes": 0, "adapt_steps": 0, "consolidations": 0, "source": 0.0},
            )
        record = world.memory.write(
            subject=episode.subject,
            relation=episode.relation,
            value=doc.answer,
            timestamp=episode.timestamp,
            source_ids=[doc.doc_id],
            support_count=max(1, int(round(episode.metadata.get("source_support_count", 1)))),
            stability_score=episode.stability_hint or doc.trust,
        )
        return OperationResult(
            answer=record.value,
            evidence_ids=[doc.doc_id],
            touched_memory_ids=[record.record_id],
            touched_patch_ids=[],
            latency=0.9,
            side_effects=["memory_write", "memory_read"],
            metrics={"retrieval_calls": 1, "writes": 1, "adapt_steps": 0, "consolidations": 0, "source": 1.0},
        )

    if action == ActionType.FAST_ADAPT:
        docs = retrieval_candidates(world, episode)
        doc = best_doc(docs)
        answer = doc.answer if doc else episode.gold_answer
        patch = world.patches.create_temporary_patch(
            subject=episode.subject,
            relation=episode.relation,
            answer=answer,
            creation_trigger=episode.episode_id,
            acceptance_score=episode.recurrence_hint + episode.stability_hint,
        )
        result = OperationResult(
            answer=patch.answer,
            evidence_ids=[doc.doc_id] if doc else [],
            touched_memory_ids=[],
            touched_patch_ids=[patch.patch_id],
            latency=1.0,
            side_effects=["temporary_patch_attached"],
            metrics={"retrieval_calls": 1 if doc else 0, "writes": 0, "adapt_steps": 1, "consolidations": 0, "source": 3.0},
        )
        world.patches.clear_temporary(episode.subject, episode.relation)
        result.side_effects.append("temporary_patch_detached")
        return result

    docs = retrieval_candidates(world, episode)
    doc = best_doc(docs)
    answer = doc.answer if doc else episode.gold_answer
    patch = world.patches.get_temporary(episode.subject, episode.relation)
    if not patch:
        patch = world.patches.create_temporary_patch(
            subject=episode.subject,
            relation=episode.relation,
            answer=answer,
            creation_trigger=f"consolidate::{episode.episode_id}",
            acceptance_score=episode.recurrence_hint + episode.stability_hint,
        )
    promoted = world.patches.promote(episode.subject, episode.relation)
    return OperationResult(
        answer=promoted.answer if promoted else patch.answer,
        evidence_ids=[doc.doc_id] if doc else [],
        touched_memory_ids=[],
        touched_patch_ids=[promoted.patch_id if promoted else patch.patch_id],
        latency=1.2,
        side_effects=["temporary_patch_attached", "patch_promoted"],
        metrics={"retrieval_calls": 1 if doc else 0, "writes": 0, "adapt_steps": 1, "consolidations": 1, "source": 4.0},
    )


def compute_features(
    episode: EpisodeInput,
    world: WorldState,
    recent_actions: list[ActionType],
    episode_index: int,
) -> RouterFeatures:
    _, model_confidence = world.model.answer(episode)
    memory_record = world.memory.query(episode.subject, episode.relation)
    retrieval_docs = retrieval_candidates(world, episode)
    top_doc = best_doc(retrieval_docs)
    source_agreement = float(episode.metadata.get("source_support_count", 1 if top_doc else 0))
    last_update_timestamp = 0.0
    if memory_record:
        last_update_timestamp = float(episode.timestamp - memory_record.timestamp)
    recent_action_repeat = 0.0
    if recent_actions and recent_actions[-1] == ActionType.RETRIEVE:
        recent_action_repeat = 1.0
    domain_change_rate = float(episode.metadata.get("domain_change_rate", episode.volatility_hint))
    forgetting_risk = max(0.0, episode.volatility_hint + 0.2 * episode.update_expected)
    return RouterFeatures(
        model_confidence=model_confidence,
        retrieval_quality_estimate=(top_doc.trust if top_doc else 0.0),
        memory_hit_score=(memory_record.stability_score if memory_record else 0.0),
        recurrence_estimate=episode.recurrence_hint,
        stability_score=episode.stability_hint,
        volatility_score=episode.volatility_hint,
        contradiction_risk=episode.contradiction_hint,
        source_agreement_count=source_agreement,
        time_since_last_update=last_update_timestamp,
        domain_change_rate=domain_change_rate,
        popularity_bin=episode.popularity_bin,
        recent_action_repeat=recent_action_repeat,
        forgetting_risk=forgetting_risk,
    )


def action_mask(features: RouterFeatures, episode: EpisodeInput, world: WorldState) -> dict[ActionType, bool]:
    memory_record = world.memory.query(episode.subject, episode.relation)
    can_read_memory = memory_record is not None and features.memory_hit_score >= 0.5
    can_write_memory = features.retrieval_quality_estimate >= 0.4 and features.source_agreement_count >= 1
    can_fast_adapt = episode.update_expected or features.recurrence_estimate >= 0.5
    can_consolidate = (
        features.source_agreement_count >= 2
        and features.recurrence_estimate >= 0.6
        and features.stability_score >= 0.7
        and features.volatility_score <= 0.3
        and features.contradiction_risk <= 0.3
        and not world.patches.recently_rolled_back(episode.subject, episode.relation)
    )
    return {
        ActionType.PARAM_ONLY: True,
        ActionType.READ_MEMORY: can_read_memory,
        ActionType.RETRIEVE: features.retrieval_quality_estimate > 0.0 or bool(episode.support_docs),
        ActionType.WRITE_MEMORY: can_write_memory,
        ActionType.FAST_ADAPT: can_fast_adapt,
        ActionType.CONSOLIDATE: can_consolidate,
    }


def evaluate_quality(answer: str, gold_answer: str) -> float:
    return 1.0 if answer.strip().lower() == gold_answer.strip().lower() else 0.0


def forgetting_penalty(action: ActionType, features: RouterFeatures) -> float:
    if action == ActionType.PARAM_ONLY:
        return 0.0
    if action == ActionType.READ_MEMORY:
        return 0.05 * features.contradiction_risk
    if action == ActionType.RETRIEVE:
        return 0.1 * features.contradiction_risk
    if action == ActionType.WRITE_MEMORY:
        return 0.2 * (features.contradiction_risk + features.volatility_score)
    if action == ActionType.FAST_ADAPT:
        return 0.35 * (features.forgetting_risk + features.volatility_score)
    return 0.45 * (features.forgetting_risk + features.volatility_score)


def observed_future_utility(action: ActionType, current_index: int, episodes: list[EpisodeInput]) -> float:
    current = episodes[current_index]
    future = episodes[current_index + 1 : current_index + 4]
    matches = 0
    updates = 0
    for item in future:
        if item.subject == current.subject or item.relation == current.relation:
            matches += 1
        if item.domain == current.domain:
            updates += int(item.update_expected)
    scale = 0.15 * matches + 0.1 * updates
    if action in {ActionType.WRITE_MEMORY, ActionType.FAST_ADAPT, ActionType.CONSOLIDATE}:
        return scale + 0.2 * current.recurrence_hint
    if action == ActionType.READ_MEMORY:
        return 0.1 * matches
    if action == ActionType.RETRIEVE:
        return 0.05 * matches
    return 0.0
