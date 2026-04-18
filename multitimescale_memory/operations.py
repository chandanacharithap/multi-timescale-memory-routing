from __future__ import annotations

from dataclasses import dataclass

from .memory import CorpusRetriever, PersistentMemoryStore
from .modeling import AnswerResult, FrozenParametricModel
from .patches import PatchBank
from .types import (
    ActionType,
    EpisodeInput,
    OperationResult,
    RouterFeatures,
    SupportDoc,
)


def is_freshness_episode(episode: EpisodeInput) -> bool:
    return episode.dataset_id in {"freshness", "freshqa_public", "mquake", "knowedit", "uniedit"}


@dataclass(slots=True)
class WorldState:
    model: object
    memory: object
    retriever: object
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


def retrieval_candidates(world: WorldState, episode: EpisodeInput, limit: int = 3) -> list[SupportDoc]:
    if episode.metadata.get("disable_retrieval", False):
        return []
    if episode.support_docs:
        docs = list(episode.support_docs)
        if is_freshness_episode(episode):
            docs = [doc for doc in docs if doc.timestamp <= episode.timestamp]
        return docs
    max_timestamp = episode.timestamp if is_freshness_episode(episode) else None
    return world.retriever.retrieve(episode.question, episode.subject, episode.relation, limit=limit, max_timestamp=max_timestamp)


def memory_alignment_score(memory_record, docs: list[SupportDoc]) -> float:
    if memory_record is None:
        return 0.0
    if not docs:
        return 0.5
    overlap = len(set(memory_record.source_ids).intersection(doc.doc_id for doc in docs))
    if overlap:
        return 1.0
    best = best_doc(docs)
    if best and memory_record.value and memory_record.value.lower() in best.text.lower():
        return 0.75
    return 0.0


def answer_with_patches(world: WorldState, episode: EpisodeInput) -> str | None:
    temporary = world.patches.get_temporary(episode.subject, episode.relation, episode.timestamp)
    if temporary:
        return temporary.answer
    durable = world.patches.get_durable(episode.subject, episode.relation)
    if durable:
        return durable.answer
    return None


def freshness_doc_answer(docs: list[SupportDoc], strategy: str) -> AnswerResult:
    if not docs:
        return AnswerResult(answer="unknown", confidence=0.0, latency=0.2)
    if strategy == "retrieve":
        chosen = max(docs, key=lambda doc: (doc.trust + doc.relevance, -doc.timestamp))
        return AnswerResult(answer=chosen.answer, confidence=min(0.95, chosen.trust), latency=0.32)
    chosen = max(docs, key=lambda doc: (doc.timestamp, doc.trust, doc.relevance))
    return AnswerResult(answer=chosen.answer, confidence=min(0.99, 0.1 + chosen.trust), latency=0.42)


def grounded_answer(world: WorldState, episode: EpisodeInput, docs: list[SupportDoc], strategy: str) -> AnswerResult:
    if is_freshness_episode(episode):
        return freshness_doc_answer(docs, strategy)
    return world.model.answer_with_evidence(episode, docs)


def maybe_trigger_rollback(world: WorldState, episode: EpisodeInput, docs: list[SupportDoc]) -> bool:
    if not episode.metadata.get("rollback_probe", False):
        return False
    durable = world.patches.get_durable(episode.subject, episode.relation)
    if durable is None or not docs:
        docs = retrieval_candidates(world, episode, limit=5)
        if durable is None or not docs:
            return False
    rollback_docs = retrieval_candidates(world, episode, limit=5)
    corrected = freshness_doc_answer(rollback_docs or docs, strategy="adapt")
    if corrected.answer != durable.answer:
        return world.patches.rollback(episode.subject, episode.relation)
    return False


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
                metrics={"retrieval_calls": 0, "memory_reads": 0, "writes": 0, "adapt_steps": 0, "consolidations": 0, "source": 0.0},
            )
        parametric = world.model.answer_parametric(episode)
        return OperationResult(
            answer=parametric.answer,
            evidence_ids=[],
            touched_memory_ids=[],
            touched_patch_ids=[],
            latency=parametric.latency,
            side_effects=[],
            metrics={"retrieval_calls": 0, "memory_reads": 0, "writes": 0, "adapt_steps": 0, "consolidations": 0, "source": 0.0},
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
            metrics={"retrieval_calls": 0, "memory_reads": 1 if record else 0, "writes": 0, "adapt_steps": 0, "consolidations": 0, "source": 1.0 if record else 0.0},
        )

    if action == ActionType.RETRIEVE:
        docs = retrieval_candidates(world, episode)
        rollback_triggered = maybe_trigger_rollback(world, episode, docs)
        grounded = grounded_answer(world, episode, docs, strategy="retrieve")
        side_effects = ["retrieved"] if docs else ["retrieval_miss"]
        if rollback_triggered:
            side_effects.append("rollback_triggered")
        return OperationResult(
            answer=grounded.answer,
            evidence_ids=[doc.doc_id for doc in docs],
            touched_memory_ids=[],
            touched_patch_ids=[],
            latency=0.3 + grounded.latency,
            side_effects=side_effects,
            metrics={"retrieval_calls": 1 if docs else 0, "memory_reads": 0, "writes": 0, "adapt_steps": 0, "consolidations": 0, "source": 2.0 if docs else 0.0},
        )

    if action == ActionType.WRITE_MEMORY:
        docs = retrieval_candidates(world, episode)
        rollback_triggered = maybe_trigger_rollback(world, episode, docs)
        grounded = grounded_answer(world, episode, docs, strategy="adapt" if is_freshness_episode(episode) else "retrieve")
        if not docs:
            return OperationResult(
                answer="unknown",
                evidence_ids=[],
                touched_memory_ids=[],
                touched_patch_ids=[],
                latency=0.8,
                side_effects=["write_skipped_no_doc"],
                metrics={"retrieval_calls": 0, "memory_reads": 0, "writes": 0, "adapt_steps": 0, "consolidations": 0, "source": 0.0},
            )
        record = world.memory.write(
            subject=episode.subject,
            relation=episode.relation,
            value=grounded.answer,
            timestamp=episode.timestamp,
            source_ids=[doc.doc_id for doc in docs],
            support_count=max(1, int(round(episode.metadata.get("source_support_count", 1)))),
            stability_score=episode.stability_hint or max(doc.trust for doc in docs),
        )
        side_effects = ["memory_write", "memory_read"]
        if record.conflict_state == "contradiction":
            side_effects.append("conflict_detected")
        if rollback_triggered:
            side_effects.append("rollback_triggered")
        return OperationResult(
            answer=record.value,
            evidence_ids=[doc.doc_id for doc in docs],
            touched_memory_ids=[record.record_id],
            touched_patch_ids=[],
            latency=0.4 + grounded.latency,
            side_effects=side_effects,
            metrics={"retrieval_calls": 1 if docs else 0, "memory_reads": 1, "writes": 1, "adapt_steps": 0, "consolidations": 0, "source": 1.0},
        )

    if action == ActionType.FAST_ADAPT:
        docs = retrieval_candidates(world, episode, limit=5)
        rollback_triggered = maybe_trigger_rollback(world, episode, docs)
        grounded = grounded_answer(world, episode, docs, strategy="adapt") if docs else world.model.answer_parametric(episode)
        temporary_horizon = int(episode.metadata.get("temporary_patch_horizon", 0))
        patch = world.patches.create_temporary_patch(
            subject=episode.subject,
            relation=episode.relation,
            answer=grounded.answer,
            creation_trigger=episode.episode_id,
            acceptance_score=episode.recurrence_hint + episode.stability_hint,
            activation_policy={
                "scope": world.patches.scope_key(episode.subject, episode.relation),
                "expires_at": episode.timestamp + temporary_horizon,
            },
        )
        side_effects = ["temporary_patch_attached"]
        if rollback_triggered:
            side_effects.append("rollback_triggered")
        result = OperationResult(
            answer=patch.answer,
            evidence_ids=[doc.doc_id for doc in docs],
            touched_memory_ids=[],
            touched_patch_ids=[patch.patch_id],
            latency=0.5 + grounded.latency,
            side_effects=side_effects,
            metrics={"retrieval_calls": 1 if docs else 0, "memory_reads": 0, "writes": 0, "adapt_steps": 1, "consolidations": 0, "source": 3.0},
        )
        if temporary_horizon <= 0:
            world.patches.clear_temporary(episode.subject, episode.relation)
            result.side_effects.append("temporary_patch_detached")
        return result

    docs = retrieval_candidates(world, episode, limit=5)
    rollback_triggered = maybe_trigger_rollback(world, episode, docs)
    grounded = grounded_answer(world, episode, docs, strategy="adapt") if docs else world.model.answer_parametric(episode)
    patch = world.patches.get_temporary(episode.subject, episode.relation, episode.timestamp)
    if not patch:
        patch = world.patches.create_temporary_patch(
            subject=episode.subject,
            relation=episode.relation,
            answer=grounded.answer,
            creation_trigger=f"consolidate::{episode.episode_id}",
            acceptance_score=episode.recurrence_hint + episode.stability_hint,
        )
    promoted = world.patches.promote(episode.subject, episode.relation)
    side_effects = ["temporary_patch_attached", "patch_promoted"]
    if rollback_triggered:
        side_effects.append("rollback_triggered")
    return OperationResult(
        answer=promoted.answer if promoted else patch.answer,
        evidence_ids=[doc.doc_id for doc in docs],
        touched_memory_ids=[],
        touched_patch_ids=[promoted.patch_id if promoted else patch.patch_id],
        latency=0.7 + grounded.latency,
        side_effects=side_effects,
        metrics={"retrieval_calls": 1 if docs else 0, "memory_reads": 0, "writes": 0, "adapt_steps": 1, "consolidations": 1, "source": 4.0},
    )


def compute_features(
    episode: EpisodeInput,
    world: WorldState,
    recent_actions: list[ActionType],
    episode_index: int,
) -> RouterFeatures:
    model_confidence = world.model.answer_parametric(episode).confidence
    memory_record = world.memory.query(episode.subject, episode.relation)
    retrieval_docs = retrieval_candidates(world, episode, limit=5 if is_freshness_episode(episode) else 3)
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
    question_seen_before = float(bool(episode.metadata.get("question_seen_before", False)))
    answer_changed_since_last_seen = float(bool(episode.metadata.get("answer_changed_since_last_seen", False)))
    weeks_since_last_change = float(episode.metadata.get("weeks_since_last_change", 0.0))
    has_aliases = float(bool(episode.metadata.get("has_aliases", False)))
    prior_stale_answer_available = float(bool(episode.metadata.get("prior_stale_answer_available", False)))
    return RouterFeatures(
        model_confidence=model_confidence,
        retrieval_quality_estimate=(top_doc.trust if top_doc else 0.0),
        memory_hit_score=(memory_record.stability_score if memory_record else 0.0),
        memory_alignment_score=memory_alignment_score(memory_record, retrieval_docs),
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
        question_seen_before=question_seen_before,
        answer_changed_since_last_seen=answer_changed_since_last_seen,
        weeks_since_last_change=weeks_since_last_change,
        has_aliases=has_aliases,
        prior_stale_answer_available=prior_stale_answer_available,
    )


def action_mask(features: RouterFeatures, episode: EpisodeInput, world: WorldState) -> dict[ActionType, bool]:
    memory_record = world.memory.query(episode.subject, episode.relation)
    can_retrieve = features.retrieval_quality_estimate > 0.0 or bool(episode.support_docs)
    can_read_memory = memory_record is not None and features.memory_hit_score >= 0.5
    popqa_mode = episode.dataset_id == "popqa"
    public_freshqa_mode = episode.dataset_id == "freshqa_public"
    popqa_guardrail_mode = episode.metadata.get("popqa_guardrail_mode", "strict")
    recurrence_gate = features.recurrence_estimate >= (0.65 if popqa_mode else 0.5)
    quality_gate = features.retrieval_quality_estimate >= (0.65 if popqa_mode else 0.4) and features.source_agreement_count >= 1
    can_write_memory = quality_gate and recurrence_gate and memory_record is None
    can_param_only = True

    if popqa_mode and popqa_guardrail_mode == "strict":
        weak_recurrence = features.recurrence_estimate < 0.78
        low_model_confidence = features.model_confidence < 0.45
        weak_memory_alignment = features.memory_alignment_score < 0.8
        can_read_memory = (
            memory_record is not None
            and features.memory_hit_score >= 0.6
            and features.recurrence_estimate >= 0.72
            and features.memory_alignment_score >= 0.8
        )
        can_write_memory = (
            memory_record is None
            and (
                (
                    features.recurrence_estimate >= 0.82
                    and features.retrieval_quality_estimate >= 0.7
                )
                or (
                    features.recurrence_estimate >= 0.72
                    and features.retrieval_quality_estimate >= 0.85
                    and low_model_confidence
                )
            )
        )
        if can_retrieve and (
            (low_model_confidence and weak_recurrence)
            or (memory_record is not None and weak_recurrence and weak_memory_alignment)
            or (features.recurrence_estimate < 0.6 and features.retrieval_quality_estimate >= 0.75)
        ):
            can_param_only = False
            if weak_recurrence and weak_memory_alignment:
                can_read_memory = False
            if features.recurrence_estimate < 0.72:
                can_write_memory = False
    elif popqa_mode:
        can_read_memory = memory_record is not None and features.memory_hit_score >= 0.5
        can_write_memory = quality_gate and recurrence_gate and memory_record is None

    if is_freshness_episode(episode):
        update_type = episode.metadata.get("update_type")
        volatile_update = update_type == "volatile_update" or features.volatility_score >= 0.7
        corrective_update = update_type == "rollback_probe" or features.contradiction_risk >= 0.45
        stable_confirmation = update_type == "confirmation" and features.stability_score >= 0.85 and features.volatility_score <= 0.2
        if world.patches.recently_rolled_back(episode.subject, episode.relation):
            can_read_memory = False
        can_read_memory = can_read_memory and features.volatility_score <= 0.45
        if volatile_update or corrective_update:
            can_write_memory = False
        if can_read_memory and (volatile_update or corrective_update):
            can_read_memory = False
        if stable_confirmation and memory_record is not None:
            can_read_memory = True
        if episode.metadata.get("consolidation_candidate", False) and world.patches.get_durable(episode.subject, episode.relation) is None:
            can_read_memory = False
        if update_type == "stable_update" and features.recurrence_estimate >= 0.78 and features.stability_score >= 0.8:
            can_write_memory = can_write_memory or (memory_record is None and features.retrieval_quality_estimate >= 0.7)

    can_fast_adapt = (episode.update_expected or features.recurrence_estimate >= 0.5) and not episode.metadata.get("disable_fast_adapt", False)
    if is_freshness_episode(episode):
        update_type = episode.metadata.get("update_type")
        can_fast_adapt = can_fast_adapt and (
            update_type in {"volatile_update", "rollback_probe", "stable_update"}
            or features.domain_change_rate >= 0.3
        )
    if public_freshqa_mode:
        question_seen_before = bool(episode.metadata.get("question_seen_before", False))
        answer_changed_since_last_seen = bool(episode.metadata.get("answer_changed_since_last_seen", False))
        prior_stale_answer_available = bool(episode.metadata.get("prior_stale_answer_available", False))
        confirmation_count_before = int(episode.metadata.get("confirmation_count_before", 0))
        low_change_risk = (
            question_seen_before
            and not answer_changed_since_last_seen
            and not prior_stale_answer_available
            and features.volatility_score <= 0.45
            and features.contradiction_risk <= 0.25
        )
        patch_visible = (
            answer_with_patches(world, episode) is not None
            or world.patches.get_durable(episode.subject, episode.relation) is not None
        )
        can_read_memory = (
            (memory_record is not None or patch_visible)
            and question_seen_before
            and confirmation_count_before >= 1
            and features.volatility_score <= 0.5
            and (memory_record is None or features.memory_alignment_score >= 0.55)
        )
        can_write_memory = (
            can_write_memory
            and (
                answer_changed_since_last_seen
                or prior_stale_answer_available
                or (
                    not question_seen_before
                    and features.stability_score >= 0.8
                    and features.retrieval_quality_estimate >= 0.85
                )
            )
        )
        can_fast_adapt = can_fast_adapt and (
            answer_changed_since_last_seen
            or prior_stale_answer_available
            or episode.metadata.get("update_type") == "rollback_probe"
            or features.contradiction_risk >= 0.4
        )
        if low_change_risk and can_retrieve:
            can_write_memory = False
            can_fast_adapt = False
            can_param_only = False
        if not question_seen_before and can_retrieve and features.model_confidence < 0.55:
            can_param_only = False
    can_consolidate = (
        features.source_agreement_count >= 2
        and features.recurrence_estimate >= 0.6
        and features.stability_score >= 0.7
        and features.volatility_score <= 0.3
        and features.contradiction_risk <= 0.3
        and not world.patches.recently_rolled_back(episode.subject, episode.relation)
        and not episode.metadata.get("disable_consolidate", False)
    )
    if public_freshqa_mode:
        question_seen_before = bool(episode.metadata.get("question_seen_before", False))
        answer_changed_since_last_seen = bool(episode.metadata.get("answer_changed_since_last_seen", False))
        confirmation_count_before = int(episode.metadata.get("confirmation_count_before", 0))
        change_count_before = int(episode.metadata.get("change_count_before", 0))
        stable_confirmation = (
            episode.metadata.get("update_type") == "confirmation"
            and features.stability_score >= 0.75
            and features.volatility_score <= 0.35
        )
        can_consolidate = (
            can_consolidate
            and question_seen_before
            and not answer_changed_since_last_seen
            and change_count_before >= 1
            and confirmation_count_before >= 2
            and stable_confirmation
        )
    elif is_freshness_episode(episode):
        update_type = episode.metadata.get("update_type")
        can_consolidate = can_consolidate and (
            (
                update_type == "confirmation"
                and episode.metadata.get("stability_class") == "stable"
                and episode.metadata.get("volatility_class") == "low"
            )
            or (
                update_type is None
                and features.stability_score >= 0.85
                and features.volatility_score <= 0.15
                and features.contradiction_risk <= 0.1
            )
        )
    return {
        ActionType.PARAM_ONLY: can_param_only,
        ActionType.READ_MEMORY: can_read_memory,
        ActionType.RETRIEVE: can_retrieve,
        ActionType.WRITE_MEMORY: can_write_memory,
        ActionType.FAST_ADAPT: can_fast_adapt,
        ActionType.CONSOLIDATE: can_consolidate,
    }


def evaluate_quality(answer: str, gold_answer: str) -> float:
    normalized = answer.strip().lower()
    if normalized == gold_answer.strip().lower():
        return 1.0
    return 0.0


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


def calibrated_future_value_scale(features: RouterFeatures, episode: EpisodeInput) -> float:
    if episode.dataset_id != "freshqa_public":
        return 1.0
    question_seen_before = bool(episode.metadata.get("question_seen_before", False))
    answer_changed_since_last_seen = bool(episode.metadata.get("answer_changed_since_last_seen", False))
    prior_stale_answer_available = bool(episode.metadata.get("prior_stale_answer_available", False))
    confirmation_count_before = int(episode.metadata.get("confirmation_count_before", 0))
    change_count_before = int(episode.metadata.get("change_count_before", 0))
    update_type = str(episode.metadata.get("update_type", "none"))
    stability_class = str(episode.metadata.get("stability_class", "mixed"))
    volatility_class = str(episode.metadata.get("volatility_class", "medium"))

    if not question_seen_before:
        return 0.0

    scale = 1.0

    if answer_changed_since_last_seen or prior_stale_answer_available:
        scale *= 0.05
    if update_type in {"volatile_update", "rollback_probe"}:
        scale *= 0.05
    elif update_type == "stable_update":
        scale *= 0.3

    if features.recurrence_estimate < 0.55:
        scale *= 0.2
    elif features.recurrence_estimate < 0.7:
        scale *= 0.45

    if features.volatility_score >= 0.65 or volatility_class == "high":
        scale *= 0.15
    elif features.volatility_score >= 0.45 or volatility_class == "medium":
        scale *= 0.5

    if features.contradiction_risk >= 0.55:
        scale *= 0.1
    elif features.contradiction_risk >= 0.35:
        scale *= 0.4

    if features.retrieval_quality_estimate >= 0.8 and confirmation_count_before < 2:
        scale *= 0.35
    if features.has_aliases >= 1.0:
        scale *= 0.8
    if features.prior_stale_answer_available >= 1.0:
        scale *= 0.5

    stable_confirmation = (
        update_type == "confirmation"
        and stability_class == "stable"
        and volatility_class == "low"
        and features.stability_score >= 0.75
        and features.volatility_score <= 0.3
        and confirmation_count_before >= 2
        and change_count_before >= 1
    )
    forgetting_probe = update_type == "forgetting_probe" and confirmation_count_before >= 1

    if stable_confirmation:
        scale = max(scale, 0.9)
    elif forgetting_probe and features.contradiction_risk <= 0.2:
        scale = max(scale, 0.55)

    return max(0.0, min(scale, 1.0))


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
    if current.dataset_id == "freshqa_public":
        question_seen_before = bool(current.metadata.get("question_seen_before", False))
        answer_changed_since_last_seen = bool(current.metadata.get("answer_changed_since_last_seen", False))
        prior_stale_answer_available = bool(current.metadata.get("prior_stale_answer_available", False))
        confirmation_count_before = int(current.metadata.get("confirmation_count_before", 0))
        low_change_risk = question_seen_before and not answer_changed_since_last_seen and not prior_stale_answer_available
        if action == ActionType.RETRIEVE:
            return (0.08 if low_change_risk else 0.03) + 0.03 * matches
        if action == ActionType.READ_MEMORY:
            if confirmation_count_before < 1:
                return 0.0
            return 0.04 * matches
        if action in {ActionType.WRITE_MEMORY, ActionType.FAST_ADAPT, ActionType.CONSOLIDATE}:
            if not question_seen_before or not (answer_changed_since_last_seen or prior_stale_answer_available):
                return 0.0
            if action == ActionType.CONSOLIDATE and confirmation_count_before < 2:
                return 0.0
            base = {
                ActionType.WRITE_MEMORY: 0.06,
                ActionType.FAST_ADAPT: 0.05,
                ActionType.CONSOLIDATE: 0.04,
            }[action]
            return base + 0.04 * matches + 0.02 * updates
        return 0.0
    scale = 0.15 * matches + 0.1 * updates
    if action in {ActionType.WRITE_MEMORY, ActionType.FAST_ADAPT, ActionType.CONSOLIDATE}:
        return scale + 0.2 * current.recurrence_hint
    if action == ActionType.READ_MEMORY:
        return 0.1 * matches
    if action == ActionType.RETRIEVE:
        return 0.05 * matches
    return 0.0
