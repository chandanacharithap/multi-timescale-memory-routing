from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .benchmarks import build_demo_benchmark, get_benchmark
from .memory import CorpusRetriever, PersistentMemoryStore
from .operations import (
    FrozenParametricModel,
    WorldState,
    action_mask,
    compute_features,
    evaluate_quality,
    execute_action,
    forgetting_penalty,
    observed_future_utility,
)
from .patches import PatchBank
from .reward import RewardWeights, compute_reward
from .router import ContextualBanditRouter
from .types import ActionType, EpisodeInput, RunTrace


@dataclass(slots=True)
class RunnerConfig:
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    alpha: float = 0.6
    trace_path: str | None = None
    baseline_mode: str = "router"
    benchmark_name: str = "demo"


class ExperimentRunner:
    def __init__(self, config: RunnerConfig | None = None) -> None:
        self.config = config or RunnerConfig()
        self.router = ContextualBanditRouter(alpha=self.config.alpha)

    def build_world(self, episodes: list[EpisodeInput]) -> WorldState:
        model = FrozenParametricModel()
        memory = PersistentMemoryStore()
        retriever = CorpusRetriever()
        patches = PatchBank()
        for episode in episodes:
            if episode.parametric_answer is not None:
                model.seed(episode.subject, episode.relation, episode.parametric_answer)
            retriever.ingest(episode.support_docs)
        return WorldState(model=model, memory=memory, retriever=retriever, patches=patches)

    def _baseline_action(self, mode: str, mask: dict[ActionType, bool], world: WorldState, episode: EpisodeInput) -> ActionType:
        if mode == "param_only":
            return ActionType.PARAM_ONLY
        if mode == "always_retrieve":
            return ActionType.RETRIEVE if mask[ActionType.RETRIEVE] else ActionType.PARAM_ONLY
        if mode == "memory_only":
            if mask[ActionType.READ_MEMORY]:
                return ActionType.READ_MEMORY
            if mask[ActionType.WRITE_MEMORY]:
                return ActionType.WRITE_MEMORY
            return ActionType.PARAM_ONLY
        if mode == "fast_adapt_only":
            return ActionType.FAST_ADAPT if mask[ActionType.FAST_ADAPT] else ActionType.PARAM_ONLY
        if mode == "retrieve_gate":
            return ActionType.RETRIEVE if episode.parametric_confidence < 0.7 and mask[ActionType.RETRIEVE] else ActionType.PARAM_ONLY
        if mode == "three_way_gate":
            if mask[ActionType.READ_MEMORY] and episode.recurrence_hint > 0.6:
                return ActionType.READ_MEMORY
            if episode.parametric_confidence >= 0.8:
                return ActionType.PARAM_ONLY
            return ActionType.RETRIEVE if mask[ActionType.RETRIEVE] else ActionType.PARAM_ONLY
        if mode == "coverage_probe":
            forced = episode.metadata.get("forced_action")
            if forced is None:
                return ActionType.PARAM_ONLY
            action = ActionType(forced)
            if not mask[action]:
                raise ValueError(f"forced action {forced} is masked for episode {episode.episode_id}")
            return action
        raise ValueError(f"unsupported baseline mode: {mode}")

    def _choose_action(
        self,
        mode: str,
        features,
        mask: dict[ActionType, bool],
        episode: EpisodeInput,
        world: WorldState,
        episodes: list[EpisodeInput],
        index: int,
    ):
        if mode == "router":
            return self.router.decide(features, mask)
        if mode == "oracle":
            best_action = ActionType.PARAM_ONLY
            best_trace = None
            best_total = float("-inf")
            for action, allowed in mask.items():
                if not allowed:
                    continue
                candidate_world = world.clone()
                result = execute_action(action, episode, candidate_world)
                quality = evaluate_quality(result.answer, episode.gold_answer)
                future_value = observed_future_utility(action, index, episodes)
                forgetting = forgetting_penalty(action, features)
                reward = compute_reward(
                    quality=quality,
                    latency=result.latency,
                    retrieval_calls=int(result.metrics["retrieval_calls"]),
                    writes=int(result.metrics["writes"]),
                    adapt_steps=int(result.metrics["adapt_steps"]),
                    consolidations=int(result.metrics["consolidations"]),
                    forgetting_risk=forgetting,
                    future_value=future_value,
                    weights=self.config.reward_weights,
                )
                if reward.total > best_total:
                    best_total = reward.total
                    best_action = action
                    best_trace = reward
            return type(
                "OracleDecision",
                (),
                {
                    "action": best_action,
                    "action_scores": {best_action.value: best_total},
                    "action_mask": {action.value: allowed for action, allowed in mask.items()},
                    "immediate_reward_estimate": best_total - (best_trace.future_value if best_trace else 0.0),
                    "future_value_estimate": best_trace.future_value if best_trace else 0.0,
                    "rationale": "oracle lookahead on immediate reward",
                },
            )()
        action = self._baseline_action(mode, mask, world, episode)
        return type(
            "BaselineDecision",
            (),
            {
                "action": action,
                "action_scores": {action.value: 0.0},
                "action_mask": {item.value: allowed for item, allowed in mask.items()},
                "immediate_reward_estimate": 0.0,
                "future_value_estimate": 0.0,
                "rationale": mode,
            },
        )()

    def run(self, episodes: list[EpisodeInput] | None = None) -> dict[str, object]:
        episodes = episodes or get_benchmark(self.config.benchmark_name)
        world = self.build_world(episodes)
        recent_actions: list[ActionType] = []
        traces: list[RunTrace] = []
        summary = Counter()
        action_summary = Counter()
        analyses = defaultdict(int)

        for index, episode in enumerate(episodes):
            features = compute_features(episode, world, recent_actions, index)
            mask = action_mask(features, episode, world)
            decision = self._choose_action(
                self.config.baseline_mode,
                features,
                mask,
                episode,
                world,
                episodes,
                index,
            )
            result = execute_action(decision.action, episode, world)
            future_value = observed_future_utility(decision.action, index, episodes)
            quality = evaluate_quality(result.answer, episode.gold_answer)
            forgetting = forgetting_penalty(decision.action, features)
            reward = compute_reward(
                quality=quality,
                latency=result.latency,
                retrieval_calls=int(result.metrics["retrieval_calls"]),
                writes=int(result.metrics["writes"]),
                adapt_steps=int(result.metrics["adapt_steps"]),
                consolidations=int(result.metrics["consolidations"]),
                forgetting_risk=forgetting,
                future_value=future_value,
                weights=self.config.reward_weights,
            )
            if self.config.baseline_mode == "router":
                self.router.update(decision.action, features, reward.total, future_value)
            trace = RunTrace(
                episode_id=episode.episode_id,
                dataset_id=episode.dataset_id,
                subject=episode.subject,
                relation=episode.relation,
                gold_answer=episode.gold_answer,
                action=decision.action.value,
                action_mask=decision.action_mask,
                action_scores=decision.action_scores,
                features=asdict(features),
                answer=result.answer,
                reward=asdict(reward),
                evidence_ids=result.evidence_ids,
                touched_memory_ids=result.touched_memory_ids,
                touched_patch_ids=result.touched_patch_ids,
                side_effects=result.side_effects,
                metrics=result.metrics,
            )
            traces.append(trace)
            recent_actions.append(decision.action)
            summary["episodes"] += 1
            summary["correct"] += int(quality)
            summary["retrieval_calls"] += int(result.metrics["retrieval_calls"])
            summary["writes"] += int(result.metrics["writes"])
            summary["adapt_steps"] += int(result.metrics["adapt_steps"])
            summary["consolidations"] += int(result.metrics["consolidations"])
            summary["total_reward"] += reward.total
            action_summary[decision.action.value] += 1

            if mask[ActionType.READ_MEMORY] and decision.action == ActionType.RETRIEVE:
                analyses["retrieval_used_instead_of_memory"] += 1
            if decision.action == ActionType.WRITE_MEMORY and episode.volatility_hint > 0.5:
                analyses["premature_writes"] += 1
            if not mask[ActionType.CONSOLIDATE] and episode.recurrence_hint > 0.7 and episode.volatility_hint > 0.5:
                analyses["consolidation_deferred_for_volatility"] += 1
            if decision.action == ActionType.READ_MEMORY and not result.evidence_ids:
                analyses["memory_reads_without_retrieval"] += 1

        if self.config.trace_path:
            path = Path(self.config.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for trace in traces:
                    handle.write(json.dumps(trace.to_json_dict(), sort_keys=True) + "\n")

        accuracy = summary["correct"] / summary["episodes"] if summary["episodes"] else 0.0
        average_reward = summary["total_reward"] / summary["episodes"] if summary["episodes"] else 0.0
        return {
            "summary": {
                "episodes": summary["episodes"],
                "accuracy": accuracy,
                "retrieval_calls": summary["retrieval_calls"],
                "writes": summary["writes"],
                "adapt_steps": summary["adapt_steps"],
                "consolidations": summary["consolidations"],
                "average_reward": average_reward,
            },
            "actions": dict(action_summary),
            "analyses": dict(analyses),
            "traces": [trace.to_json_dict() for trace in traces],
        }
