from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .benchmarks import get_benchmark
from .freshness import FRESHNESS_BENCHMARK_SOURCE, build_freshness_benchmark
from .knowedit import KNOWEDIT_BENCHMARK_SOURCE, build_knowedit_benchmark
from .memory import BM25Retriever, CorpusRetriever, PersistentMemoryStore
from .mquake import MQUAKE_BENCHMARK_SOURCE, build_mquake_benchmark
from .modeling import FrozenParametricModel, build_model
from .operations import (
    WorldState,
    action_mask,
    calibrated_future_value_scale,
    compute_features,
    evaluate_quality,
    execute_action,
    forgetting_penalty,
    observed_future_utility,
)
from .patches import PatchBank
from .popqa import (
    POPQA_TOTAL_ROWS,
    SqliteMemoryStore,
    build_popqa_benchmark,
    normalize_text,
    prefetch_popqa_benchmark,
)
from .public_freshness import PUBLIC_FRESHNESS_BENCHMARK_SOURCE, build_public_freshness_benchmark
from .reporting import export_freshness_bundle, export_popqa_bundle
from .reward import RewardWeights, compute_reward
from .router import ContextualBanditRouter
from .types import ActionType, EpisodeInput, ResultRow, RunTrace
from .uniedit import UNIEDIT_BENCHMARK_SOURCE, build_uniedit_benchmark


@dataclass(slots=True)
class RunnerConfig:
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    alpha: float = 0.6
    trace_path: str | None = None
    baseline_mode: str = "router"
    benchmark_name: str = "demo"
    model_name: str = "google/flan-t5-small"
    cache_dir: str = ".cache"
    popqa_limit: int = 8
    sqlite_memory_path: str | None = None
    use_v_future: bool = True
    disable_actions: tuple[str, ...] = ()
    allow_popqa_network: bool = False
    popqa_cached_only: bool = False
    popqa_guardrail_mode: str = "strict"
    freshness_limit: int | None = None
    benchmark_limit: int | None = None
    benchmark_sample_seed: int = 2026
    router_seed: int = 0
    stochastic_router: bool = False
    public_freshness_path: str | None = None
    sequence_repeats: int = 1
    future_value_scale: float = 1.0


class ExperimentRunner:
    def __init__(self, config: RunnerConfig | None = None) -> None:
        self.config = config or RunnerConfig()
        self.router = ContextualBanditRouter(
            alpha=self.config.alpha,
            use_future_value=self.config.use_v_future,
            seed=self.config.router_seed,
            stochastic_init=self.config.stochastic_router,
            future_value_scale=self.config.future_value_scale,
        )

    def prefetch_popqa(self, candidate_limit: int | None = None) -> dict[str, int]:
        if self.config.benchmark_name != "popqa":
            raise ValueError("PopQA prefetch is only available for the popqa benchmark.")
        return prefetch_popqa_benchmark(
            limit=self.config.popqa_limit,
            cache_dir=Path(self.config.cache_dir),
            candidate_limit=candidate_limit,
            cached_only=self.config.popqa_cached_only,
        )

    def export_popqa_bundle(self, output_dir: str, audit_limit: int = 50, run_label: str = "post_fix") -> dict[str, object]:
        if self.config.benchmark_name != "popqa":
            raise ValueError("PopQA export is only available for the popqa benchmark.")
        suite_rows = self.run_baseline_suite()
        ablation_rows = self.run_router_ablation_suite()
        trace_path = Path(self.config.trace_path or Path(output_dir) / f"popqa_router_{run_label}_{self.config.popqa_limit}.jsonl")
        router_runner = ExperimentRunner(
            RunnerConfig(
                reward_weights=self.config.reward_weights,
                alpha=self.config.alpha,
                trace_path=str(trace_path),
                baseline_mode="router",
                benchmark_name=self.config.benchmark_name,
                model_name=self.config.model_name,
                cache_dir=self.config.cache_dir,
                popqa_limit=self.config.popqa_limit,
                sqlite_memory_path=self.config.sqlite_memory_path,
                use_v_future=True,
                allow_popqa_network=self.config.allow_popqa_network,
                popqa_cached_only=self.config.popqa_cached_only,
                popqa_guardrail_mode=self.config.popqa_guardrail_mode,
            )
        )
        router_result = router_runner.run()
        bundle_paths = export_popqa_bundle(
            output_dir=Path(output_dir),
            suite_rows=suite_rows,
            ablation_rows=ablation_rows,
            router_result=router_result,
            router_trace_path=trace_path,
            audit_limit=audit_limit,
            run_label=run_label,
        )
        frozen_config = {
            "benchmark": self.config.benchmark_name,
            "popqa_limit": self.config.popqa_limit,
            "model_name": self.config.model_name,
            "cache_dir": self.config.cache_dir,
            "cache_first": not self.config.allow_popqa_network,
            "cached_only_subset": self.config.popqa_cached_only,
            "router_alpha": self.config.alpha,
            "use_v_future": True,
            "guardrail_mode": self.config.popqa_guardrail_mode,
            "run_label": run_label,
            "baselines": ["param_only", "always_retrieve", "retrieve_gate", "router"],
            "ablations": ["router_full", "router_no_read_memory", "router_no_write_memory", "router_no_v_future"],
        }
        frozen_path = Path(output_dir) / "frozen_popqa_stack.json"
        frozen_path.write_text(json.dumps(frozen_config, indent=2), encoding="utf-8")
        return {
            "bundle_paths": bundle_paths,
            "frozen_config": str(frozen_path),
            "router_summary": router_result["summary"],
            "router_actions": router_result["actions"],
            "router_analyses": router_result["analyses"],
            "router_subsets": router_result["subsets"],
        }

    def export_freshness_bundle(self, output_dir: str, audit_limit: int = 50, run_label: str = "freshness_v1") -> dict[str, object]:
        if self.config.benchmark_name not in {"freshness", "freshqa_public", "mquake", "knowedit", "uniedit"}:
            raise ValueError("Freshness export is only available for the freshness benchmark.")
        suite_rows = self.run_baseline_suite()
        ablation_rows = self.run_router_ablation_suite()
        trace_path = Path(self.config.trace_path or Path(output_dir) / f"freshness_router_{run_label}.jsonl")
        router_runner = ExperimentRunner(
            RunnerConfig(
                reward_weights=self.config.reward_weights,
                alpha=self.config.alpha,
                trace_path=str(trace_path),
                baseline_mode="router",
                benchmark_name=self.config.benchmark_name,
                model_name=self.config.model_name,
                cache_dir=self.config.cache_dir,
                freshness_limit=self.config.freshness_limit,
                use_v_future=True,
            )
        )
        router_result = router_runner.run()
        manifest = {
            "benchmark_source": FRESHNESS_BENCHMARK_SOURCE if self.config.benchmark_name == "freshness" else PUBLIC_FRESHNESS_BENCHMARK_SOURCE,
            "benchmark": self.config.benchmark_name,
            "freshness_limit": self.config.freshness_limit,
            "model_name": self.config.model_name,
            "cache_mode": "local_deterministic",
            "baselines": ["param_only", "always_retrieve", "memory_only", "fast_adapt_only", "router"],
            "ablations": ["router_full", "router_no_fast_adapt", "router_no_consolidate", "router_no_v_future"],
            "run_label": run_label,
        }
        if self.config.benchmark_name == "freshqa_public":
            manifest["benchmark_source"] = PUBLIC_FRESHNESS_BENCHMARK_SOURCE
            manifest["data_path"] = self.config.public_freshness_path
            manifest["sequence_repeats"] = self.config.sequence_repeats
        elif self.config.benchmark_name == "mquake":
            manifest["benchmark_source"] = MQUAKE_BENCHMARK_SOURCE
            manifest["benchmark_limit"] = self.config.benchmark_limit
        elif self.config.benchmark_name == "knowedit":
            manifest["benchmark_source"] = KNOWEDIT_BENCHMARK_SOURCE
            manifest["benchmark_limit"] = self.config.benchmark_limit
            manifest["benchmark_sample_seed"] = self.config.benchmark_sample_seed
        elif self.config.benchmark_name == "uniedit":
            manifest["benchmark_source"] = UNIEDIT_BENCHMARK_SOURCE
            manifest["benchmark_limit"] = self.config.benchmark_limit
            manifest["benchmark_sample_seed"] = self.config.benchmark_sample_seed
        bundle_paths = export_freshness_bundle(
            output_dir=Path(output_dir),
            suite_rows=suite_rows,
            ablation_rows=ablation_rows,
            router_result=router_result,
            router_trace_path=trace_path,
            manifest=manifest,
            audit_limit=audit_limit,
            run_label=run_label,
        )
        return {
            "bundle_paths": bundle_paths,
            "manifest": manifest,
            "router_summary": router_result["summary"],
            "router_actions": router_result["actions"],
            "router_analyses": router_result["analyses"],
            "router_subsets": router_result["subsets"],
        }

    @staticmethod
    def _empty_subset_summary() -> dict[str, float]:
        return {
            "episodes": 0,
            "accuracy": 0.0,
            "latency": 0.0,
            "retrieval_calls": 0.0,
            "memory_reads": 0.0,
            "memory_writes": 0.0,
            "stale_answer_rate": 0.0,
            "adaptation_count": 0.0,
            "consolidation_count": 0.0,
            "rollback_count": 0.0,
            "forgetting_probe_accuracy": 0.0,
        }

    def _summarize_subset(self, traces: list[RunTrace]) -> dict[str, float]:
        if not traces:
            return self._empty_subset_summary()
        count = len(traces)
        return {
            "episodes": count,
            "accuracy": sum(trace.reward["quality"] for trace in traces) / count,
            "latency": sum(trace.metrics["latency_observed"] for trace in traces) / count,
            "retrieval_calls": sum(trace.metrics["retrieval_calls"] for trace in traces) / count,
            "memory_reads": sum(trace.metrics["memory_reads"] for trace in traces) / count,
            "memory_writes": sum(trace.metrics["writes"] for trace in traces) / count,
            "stale_answer_rate": sum(float(trace.metrics.get("stale_answer", 0)) for trace in traces) / count,
            "adaptation_count": sum(float(trace.metrics.get("adapt_steps", 0)) for trace in traces) / count,
            "consolidation_count": sum(float(trace.metrics.get("consolidations", 0)) for trace in traces) / count,
            "rollback_count": sum(float(trace.metrics.get("rollback_triggered", 0)) for trace in traces),
            "forgetting_probe_accuracy": (
                sum(float(trace.metrics.get("forgetting_probe_correct", 0)) for trace in traces if trace.metrics.get("update_type") == "forgetting_probe")
                / max(1, sum(1 for trace in traces if trace.metrics.get("update_type") == "forgetting_probe"))
            ),
        }

    def _summarize_subsets(self, traces: list[RunTrace]) -> dict[str, dict[str, float]]:
        recurring = [trace for trace in traces if trace.metrics.get("is_recurring_case", 0) == 1]
        non_recurring = [trace for trace in traces if trace.metrics.get("is_recurring_case", 0) == 0]
        repeated_subject = [trace for trace in traces if trace.metrics.get("repeated_subject", 0) == 1]
        repeated_relation = [trace for trace in traces if trace.metrics.get("repeated_relation", 0) == 1]
        near_duplicate = [trace for trace in traces if trace.metrics.get("near_duplicate_question", 0) == 1]
        recurring_fact = [trace for trace in traces if trace.metrics.get("recurring_fact", 0) == 1]
        subsets = {
            "overall": self._summarize_subset(traces),
            "recurring": self._summarize_subset(recurring),
            "non_recurring": self._summarize_subset(non_recurring),
            "repeated_subject": self._summarize_subset(repeated_subject),
            "repeated_relation": self._summarize_subset(repeated_relation),
            "near_duplicate_question": self._summarize_subset(near_duplicate),
            "recurring_fact": self._summarize_subset(recurring_fact),
        }
        for update_type in ["stable_update", "volatile_update", "confirmation", "rollback_probe", "forgetting_probe"]:
            matching = [trace for trace in traces if trace.metrics.get("update_type") == update_type]
            subsets[update_type] = self._summarize_subset(matching)
        return subsets

    def load_benchmark(self) -> tuple[list[EpisodeInput], list]:
        if self.config.benchmark_name == "popqa":
            return build_popqa_benchmark(
                limit=POPQA_TOTAL_ROWS if self.config.popqa_limit <= 0 else self.config.popqa_limit,
                cache_dir=Path(self.config.cache_dir),
                allow_network=self.config.allow_popqa_network,
            )
        if self.config.benchmark_name == "freshness":
            episodes, corpus_docs, _manifest = build_freshness_benchmark(limit=self.config.freshness_limit)
            return episodes, corpus_docs
        if self.config.benchmark_name == "freshqa_public":
            episodes, corpus_docs, _manifest = build_public_freshness_benchmark(
                data_path=self.config.public_freshness_path,
                limit=self.config.freshness_limit,
                sequence_repeats=self.config.sequence_repeats,
            )
            return episodes, corpus_docs
        if self.config.benchmark_name == "mquake":
            episodes, corpus_docs, _manifest = build_mquake_benchmark(limit=self.config.benchmark_limit)
            return episodes, corpus_docs
        if self.config.benchmark_name == "knowedit":
            episodes, corpus_docs, _manifest = build_knowedit_benchmark(
                sample_per_subset=self.config.benchmark_limit or 200,
                sample_seed=self.config.benchmark_sample_seed,
            )
            return episodes, corpus_docs
        if self.config.benchmark_name == "uniedit":
            episodes, corpus_docs, _manifest = build_uniedit_benchmark(
                sample_per_domain=self.config.benchmark_limit or 200,
                sample_seed=self.config.benchmark_sample_seed,
            )
            return episodes, corpus_docs
        return get_benchmark(self.config.benchmark_name), []

    def build_world(self, episodes: list[EpisodeInput], corpus_docs: list | None = None) -> WorldState:
        if self.config.benchmark_name == "popqa":
            model = build_model(self.config.model_name)
            sqlite_name = (
                f"popqa_memory_{self.config.baseline_mode}_{'vf' if self.config.use_v_future else 'novf'}_{os.getpid()}.sqlite"
            )
            sqlite_path = Path(self.config.sqlite_memory_path or str(Path(self.config.cache_dir) / sqlite_name))
            if sqlite_path.exists():
                sqlite_path.unlink()
            memory = SqliteMemoryStore(sqlite_path)
            retriever = BM25Retriever(corpus_docs or [])
        elif self.config.benchmark_name in {"freshness", "freshqa_public", "mquake", "knowedit", "uniedit"}:
            model = build_model(self.config.model_name)
            memory = PersistentMemoryStore()
            retriever = BM25Retriever(corpus_docs or [])
        else:
            model = FrozenParametricModel()
            memory = PersistentMemoryStore()
            retriever = CorpusRetriever()
        patches = PatchBank()
        for episode in episodes:
            if isinstance(model, FrozenParametricModel) and episode.parametric_answer is not None:
                model.seed(episode.subject, episode.relation, episode.parametric_answer)
            if not corpus_docs:
                retriever.ingest(episode.support_docs)
        return WorldState(model=model, memory=memory, retriever=retriever, patches=patches)

    def _baseline_action(self, mode: str, mask: dict[ActionType, bool], episode: EpisodeInput) -> ActionType:
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
        if mode == "self_rag_like":
            if mask[ActionType.RETRIEVE] and (episode.parametric_confidence < 0.72 or episode.update_expected):
                return ActionType.RETRIEVE
            return ActionType.PARAM_ONLY
        if mode == "memllm_like":
            if mask[ActionType.READ_MEMORY]:
                return ActionType.READ_MEMORY
            if mask[ActionType.WRITE_MEMORY]:
                return ActionType.WRITE_MEMORY
            return ActionType.RETRIEVE if mask[ActionType.RETRIEVE] else ActionType.PARAM_ONLY
        if mode == "wise_like":
            if mask[ActionType.CONSOLIDATE]:
                return ActionType.CONSOLIDATE
            if mask[ActionType.READ_MEMORY]:
                return ActionType.READ_MEMORY
            if mask[ActionType.WRITE_MEMORY] and episode.update_expected:
                return ActionType.WRITE_MEMORY
            return ActionType.RETRIEVE if mask[ActionType.RETRIEVE] else ActionType.PARAM_ONLY
        if mode == "melo_like":
            if episode.update_expected and mask[ActionType.FAST_ADAPT]:
                return ActionType.FAST_ADAPT
            if mask[ActionType.READ_MEMORY] and episode.recurrence_hint >= 0.7:
                return ActionType.READ_MEMORY
            return ActionType.RETRIEVE if mask[ActionType.RETRIEVE] else ActionType.PARAM_ONLY
        if mode == "mello_like":
            probe_role = str(episode.metadata.get("probe_role", ""))
            if probe_role == "update" and mask[ActionType.WRITE_MEMORY]:
                return ActionType.WRITE_MEMORY
            if probe_role != "update" and mask[ActionType.READ_MEMORY]:
                return ActionType.READ_MEMORY
            if mask[ActionType.RETRIEVE]:
                return ActionType.RETRIEVE
            return ActionType.PARAM_ONLY
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

    def _apply_disabled_actions(self, mask: dict[ActionType, bool]) -> dict[ActionType, bool]:
        disabled = {ActionType(item) for item in self.config.disable_actions}
        return {action: (allowed and action not in disabled) for action, allowed in mask.items()}

    def _choose_action(
        self,
        features,
        mask: dict[ActionType, bool],
        episode: EpisodeInput,
        world: WorldState,
        episodes: list[EpisodeInput],
        index: int,
    ):
        mode = self.config.baseline_mode
        if mode in {"router", "router_calibrated"}:
            self.router.future_value_scale = self._effective_future_value_scale(features, episode)
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
                quality = self._quality_for_episode(result.answer, episode)
                future_value = observed_future_utility(action, index, episodes) * self.config.future_value_scale
                reward = compute_reward(
                    quality=quality,
                    latency=result.latency,
                    retrieval_calls=int(result.metrics["retrieval_calls"]),
                    writes=int(result.metrics["writes"]),
                    adapt_steps=int(result.metrics["adapt_steps"]),
                    consolidations=int(result.metrics["consolidations"]),
                    forgetting_risk=forgetting_penalty(action, features),
                    future_value=future_value if self.config.use_v_future else 0.0,
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
        action = self._baseline_action(mode, mask, episode)
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

    def _effective_future_value_scale(self, features, episode: EpisodeInput) -> float:
        scale = self.config.future_value_scale
        if not self.config.use_v_future:
            return 0.0
        if self.config.baseline_mode == "router_calibrated":
            scale *= calibrated_future_value_scale(features, episode)
        return scale

    @staticmethod
    def _quality_for_episode(answer: str, episode: EpisodeInput) -> float:
        alternate_answers = episode.metadata.get("possible_answers")
        if alternate_answers:
            normalized_answer = normalize_text(answer)
            accepted = {normalize_text(item) for item in alternate_answers}
            return 1.0 if normalized_answer in accepted else 0.0
        return evaluate_quality(answer, episode.gold_answer)

    def run(self, episodes: list[EpisodeInput] | None = None) -> dict[str, object]:
        if episodes is None:
            episodes, corpus_docs = self.load_benchmark()
        else:
            corpus_docs = []

        for episode in episodes:
            if self.config.benchmark_name == "popqa":
                episode.metadata["popqa_guardrail_mode"] = self.config.popqa_guardrail_mode

        world = self.build_world(episodes, corpus_docs)
        recent_actions: list[ActionType] = []
        traces: list[RunTrace] = []
        summary = Counter()
        action_summary = Counter()
        analyses = Counter(
            {
                "memory_reads_after_prior_writes": 0,
                "retrieval_skipped_because_memory_exists": 0,
                "retrieval_used_instead_of_memory": 0,
                "premature_writes": 0,
                "consolidation_deferred_for_volatility": 0,
                "memory_reads_without_retrieval": 0,
            }
        )

        for index, episode in enumerate(episodes):
            features = compute_features(episode, world, recent_actions, index)
            mask = self._apply_disabled_actions(action_mask(features, episode, world))
            had_memory_before = world.memory.query(episode.subject, episode.relation) is not None
            decision = self._choose_action(features, mask, episode, world, episodes, index)
            result = execute_action(decision.action, episode, world)
            effective_future_value_scale = self._effective_future_value_scale(features, episode)
            future_value = observed_future_utility(decision.action, index, episodes) * effective_future_value_scale
            quality = self._quality_for_episode(result.answer, episode)
            reward = compute_reward(
                quality=quality,
                latency=result.latency,
                retrieval_calls=int(result.metrics["retrieval_calls"]),
                writes=int(result.metrics["writes"]),
                adapt_steps=int(result.metrics["adapt_steps"]),
                consolidations=int(result.metrics["consolidations"]),
                forgetting_risk=forgetting_penalty(decision.action, features),
                future_value=future_value if self.config.use_v_future else 0.0,
                weights=self.config.reward_weights,
            )
            if self.config.baseline_mode in {"router", "router_calibrated"}:
                self.router.update(
                    decision.action,
                    features,
                    reward.total,
                    future_value if self.config.use_v_future else 0.0,
                )

            result.metrics["had_memory_before"] = 1 if had_memory_before else 0
            result.metrics["latency_observed"] = result.latency
            result.metrics["repeated_subject"] = 1 if episode.metadata.get("repeated_subject", False) else 0
            result.metrics["repeated_relation"] = 1 if episode.metadata.get("repeated_relation", False) else 0
            result.metrics["near_duplicate_question"] = 1 if episode.metadata.get("near_duplicate_question", False) else 0
            result.metrics["recurring_fact"] = 1 if episode.metadata.get("recurring_fact", False) else 0
            result.metrics["is_recurring_case"] = 1 if episode.metadata.get("is_recurring_case", False) else 0
            result.metrics["stale_answer"] = 1 if normalize_text(result.answer) in {normalize_text(item) for item in episode.metadata.get("stale_answers", [])} else 0
            result.metrics["update_type"] = episode.metadata.get("update_type", "none")
            result.metrics["probe_role"] = episode.metadata.get("probe_role")
            result.metrics["probe_family"] = episode.metadata.get("probe_family")
            result.metrics["subset"] = episode.metadata.get("subset")
            result.metrics["case_id"] = episode.metadata.get("case_id")
            result.metrics["original_answers"] = episode.metadata.get("original_answers", [])
            result.metrics["conflict_detected"] = 1 if "conflict_detected" in result.side_effects else 0
            result.metrics["temporary_patch_used"] = 1 if decision.action == ActionType.FAST_ADAPT else 0
            result.metrics["durable_patch_used"] = 1 if ("durable_patch_read" in result.side_effects or "patch_promoted" in result.side_effects) else 0
            result.metrics["rollback_triggered"] = 1 if "rollback_triggered" in result.side_effects else 0
            result.metrics["forgetting_probe_correct"] = int(quality) if episode.metadata.get("update_type") == "forgetting_probe" else 0
            result.metrics["volatility_score"] = features.volatility_score
            result.metrics["effective_future_value_scale"] = effective_future_value_scale
            result.metrics["router_variant"] = self.config.baseline_mode

            trace = RunTrace(
                episode_id=episode.episode_id,
                dataset_id=episode.dataset_id,
                question=episode.question,
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
            summary["memory_reads"] += int(result.metrics["memory_reads"])
            summary["writes"] += int(result.metrics["writes"])
            summary["adapt_steps"] += int(result.metrics["adapt_steps"])
            summary["consolidations"] += int(result.metrics["consolidations"])
            summary["rollback_count"] += int(result.metrics.get("rollback_triggered", 0))
            summary["durable_patch_uses"] += int(result.metrics.get("durable_patch_used", 0))
            summary["stale_answers"] += int(result.metrics.get("stale_answer", 0))
            if result.metrics.get("update_type") == "forgetting_probe":
                summary["forgetting_probe_total"] += 1
                summary["forgetting_probe_correct"] += int(result.metrics.get("forgetting_probe_correct", 0))
            summary["latency"] += result.latency
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
            if had_memory_before and decision.action == ActionType.READ_MEMORY:
                analyses["memory_reads_after_prior_writes"] += 1
            if had_memory_before and mask[ActionType.RETRIEVE] and decision.action == ActionType.READ_MEMORY:
                analyses["retrieval_skipped_because_memory_exists"] += 1

        if self.config.trace_path:
            path = Path(self.config.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for trace in traces:
                    handle.write(json.dumps(trace.to_json_dict(), sort_keys=True) + "\n")

        accuracy = summary["correct"] / summary["episodes"] if summary["episodes"] else 0.0
        average_reward = summary["total_reward"] / summary["episodes"] if summary["episodes"] else 0.0
        stale_answer_rate = summary["stale_answers"] / summary["episodes"] if summary["episodes"] else 0.0
        forgetting_probe_accuracy = summary["forgetting_probe_correct"] / summary["forgetting_probe_total"] if summary["forgetting_probe_total"] else 0.0
        forgetting_delta = forgetting_probe_accuracy - 1.0 if summary["forgetting_probe_total"] else 0.0
        return {
            "summary": {
                "episodes": summary["episodes"],
                "accuracy": accuracy,
                "retrieval_calls": summary["retrieval_calls"],
                "memory_reads": summary["memory_reads"],
                "writes": summary["writes"],
                "adapt_steps": summary["adapt_steps"],
                "consolidations": summary["consolidations"],
                "rollback_count": summary["rollback_count"],
                "durable_patch_uses": summary["durable_patch_uses"],
                "stale_answer_rate": stale_answer_rate,
                "forgetting_probe_accuracy": forgetting_probe_accuracy,
                "forgetting_delta": forgetting_delta,
                "latency": summary["latency"] / summary["episodes"] if summary["episodes"] else 0.0,
                "average_reward": average_reward,
            },
            "actions": dict(action_summary),
            "analyses": dict(analyses),
            "subsets": self._summarize_subsets(traces),
            "traces": [trace.to_json_dict() for trace in traces],
        }

    def run_baseline_suite(self, episodes: list[EpisodeInput] | None = None) -> list[ResultRow]:
        rows: list[ResultRow] = []
        if self.config.benchmark_name in {"freshness", "freshqa_public", "mquake", "knowedit", "uniedit"}:
            modes = ["param_only", "always_retrieve", "memory_only", "fast_adapt_only", "self_rag_like", "memllm_like", "wise_like", "melo_like", "router", "router_calibrated"]
            if self.config.benchmark_name == "mquake":
                modes.insert(5, "mello_like")
        else:
            modes = ["param_only", "always_retrieve", "retrieve_gate", "self_rag_like", "memllm_like", "router", "router_calibrated"]
        for mode in modes:
            runner = ExperimentRunner(
                RunnerConfig(
                    reward_weights=self.config.reward_weights,
                    alpha=self.config.alpha,
                    trace_path=None,
                    baseline_mode=mode,
                    benchmark_name=self.config.benchmark_name,
                    model_name=self.config.model_name,
                    cache_dir=self.config.cache_dir,
                    popqa_limit=self.config.popqa_limit,
                    sqlite_memory_path=self.config.sqlite_memory_path,
                    use_v_future=self.config.use_v_future,
                    allow_popqa_network=self.config.allow_popqa_network,
                    popqa_guardrail_mode=self.config.popqa_guardrail_mode,
                    freshness_limit=self.config.freshness_limit,
                    router_seed=self.config.router_seed,
                    stochastic_router=self.config.stochastic_router,
                    public_freshness_path=self.config.public_freshness_path,
                    sequence_repeats=self.config.sequence_repeats,
                )
            )
            result = runner.run(episodes if episodes is not None else None)
            summary = result["summary"]
            recurring = result["subsets"]["recurring"]
            non_recurring = result["subsets"]["non_recurring"]
            rows.append(
                ResultRow(
                    mode=mode,
                    answer_quality=float(summary["accuracy"]),
                    recurring_quality=float(recurring["accuracy"]),
                    non_recurring_quality=float(non_recurring["accuracy"]),
                    latency=float(summary["latency"]),
                    retrieval_calls=int(summary["retrieval_calls"]),
                    memory_reads=int(summary["memory_reads"]),
                    memory_writes=int(summary["writes"]),
                    adaptation_count=int(summary["adapt_steps"]),
                    recurring_retrieval_calls=float(recurring["retrieval_calls"]),
                    recurring_memory_reads=float(recurring["memory_reads"]),
                    recurring_memory_writes=float(recurring["memory_writes"]),
                    action_distribution=result["actions"],
                    extra_metrics={
                        "stale_answer_rate": float(summary.get("stale_answer_rate", 0.0)),
                        "consolidation_count": float(summary.get("consolidations", 0.0)),
                        "rollback_count": float(summary.get("rollback_count", 0.0)),
                        "forgetting_delta": float(summary.get("forgetting_delta", 0.0)),
                    },
                )
            )
        return rows

    def run_router_ablation_suite(self, episodes: list[EpisodeInput] | None = None) -> list[ResultRow]:
        rows: list[ResultRow] = []
        if self.config.benchmark_name in {"freshness", "freshqa_public", "mquake", "knowedit", "uniedit"}:
            variants = [
                ("router_full", (), True),
                ("router_no_fast_adapt", ("fast_adapt",), True),
                ("router_no_consolidate", ("consolidate",), True),
                ("router_no_v_future", (), False),
            ]
        else:
            variants = [
                ("router_full", ("",)[:0], True),
                ("router_no_read_memory", ("read_memory",), True),
                ("router_no_write_memory", ("write_memory",), True),
                ("router_no_v_future", (), False),
            ]
        for mode, disabled_actions, use_v_future in variants:
            runner = ExperimentRunner(
                RunnerConfig(
                    reward_weights=self.config.reward_weights,
                    alpha=self.config.alpha,
                    trace_path=None,
                    baseline_mode="router",
                    benchmark_name=self.config.benchmark_name,
                    model_name=self.config.model_name,
                    cache_dir=self.config.cache_dir,
                    popqa_limit=self.config.popqa_limit,
                    sqlite_memory_path=self.config.sqlite_memory_path,
                    use_v_future=use_v_future,
                    disable_actions=disabled_actions,
                    allow_popqa_network=self.config.allow_popqa_network,
                    popqa_guardrail_mode=self.config.popqa_guardrail_mode,
                    freshness_limit=self.config.freshness_limit,
                    router_seed=self.config.router_seed,
                    stochastic_router=self.config.stochastic_router,
                    public_freshness_path=self.config.public_freshness_path,
                    sequence_repeats=self.config.sequence_repeats,
                )
            )
            result = runner.run(episodes if episodes is not None else None)
            summary = result["summary"]
            recurring = result["subsets"]["recurring"]
            non_recurring = result["subsets"]["non_recurring"]
            rows.append(
                ResultRow(
                    mode=mode,
                    answer_quality=float(summary["accuracy"]),
                    recurring_quality=float(recurring["accuracy"]),
                    non_recurring_quality=float(non_recurring["accuracy"]),
                    latency=float(summary["latency"]),
                    retrieval_calls=int(summary["retrieval_calls"]),
                    memory_reads=int(summary["memory_reads"]),
                    memory_writes=int(summary["writes"]),
                    adaptation_count=int(summary["adapt_steps"]),
                    recurring_retrieval_calls=float(recurring["retrieval_calls"]),
                    recurring_memory_reads=float(recurring["memory_reads"]),
                    recurring_memory_writes=float(recurring["memory_writes"]),
                    action_distribution=result["actions"],
                    extra_metrics={
                        "stale_answer_rate": float(summary.get("stale_answer_rate", 0.0)),
                        "consolidation_count": float(summary.get("consolidations", 0.0)),
                        "rollback_count": float(summary.get("rollback_count", 0.0)),
                        "forgetting_delta": float(summary.get("forgetting_delta", 0.0)),
                    },
                )
            )
        return rows


def format_result_table(rows: list[ResultRow]) -> str:
    header = (
        "| Mode | Overall Quality | Recurring Quality | Non-Recurring Quality | Latency | "
        "Recurring Retrieval Calls | Recurring Memory Reads | Recurring Memory Writes | Action Distribution |"
    )
    divider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    body = []
    for row in rows:
        body.append(
            "| {mode} | {overall:.3f} | {recurring:.3f} | {non_recurring:.3f} | {latency:.3f} | {rec_retrieval:.3f} | {rec_reads:.3f} | {rec_writes:.3f} | {actions} |".format(
                mode=row.mode,
                overall=row.answer_quality,
                recurring=row.recurring_quality,
                non_recurring=row.non_recurring_quality,
                latency=row.latency,
                rec_retrieval=row.recurring_retrieval_calls,
                rec_reads=row.recurring_memory_reads,
                rec_writes=row.recurring_memory_writes,
                actions=json.dumps(row.action_distribution, sort_keys=True),
            )
        )
    return "\n".join([header, divider, *body])


def format_freshness_result_table(rows: list[ResultRow]) -> str:
    header = (
        "| Mode | Answer Quality | Stale Answer Rate | Latency | Retrieval Calls | Memory Reads | "
        "Memory Writes | Adapt Count | Consolidations | Rollbacks | Forgetting Delta | Action Distribution |"
    )
    divider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    body = []
    for row in rows:
        body.append(
            "| {mode} | {quality:.3f} | {stale:.3f} | {latency:.3f} | {retrieval} | {reads} | {writes} | {adapt} | {consolidations:.0f} | {rollbacks:.0f} | {forgetting:.3f} | {actions} |".format(
                mode=row.mode,
                quality=row.answer_quality,
                stale=row.extra_metrics.get("stale_answer_rate", 0.0),
                latency=row.latency,
                retrieval=row.retrieval_calls,
                reads=row.memory_reads,
                writes=row.memory_writes,
                adapt=row.adaptation_count,
                consolidations=row.extra_metrics.get("consolidation_count", 0.0),
                rollbacks=row.extra_metrics.get("rollback_count", 0.0),
                forgetting=row.extra_metrics.get("forgetting_delta", 0.0),
                actions=json.dumps(row.action_distribution, sort_keys=True),
            )
        )
    return "\n".join([header, divider, *body])
