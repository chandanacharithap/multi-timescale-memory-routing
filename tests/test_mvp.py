from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multitimescale_memory.benchmarks import build_action_coverage_benchmark, build_demo_benchmark
from multitimescale_memory.memory import CorpusRetriever, PersistentMemoryStore
from multitimescale_memory.operations import (
    FrozenParametricModel,
    WorldState,
    action_mask,
    compute_features,
    execute_action,
)
from multitimescale_memory.patches import PatchBank
from multitimescale_memory.reward import RewardWeights, compute_reward
from multitimescale_memory.runner import ExperimentRunner, RunnerConfig
from multitimescale_memory.types import ActionType


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = build_demo_benchmark()
        retriever = CorpusRetriever()
        for episode in self.episodes:
            retriever.ingest(episode.support_docs)
        self.world = WorldState(
            model=FrozenParametricModel(),
            memory=PersistentMemoryStore(),
            retriever=retriever,
            patches=PatchBank(),
        )
        for episode in self.episodes:
            if episode.parametric_answer:
                self.world.model.seed(episode.subject, episode.relation, episode.parametric_answer)

    def test_reward_includes_future_value(self) -> None:
        reward = compute_reward(
            quality=1.0,
            latency=0.5,
            retrieval_calls=1,
            writes=0,
            adapt_steps=0,
            consolidations=0,
            forgetting_risk=0.1,
            future_value=0.8,
            weights=RewardWeights(beta_future=0.5),
        )
        self.assertAlmostEqual(reward.future_value, 0.4)
        self.assertGreater(reward.total, 0.0)

    def test_feature_computation_exposes_novelty_signals(self) -> None:
        episode = self.episodes[1]
        features = compute_features(episode, self.world, [], 1)
        self.assertGreater(features.recurrence_estimate, 0.0)
        self.assertGreater(features.stability_score, 0.0)
        self.assertGreater(features.domain_change_rate, 0.0)
        self.assertGreaterEqual(features.source_agreement_count, 2.0)

    def test_action_mask_covers_all_actions_and_guards_consolidation(self) -> None:
        stable_episode = self.episodes[-1]
        features = compute_features(stable_episode, self.world, [], len(self.episodes) - 1)
        mask = action_mask(features, stable_episode, self.world)
        self.assertTrue(mask[ActionType.RETRIEVE])
        self.assertTrue(mask[ActionType.CONSOLIDATE])

        volatile_episode = self.episodes[5]
        volatile_features = compute_features(volatile_episode, self.world, [], 5)
        volatile_mask = action_mask(volatile_features, volatile_episode, self.world)
        self.assertFalse(volatile_mask[ActionType.CONSOLIDATE])

    def test_memory_read_and_retrieval_are_separate_sources(self) -> None:
        write_episode = self.episodes[3]
        execute_action(ActionType.WRITE_MEMORY, write_episode, self.world)
        read_episode = self.episodes[4]
        read_result = execute_action(ActionType.READ_MEMORY, read_episode, self.world)
        retrieve_result = execute_action(ActionType.RETRIEVE, read_episode, self.world)
        self.assertEqual(read_result.metrics["retrieval_calls"], 0)
        self.assertEqual(read_result.answer, "Austin")
        self.assertEqual(retrieve_result.metrics["retrieval_calls"], 1)

    def test_temporary_patch_lifecycle_detaches(self) -> None:
        episode = self.episodes[1]
        result = execute_action(ActionType.FAST_ADAPT, episode, self.world)
        self.assertIn("temporary_patch_attached", result.side_effects)
        self.assertIn("temporary_patch_detached", result.side_effects)
        self.assertIsNone(self.world.patches.get_temporary(episode.subject, episode.relation))

    def test_consolidation_promotes_patch_and_supports_rollback(self) -> None:
        episode = self.episodes[-2]
        execute_action(ActionType.CONSOLIDATE, episode, self.world)
        durable = self.world.patches.get_durable(episode.subject, episode.relation)
        self.assertIsNotNone(durable)
        self.assertTrue(self.world.patches.rollback(episode.subject, episode.relation))
        self.assertTrue(self.world.patches.recently_rolled_back(episode.subject, episode.relation))

    def test_routed_run_emits_traces_and_reaches_memory_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "traces.jsonl"
            runner = ExperimentRunner(RunnerConfig(trace_path=str(trace_path), baseline_mode="router"))
            result = runner.run(self.episodes)
            self.assertTrue(trace_path.exists())
            lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), len(self.episodes))
            parsed = [json.loads(line) for line in lines]
            actions = {line["action"] for line in parsed}
            self.assertIn(ActionType.READ_MEMORY.value, actions)
            self.assertIn("summary", result)

    def test_oracle_and_baselines_run(self) -> None:
        for mode in ["param_only", "always_retrieve", "retrieve_gate", "three_way_gate", "memory_only", "fast_adapt_only", "oracle"]:
            runner = ExperimentRunner(RunnerConfig(baseline_mode=mode))
            result = runner.run(self.episodes)
            self.assertEqual(result["summary"]["episodes"], len(self.episodes))

    def test_coverage_benchmark_exercises_all_six_actions(self) -> None:
        coverage_episodes = build_action_coverage_benchmark()
        runner = ExperimentRunner(RunnerConfig(baseline_mode="coverage_probe", benchmark_name="coverage"))
        result = runner.run(coverage_episodes)
        actions = {trace["action"] for trace in result["traces"]}
        self.assertEqual(
            actions,
            {
                ActionType.PARAM_ONLY.value,
                ActionType.READ_MEMORY.value,
                ActionType.RETRIEVE.value,
                ActionType.WRITE_MEMORY.value,
                ActionType.FAST_ADAPT.value,
                ActionType.CONSOLIDATE.value,
            },
        )


if __name__ == "__main__":
    unittest.main()
