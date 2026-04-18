from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from multitimescale_memory.benchmarks import build_action_coverage_benchmark, build_demo_benchmark
from multitimescale_memory.campaign import run_manifest_campaign
from multitimescale_memory.freshness import build_freshness_benchmark
from multitimescale_memory.freshqa_export import (
    build_public_freshqa_export,
    build_public_freshqa_traceability,
    public_freshqa_provenance_manifest,
)
from multitimescale_memory.journal import (
    aggregate_journal_rows,
    export_freshqa_future_scale_sweep,
    export_journal_bundle,
    run_freshqa_future_scale_sweep,
    run_journal_matrix,
)
from multitimescale_memory.knowedit import build_knowedit_benchmark
from multitimescale_memory.memory import BM25Retriever, CorpusRetriever, PersistentMemoryStore
from multitimescale_memory.modeling import FrozenParametricModel, HuggingFaceQAModel, model_runtime_status, resolve_model_spec
from multitimescale_memory.mquake import build_mquake_benchmark
from multitimescale_memory.popqa import parse_answer_list
from multitimescale_memory.public_freshness import build_public_freshness_benchmark, validate_public_freshness_rows
from multitimescale_memory.readiness import benchmark_input_status, deep_model_runtime_probe, workspace_readiness_report
from multitimescale_memory.uniedit import build_uniedit_benchmark
from multitimescale_memory.operations import (
    WorldState,
    action_mask,
    calibrated_future_value_scale,
    compute_features,
    execute_action,
)
from multitimescale_memory.patches import PatchBank
from multitimescale_memory.reward import RewardWeights, compute_reward
from multitimescale_memory.reporting import (
    NON_RECURRING_AUDIT_LABELS,
    PUBLIC_FRESHQA_AUDIT_LABELS,
    build_non_recurring_error_audit,
    build_public_freshqa_error_audit,
    export_popqa_bundle,
)
from multitimescale_memory.runner import ExperimentRunner, RunnerConfig, format_result_table
from multitimescale_memory.types import ResultRow, SupportDoc
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
        self.assertGreaterEqual(features.memory_alignment_score, 0.0)

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

    def test_popqa_strict_guardrails_force_retrieval_on_weak_nonrecurring_case(self) -> None:
        episode = replace(
            self.episodes[0],
            dataset_id="popqa",
            support_docs=[
                SupportDoc(
                    "wiki:test",
                    "Alice Example is a scientist from Canada.",
                    "scientist",
                    "wikipedia",
                    1,
                    trust=0.95,
                )
            ],
            recurrence_hint=0.2,
            parametric_confidence=0.0,
            metadata={**self.episodes[0].metadata, "popqa_guardrail_mode": "strict"},
        )
        features = compute_features(episode, self.world, [], 0)
        strict_mask = action_mask(features, episode, self.world)
        self.assertFalse(strict_mask[ActionType.PARAM_ONLY])
        self.assertTrue(strict_mask[ActionType.RETRIEVE])

        legacy_episode = replace(
            episode,
            metadata={**episode.metadata, "popqa_guardrail_mode": "legacy"},
        )
        legacy_features = compute_features(legacy_episode, self.world, [], 0)
        legacy_mask = action_mask(legacy_features, legacy_episode, self.world)
        self.assertTrue(legacy_mask[ActionType.PARAM_ONLY])

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

    def test_demo_benchmark_now_hits_param_only(self) -> None:
        runner = ExperimentRunner(RunnerConfig(baseline_mode="router", benchmark_name="demo"))
        result = runner.run(self.episodes)
        actions = {trace["action"] for trace in result["traces"]}
        self.assertIn(ActionType.PARAM_ONLY.value, actions)

    def test_popqa_answer_list_parser(self) -> None:
        answers = parse_answer_list("[\"politician\", \"political leader\"]")
        self.assertEqual(answers, ["politician", "political leader"])

    def test_bm25_retriever_returns_real_doc_ids(self) -> None:
        retriever = BM25Retriever(
            [
                SupportDoc("wiki:1", "George Rankin was a politician from Canada.", "", "wikipedia", 0, trust=0.8),
                SupportDoc("wiki:2", "Paris is the capital of France.", "", "wikipedia", 0, trust=0.8),
            ]
        )
        docs = retriever.retrieve("What is George Rankin's occupation?", "George Rankin", "occupation")
        self.assertEqual(docs[0].doc_id, "wiki:1")

    def test_result_table_formatter(self) -> None:
        rows = ExperimentRunner(RunnerConfig()).run_baseline_suite(self.episodes)
        table = format_result_table(rows)
        self.assertIn("| Mode | Overall Quality |", table)

    def test_subset_analysis_is_reported(self) -> None:
        runner = ExperimentRunner(RunnerConfig(baseline_mode="router", benchmark_name="demo"))
        result = runner.run(self.episodes)
        self.assertIn("subsets", result)
        self.assertIn("overall", result["subsets"])
        self.assertIn("recurring", result["subsets"])

    def test_v_future_ablation_runs(self) -> None:
        rows = ExperimentRunner(RunnerConfig()).run_router_ablation_suite(self.episodes)
        self.assertEqual(
            [row.mode for row in rows],
            ["router_full", "router_no_read_memory", "router_no_write_memory", "router_no_v_future"],
        )

    def test_non_recurring_audit_uses_only_allowed_labels(self) -> None:
        traces = [
            {
                "episode_id": "e1",
                "subject": "Alice",
                "relation": "occupation",
                "action": "param_only",
                "answer": "writer",
                "gold_answer": "scientist",
                "reward": {"quality": 0.0},
                "features": {
                    "model_confidence": 0.2,
                    "retrieval_quality_estimate": 0.8,
                    "memory_alignment_score": 0.0,
                    "recurrence_estimate": 0.1,
                },
                "metrics": {"is_recurring_case": 0, "had_memory_before": 0},
                "evidence_ids": [],
            },
            {
                "episode_id": "e2",
                "subject": "Bob",
                "relation": "capital",
                "action": "read_memory",
                "answer": "Paris",
                "gold_answer": "Lyon",
                "reward": {"quality": 0.0},
                "features": {
                    "model_confidence": 0.3,
                    "retrieval_quality_estimate": 0.9,
                    "memory_alignment_score": 0.1,
                    "recurrence_estimate": 0.2,
                },
                "metrics": {"is_recurring_case": 0, "had_memory_before": 1},
                "evidence_ids": [],
            },
        ]
        audit = build_non_recurring_error_audit(traces, limit=50)
        self.assertEqual(len(audit), 2)
        self.assertTrue(all(row["label"] in NON_RECURRING_AUDIT_LABELS for row in audit))
        self.assertTrue(all(row["recurring_case"] is False for row in audit))

    def test_export_bundle_writes_audit_summary_and_run_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            rows = [
                ResultRow(
                    mode="router",
                    answer_quality=0.2,
                    recurring_quality=0.3,
                    non_recurring_quality=0.1,
                    latency=0.4,
                    retrieval_calls=1,
                    memory_reads=1,
                    memory_writes=1,
                    adaptation_count=0,
                    recurring_retrieval_calls=0.5,
                    recurring_memory_reads=0.9,
                    recurring_memory_writes=0.6,
                    action_distribution={"retrieve": 1},
                )
            ]
            router_result = {
                "actions": {"retrieve": 1},
                "traces": [
                    {
                        "episode_id": "e1",
                        "subject": "Alice",
                        "relation": "occupation",
                        "action": "param_only",
                        "answer": "writer",
                        "gold_answer": "scientist",
                        "reward": {"quality": 0.0},
                        "features": {
                            "model_confidence": 0.2,
                            "retrieval_quality_estimate": 0.8,
                            "memory_alignment_score": 0.0,
                            "recurrence_estimate": 0.1,
                        },
                        "metrics": {"is_recurring_case": 0, "had_memory_before": 0},
                        "evidence_ids": [],
                    }
                ],
            }
            paths = export_popqa_bundle(
                output_dir=output_dir,
                suite_rows=rows,
                ablation_rows=rows,
                router_result=router_result,
                router_trace_path=Path("/tmp/router.jsonl"),
                audit_limit=50,
                run_label="pre_fix",
            )
            self.assertTrue((output_dir / "error_audit_summary.json").exists())
            self.assertEqual((output_dir / "run_label.txt").read_text(encoding="utf-8"), "pre_fix")
            self.assertIn("error_audit_summary", paths)

    def test_freshness_benchmark_is_time_ordered_and_tagged(self) -> None:
        episodes, corpus_docs, manifest = build_freshness_benchmark()
        self.assertEqual(len(episodes), 15)
        self.assertEqual(sorted(episode.timestamp for episode in episodes), [episode.timestamp for episode in episodes])
        self.assertTrue(any(episode.metadata.get("update_type") == "volatile_update" for episode in episodes))
        self.assertTrue(any(episode.metadata.get("update_type") == "rollback_probe" for episode in episodes))
        self.assertEqual(manifest["benchmark_source"], "bundled_freshqa_style_snapshot_v1")
        self.assertGreater(len(corpus_docs), 0)

    def test_freshness_volatile_update_suppresses_consolidation(self) -> None:
        episodes, corpus_docs, _manifest = build_freshness_benchmark()
        volatile_episode = next(episode for episode in episodes if episode.metadata.get("update_type") == "volatile_update")
        world = WorldState(
            model=FrozenParametricModel(),
            memory=PersistentMemoryStore(),
            retriever=BM25Retriever(corpus_docs),
            patches=PatchBank(),
        )
        features = compute_features(volatile_episode, world, [], 0)
        mask = action_mask(features, volatile_episode, world)
        self.assertFalse(mask[ActionType.CONSOLIDATE])
        self.assertTrue(mask[ActionType.FAST_ADAPT])

    def test_freshness_rollback_probe_removes_durable_patch(self) -> None:
        episodes, corpus_docs, _manifest = build_freshness_benchmark()
        world = WorldState(
            model=FrozenParametricModel(),
            memory=PersistentMemoryStore(),
            retriever=BM25Retriever(corpus_docs),
            patches=PatchBank(),
        )
        update_episode = next(episode for episode in episodes if episode.episode_id == "terra-update-1")
        confirm_episode = next(episode for episode in episodes if episode.episode_id == "terra-confirm-1")
        rollback_episode = next(episode for episode in episodes if episode.episode_id == "terra-rollback-1")
        execute_action(ActionType.FAST_ADAPT, update_episode, world)
        execute_action(ActionType.CONSOLIDATE, confirm_episode, world)
        self.assertIsNotNone(world.patches.get_durable(confirm_episode.subject, confirm_episode.relation))
        rollback_result = execute_action(ActionType.RETRIEVE, rollback_episode, world)
        self.assertIn("rollback_triggered", rollback_result.side_effects)
        self.assertIsNone(world.patches.get_durable(confirm_episode.subject, confirm_episode.relation))

    def test_public_freshness_loader_preserves_timestamp_order(self) -> None:
        episodes, corpus_docs, manifest = build_public_freshness_benchmark(
            data_path=Path("data/freshqa_public_sample.jsonl"),
            sequence_repeats=2,
        )
        self.assertEqual(manifest["benchmark_source"], "public_freshqa_track_v1")
        self.assertEqual(len(episodes), 8)
        self.assertEqual(len(corpus_docs), 6)
        self.assertEqual(sorted(episode.timestamp for episode in episodes), [episode.timestamp for episode in episodes])
        self.assertTrue(all(episode.dataset_id == "freshqa_public" for episode in episodes))

    def test_new_baseline_families_run_on_public_freshness_fixture(self) -> None:
        for mode in ["self_rag_like", "memllm_like", "wise_like", "melo_like"]:
            runner = ExperimentRunner(
                RunnerConfig(
                    baseline_mode=mode,
                    benchmark_name="freshqa_public",
                    model_name="frozen-parametric",
                    public_freshness_path="data/freshqa_public_sample.jsonl",
                )
            )
            result = runner.run()
            self.assertEqual(result["summary"]["episodes"], 4)

    def test_public_freshness_retrieve_path_uses_grounded_answers(self) -> None:
        runner = ExperimentRunner(
            RunnerConfig(
                baseline_mode="always_retrieve",
                benchmark_name="freshqa_public",
                model_name="frozen-parametric",
                public_freshness_path="data/freshqa_public_sample.jsonl",
            )
        )
        result = runner.run()
        self.assertGreater(result["summary"]["accuracy"], 0.0)

    def test_model_spec_registry_resolves_known_and_custom_names(self) -> None:
        self.assertEqual(resolve_model_spec("google/flan-t5-small").family, "flan-t5")
        self.assertEqual(resolve_model_spec("custom/model").backend, "hf_seq2seq")
        self.assertTrue(hasattr(HuggingFaceQAModel, "answer_with_evidence"))

    def test_public_freshness_validator_accepts_sample_fixture(self) -> None:
        rows = []
        with Path("data/freshqa_public_sample.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        report = validate_public_freshness_rows(rows)
        self.assertEqual(report["errors"], [])
        self.assertGreaterEqual(report["repeated_subject_relation_pairs"], 1)

    def test_build_public_freshqa_export_from_csv_snapshots(self) -> None:
        csv_one = (
            "warning\n"
            "\n"
            "id,split,question,effective_year,next_review,false_premise,num_hops,fact_type,source,answer_0,answer_1,note\n"
            "1,TEST,Who leads AcmeAI?,2025,occasionally,FALSE,one-hop,fast-changing,\"https://example.com/a\nhttps://example.com/a2\",Jordan Lee,Jordan Lee PhD,\n"
            "2,TEST,Where is AcmeAI headquartered?,2025,yearly,FALSE,one-hop,slow-changing,https://example.com/h,Austin,,\n"
        )
        csv_two = (
            "warning\n"
            "\n"
            "id,split,question,effective_year,next_review,false_premise,num_hops,fact_type,source,answer_0,answer_1,note\n"
            "1,TEST,Who leads AcmeAI?,2025,occasionally,FALSE,one-hop,fast-changing,https://example.com/b,Sam Patel,,\n"
            "2,TEST,Where is AcmeAI headquartered?,2025,yearly,FALSE,one-hop,slow-changing,https://example.com/h,Austin,,\n"
        )
        csv_three = (
            "warning\n"
            "\n"
            "id,split,question,effective_year,next_review,false_premise,num_hops,fact_type,source,answer_0,answer_1,note\n"
            "1,TEST,Who leads AcmeAI?,2025,occasionally,FALSE,one-hop,fast-changing,https://example.com/c,Sam Patel,,\n"
            "2,TEST,Where is AcmeAI headquartered?,2025,yearly,FALSE,one-hop,slow-changing,https://example.com/h,Austin,,\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            snapshot_one = tmpdir_path / "freshqa_2025-03-10.csv"
            snapshot_two = tmpdir_path / "freshqa_2025-03-17.csv"
            snapshot_three = tmpdir_path / "freshqa_2025-03-24.csv"
            output_path = tmpdir_path / "freshqa_export.jsonl"
            snapshot_one.write_text(csv_one, encoding="utf-8")
            snapshot_two.write_text(csv_two, encoding="utf-8")
            snapshot_three.write_text(csv_three, encoding="utf-8")
            manifest = build_public_freshqa_export(
                [snapshot_one, snapshot_two, snapshot_three],
                output_path,
                min_snapshots_per_question=2,
                slice_mode="sequence",
            )
            self.assertEqual(manifest["episodes"], 3)
            self.assertEqual(manifest["changed_questions"], 1)
            self.assertEqual(manifest["forgetting_probe_count"], 0)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row["metadata"]["update_type"] in {"stable_update", "volatile_update"} for row in rows))
            self.assertTrue(any(row["metadata"]["update_type"] == "confirmation" for row in rows))
            self.assertTrue(all("possible_answers" in row["metadata"] for row in rows))
            self.assertTrue(rows[1]["metadata"]["answer_changed_since_last_seen"])
            self.assertTrue(rows[1]["metadata"]["prior_stale_answer_available"])
            self.assertTrue(rows[0]["metadata"]["has_aliases"])

    def test_public_freshness_guardrails_require_confirmation_before_read_or_consolidate(self) -> None:
        episode = build_public_freshness_benchmark(
            data_path=Path("data/freshqa_public_sample.jsonl"),
            sequence_repeats=1,
        )[0][1]
        world = WorldState(
            model=FrozenParametricModel(),
            memory=PersistentMemoryStore(),
            retriever=BM25Retriever(episode.support_docs),
            patches=PatchBank(),
        )
        execute_action(ActionType.WRITE_MEMORY, episode, world)
        features = compute_features(episode, world, [], 1)
        mask = action_mask(features, episode, world)
        self.assertFalse(mask[ActionType.READ_MEMORY])
        self.assertFalse(mask[ActionType.CONSOLIDATE])

    def test_calibrated_future_value_scale_suppresses_public_change_cases(self) -> None:
        episodes, corpus_docs, _manifest = build_public_freshness_benchmark(
            data_path=Path("data/freshqa_public_sample.jsonl"),
            sequence_repeats=1,
        )
        world = WorldState(
            model=FrozenParametricModel(),
            memory=PersistentMemoryStore(),
            retriever=BM25Retriever(corpus_docs),
            patches=PatchBank(),
        )
        changed_episode = replace(
            episodes[1],
            metadata={
                **episodes[1].metadata,
                "question_seen_before": True,
                "answer_changed_since_last_seen": True,
                "prior_stale_answer_available": True,
                "confirmation_count_before": 0,
                "change_count_before": 1,
            },
        )
        stable_episode = replace(
            episodes[1],
            metadata={
                **episodes[1].metadata,
                "question_seen_before": True,
                "answer_changed_since_last_seen": False,
                "prior_stale_answer_available": False,
                "confirmation_count_before": 2,
                "change_count_before": 1,
            },
        )
        changed_features = compute_features(changed_episode, world, [], 1)
        stable_features = compute_features(stable_episode, world, [], 1)
        self.assertLess(calibrated_future_value_scale(changed_features, changed_episode), 0.2)
        self.assertGreaterEqual(calibrated_future_value_scale(stable_features, stable_episode), 0.5)

    def test_public_freshqa_audit_compares_full_and_calibrated_traces(self) -> None:
        router_traces = [
            {
                "episode_id": "e1",
                "subject": "AcmeAI",
                "relation": "ceo",
                "action": "param_only",
                "answer": "Jordan Lee",
                "gold_answer": "Sam Patel",
                "reward": {"quality": 0.0},
                "features": {
                    "question_seen_before": 1.0,
                    "answer_changed_since_last_seen": 1.0,
                    "contradiction_risk": 0.6,
                    "retrieval_quality_estimate": 0.95,
                    "has_aliases": 0.0,
                },
                "metrics": {
                    "update_type": "volatile_update",
                    "had_memory_before": 0,
                    "effective_future_value_scale": 0.75,
                    "stale_answer": 1,
                },
                "evidence_ids": ["doc1"],
            }
        ]
        calibrated_traces = [
            {
                "episode_id": "e1",
                "subject": "AcmeAI",
                "relation": "ceo",
                "action": "retrieve",
                "answer": "Sam Patel",
                "gold_answer": "Sam Patel",
                "reward": {"quality": 1.0},
                "features": {},
                "metrics": {"update_type": "volatile_update"},
                "evidence_ids": ["doc1"],
            }
        ]
        audit = build_public_freshqa_error_audit(router_traces, calibrated_traces, limit=10)
        self.assertEqual(len(audit), 1)
        self.assertIn(audit[0]["label"], PUBLIC_FRESHQA_AUDIT_LABELS)
        self.assertEqual(audit[0]["calibrated_action"], "retrieve")

    def test_public_freshqa_provenance_manifest_and_traceability(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            snapshot = tmpdir_path / "freshqa_2025-03-10.csv"
            derived_main = tmpdir_path / "freshqa_public_main_v2.jsonl"
            derived_sequence = tmpdir_path / "freshqa_public_sequence_v2.jsonl"
            snapshot.write_text(
                "warning\n\n"
                "id,split,question,effective_year,next_review,false_premise,num_hops,fact_type,source,answer_0,answer_1,note\n"
                "1,TEST,Who leads AcmeAI?,2025,occasionally,FALSE,one-hop,fast-changing,https://example.com/a,Jordan Lee,Jordan Lee PhD,\n",
                encoding="utf-8",
            )
            derived_main.write_text(
                json.dumps(
                    {
                        "episode_id": "freshqa::1::2025-03-10::1",
                        "question": "Who leads AcmeAI?",
                        "gold_answer": "Jordan Lee",
                        "metadata": {
                            "snapshot_date": "2025-03-10",
                            "update_type": "stable_update",
                            "stale_answers": [],
                            "possible_answers": ["Jordan Lee", "Jordan Lee PhD"],
                            "rollback_probe": False,
                            "confirmation_count_before": 0,
                            "change_count_before": 0,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            derived_sequence.write_text(derived_main.read_text(encoding="utf-8"), encoding="utf-8")
            manifest = public_freshqa_provenance_manifest(
                [snapshot],
                derived_paths=[derived_main, derived_sequence],
                derivation_script_path=Path("multitimescale_memory/freshqa_export.py"),
            )
            traceability = build_public_freshqa_traceability({"main": derived_main, "sequence": derived_sequence})
            self.assertEqual(len(manifest["source_files"]), 1)
            self.assertEqual(len(manifest["derived_files"]), 2)
            self.assertEqual(len(traceability), 2)
            self.assertEqual(traceability[0]["source_week"], "2025-03-10")

    def test_model_runtime_status_reports_missing_hf_dependencies(self) -> None:
        status = model_runtime_status("google/flan-t5-small")
        self.assertEqual(status["backend"], "hf_seq2seq")
        self.assertIn("missing_dependencies", status)
        self.assertIn("loadable", status)
        self.assertEqual(resolve_model_spec("Qwen/Qwen2.5-1.5B-Instruct").backend, "hf_causal")

    def test_workspace_readiness_report_flags_missing_optional_benchmarks(self) -> None:
        report = workspace_readiness_report(Path("."))
        self.assertTrue(report["core_source_ready"])
        self.assertTrue(report["required_data_ready"])
        self.assertIn("benchmark_inputs", report)
        self.assertIn("mquake", report["benchmark_inputs"])

    def test_benchmark_input_status_reports_downloaded_inputs(self) -> None:
        status = benchmark_input_status(Path("."))
        self.assertIn("mquake", status)
        self.assertTrue(status["mquake"]["present"])
        self.assertTrue(status["knowedit"]["present"])
        self.assertTrue(status["uniedit"]["present"])

    def test_deep_model_runtime_probe_handles_frozen_backend(self) -> None:
        report = deep_model_runtime_probe(["frozen-parametric"], timeout_seconds=1)
        self.assertTrue(report["all_healthy"])
        self.assertEqual(report["probes"][0]["backend"], "frozen")

    def test_full_public_freshqa_export_supports_zero_max_questions(self) -> None:
        snapshots = sorted(Path("data").glob("freshqa_*.csv"))
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "freshqa_full.jsonl"
            manifest = build_public_freshqa_export(
                snapshots,
                output,
                min_snapshots_per_question=3,
                max_questions=0,
                test_split_only=True,
                slice_mode="main",
            )
            self.assertTrue(output.exists())
            self.assertGreater(manifest["episodes"], 192)

    def test_mquake_builder_emits_update_single_and_multi_hop_episodes(self) -> None:
        episodes, corpus_docs, manifest = build_mquake_benchmark(limit=2)
        self.assertEqual(manifest["cases"], 2)
        roles = {episode.metadata.get("probe_role") for episode in episodes}
        self.assertIn("update", roles)
        self.assertIn("single_hop", roles)
        self.assertIn("multi_hop", roles)
        self.assertTrue(corpus_docs)

    def test_knowedit_builder_parses_selected_subsets(self) -> None:
        episodes, corpus_docs, manifest = build_knowedit_benchmark(sample_per_subset=2)
        self.assertEqual(set(manifest["subsets"]), {"WikiBio", "ZsRE", "wiki_counterfact", "wiki_recent"})
        roles = {episode.metadata.get("probe_role") for episode in episodes}
        self.assertIn("update", roles)
        self.assertIn("locality", roles)
        self.assertTrue(corpus_docs)

    def test_uniedit_builder_parses_selected_domains(self) -> None:
        episodes, corpus_docs, manifest = build_uniedit_benchmark(domains=["physics", "history"], sample_per_domain=1)
        self.assertEqual(set(manifest["domains"]), {"physics", "history"})
        roles = {episode.metadata.get("probe_role") for episode in episodes}
        self.assertIn("update", roles)
        self.assertIn("generality", roles)
        self.assertIn("locality", roles)
        self.assertTrue(corpus_docs)

    def test_mello_like_prefers_write_then_read_on_mquake(self) -> None:
        episodes, _docs, _manifest = build_mquake_benchmark(limit=1)
        runner = ExperimentRunner(
            RunnerConfig(
                baseline_mode="mello_like",
                benchmark_name="mquake",
                model_name="frozen-parametric",
                benchmark_limit=1,
            )
        )
        result = runner.run(episodes)
        actions = {trace["action"] for trace in result["traces"]}
        self.assertIn(ActionType.WRITE_MEMORY.value, actions)
        self.assertIn(ActionType.READ_MEMORY.value, actions)

    def test_manifest_campaign_smoke_exports_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {
                "benchmark": "mquake",
                "model_names": ["frozen-parametric"],
                "seeds": [0],
                "modes": ["param_only", "always_retrieve", "router_no_v_future"],
                "output_dir": tmpdir,
                "benchmark_limit": 2,
            }
            result = run_manifest_campaign(manifest)
            self.assertTrue(Path(result["bootstrap"]).exists())
            self.assertTrue(Path(result["effect_sizes"]).exists())

    def test_journal_aggregation_and_export(self) -> None:
        run_rows = [
            aggregate_journal_rows.__globals__["_row_from_result_row"](
                "freshqa_public",
                "frozen-parametric",
                0,
                ResultRow(
                    mode="router",
                    answer_quality=0.5,
                    recurring_quality=0.6,
                    non_recurring_quality=0.4,
                    latency=0.3,
                    retrieval_calls=2,
                    memory_reads=1,
                    memory_writes=1,
                    adaptation_count=1,
                    recurring_retrieval_calls=0.4,
                    recurring_memory_reads=0.5,
                    recurring_memory_writes=0.3,
                    action_distribution={"router": 1},
                    extra_metrics={"consolidation_count": 1.0, "rollback_count": 0.0, "forgetting_delta": -0.1, "average_reward": 0.2},
                ),
            ),
            aggregate_journal_rows.__globals__["_row_from_result_row"](
                "freshqa_public",
                "frozen-parametric",
                1,
                ResultRow(
                    mode="router",
                    answer_quality=0.7,
                    recurring_quality=0.8,
                    non_recurring_quality=0.6,
                    latency=0.5,
                    retrieval_calls=3,
                    memory_reads=2,
                    memory_writes=1,
                    adaptation_count=2,
                    recurring_retrieval_calls=0.3,
                    recurring_memory_reads=0.6,
                    recurring_memory_writes=0.4,
                    action_distribution={"router": 1},
                    extra_metrics={"consolidation_count": 2.0, "rollback_count": 1.0, "forgetting_delta": -0.2, "average_reward": 0.4},
                ),
            ),
        ]
        aggregate_rows = aggregate_journal_rows(run_rows)
        self.assertEqual(len(aggregate_rows), 1)
        self.assertEqual(aggregate_rows[0].answer_quality.count, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_journal_bundle(Path(tmpdir), {"benchmark": "freshqa_public"}, run_rows, aggregate_rows)
            self.assertTrue(Path(paths["aggregate_table"]).exists())
            self.assertTrue(Path(paths["frontier_table"]).exists())
            self.assertTrue(Path(paths["stale_answer_table"]).exists())
            self.assertTrue(Path(paths["rollback_report"]).exists())
            self.assertTrue(Path(paths["forgetting_report"]).exists())

    def test_run_journal_matrix_uses_models_and_seeds(self) -> None:
        manifest, run_rows, aggregate_rows = run_journal_matrix(
            base_config=RunnerConfig(
                benchmark_name="freshqa_public",
                model_name="frozen-parametric",
                public_freshness_path="data/freshqa_public_sample.jsonl",
                use_v_future=True,
            ),
            benchmark_name="freshqa_public",
            model_names=["frozen-parametric"],
            seeds=[0, 1],
            include_ablations=False,
        )
        self.assertEqual(manifest["models"], ["frozen-parametric"])
        self.assertEqual(manifest["seeds"], [0, 1])
        self.assertTrue(any(row.mode == "router" for row in run_rows))
        self.assertTrue(any(row.mode == "router" for row in aggregate_rows))

    def test_freshqa_future_scale_sweep_exports_rows(self) -> None:
        manifest, rows = run_freshqa_future_scale_sweep(
            base_config=RunnerConfig(
                benchmark_name="freshqa_public",
                model_name="frozen-parametric",
                public_freshness_path="data/freshqa_public_sample.jsonl",
                use_v_future=True,
            ),
            model_names=["frozen-parametric"],
            seeds=[0, 1],
            scales=[0.0, 0.5, 1.0],
        )
        self.assertEqual(manifest["future_value_scales"], [0.0, 0.5, 1.0])
        self.assertEqual({row["future_value_scale"] for row in rows}, {0.0, 0.5, 1.0})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_freshqa_future_scale_sweep(Path(tmpdir), manifest, rows)
            self.assertTrue(Path(paths["aggregate_table"]).exists())
            self.assertTrue(Path(paths["rows"]).exists())


if __name__ == "__main__":
    unittest.main()
