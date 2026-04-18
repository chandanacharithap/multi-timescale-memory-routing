from __future__ import annotations

import argparse
import json
from pathlib import Path

from .journal import (
    export_freshqa_future_scale_sweep,
    export_journal_bundle,
    run_freshqa_future_scale_sweep,
    run_journal_matrix,
)
from .modeling import assert_models_available
from .public_freshness import validate_public_freshness_rows
from .runner import ExperimentRunner, RunnerConfig, format_freshness_result_table, format_result_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-timescale memory controller MVP.")
    parser.add_argument(
        "--mode",
        default="router",
        choices=[
            "router",
            "router_calibrated",
            "param_only",
            "always_retrieve",
            "retrieve_gate",
            "three_way_gate",
            "memory_only",
            "fast_adapt_only",
            "self_rag_like",
            "memllm_like",
            "wise_like",
            "melo_like",
            "mello_like",
            "oracle",
            "coverage_probe",
        ],
        help="Policy or baseline to run.",
    )
    parser.add_argument("--trace-path", default=None, help="Optional JSONL trace output path.")
    parser.add_argument(
        "--benchmark",
        default="popqa",
        choices=["demo", "coverage", "popqa", "freshness", "freshqa_public", "mquake", "knowedit", "uniedit"],
        help="Benchmark to run. The demo and coverage fixtures are internal debug-only checks and are not submission-facing.",
    )
    parser.add_argument("--model-name", default="google/flan-t5-small", help="Open model name for real runs.")
    parser.add_argument("--cache-dir", default=".cache", help="Cache directory for benchmark assets and memory.")
    parser.add_argument("--popqa-limit", type=int, default=8, help="Number of PopQA rows to load for the pilot.")
    parser.add_argument("--freshness-limit", type=int, default=None, help="Optional number of freshness episodes to load.")
    parser.add_argument("--benchmark-limit", type=int, default=None, help="Optional benchmark-specific sample size for MQuAKE, KnowEdit, or UniEdit.")
    parser.add_argument("--benchmark-sample-seed", type=int, default=2026, help="Sampling seed for benchmark subsets such as KnowEdit or UniEdit.")
    parser.add_argument("--freshness-data-path", default=None, help="Path to a real public freshness benchmark JSONL export.")
    parser.add_argument("--sequence-repeats", type=int, default=1, help="Repeat update sequences to create longer sequential-edit stress tests.")
    parser.add_argument("--router-seed", type=int, default=0, help="Seed for journal-style stochastic router initialization.")
    parser.add_argument("--stochastic-router", action="store_true", help="Enable seeded stochastic router initialization for robustness runs.")
    parser.add_argument("--model-names", default=None, help="Comma-separated model names for journal matrix runs.")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds for journal matrix runs.")
    parser.add_argument(
        "--prefetch-popqa",
        action="store_true",
        help="Prefetch and cache the PopQA subset and evidence locally before running experiments.",
    )
    parser.add_argument(
        "--popqa-candidate-limit",
        type=int,
        default=None,
        help="Optional candidate row budget when prefetching a recurring-heavy PopQA subset.",
    )
    parser.add_argument(
        "--popqa-cached-only",
        action="store_true",
        help="Build the PopQA subset using only locally cached evidence pages, with no network fetches.",
    )
    parser.add_argument("--suite", action="store_true", help="Run the four-baseline suite and print a markdown result table.")
    parser.add_argument("--compare-v-future", action="store_true", help="Compare router with and without V_future.")
    parser.add_argument(
        "--compare-memory-ablations",
        action="store_true",
        help="Compare the full router against no READ_MEMORY, no WRITE_MEMORY, and no V_future.",
    )
    parser.add_argument(
        "--export-popqa-bundle",
        action="store_true",
        help="Export PopQA tables, figures, traces, and a 50-example error audit for the current cached setup.",
    )
    parser.add_argument(
        "--export-freshness-bundle",
        action="store_true",
        help="Export freshness tables, figures, traces, and a 50-example error audit for the current benchmark run.",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional output directory for exported PopQA bundle artifacts.",
    )
    parser.add_argument(
        "--audit-limit",
        type=int,
        default=50,
        help="Number of labeled examples to include in the PopQA error audit export.",
    )
    parser.add_argument(
        "--popqa-guardrail-mode",
        default="strict",
        choices=["legacy", "strict"],
        help="Use legacy PopQA masks for the frozen baseline or strict masks for the guarded rerun.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional label for exported PopQA bundles, for example pre_fix or post_fix.",
    )
    parser.add_argument(
        "--export-journal-bundle",
        action="store_true",
        help="Run a multi-model, multi-seed journal matrix and export aggregate bundle artifacts.",
    )
    parser.add_argument(
        "--validate-public-freshness",
        action="store_true",
        help="Validate a real public freshness JSONL export and print the validation report.",
    )
    parser.add_argument(
        "--check-model-loads",
        action="store_true",
        help="Smoke-check that the requested model or model matrix can load before starting large runs.",
    )
    parser.add_argument(
        "--future-value-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to delayed utility during routing and reward shaping.",
    )
    parser.add_argument(
        "--export-freshqa-future-sweep",
        action="store_true",
        help="Run a public FreshQA V_future calibration sweep for the router and export aggregate results.",
    )
    parser.add_argument(
        "--future-value-scales",
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated delayed-utility scales for the FreshQA calibration sweep.",
    )
    args = parser.parse_args()
    runner = ExperimentRunner(
        RunnerConfig(
            trace_path=args.trace_path,
            baseline_mode=args.mode,
            benchmark_name=args.benchmark,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            popqa_limit=args.popqa_limit,
            use_v_future=True,
            allow_popqa_network=False,
            popqa_cached_only=args.popqa_cached_only,
            popqa_guardrail_mode=args.popqa_guardrail_mode,
            freshness_limit=args.freshness_limit,
            benchmark_limit=args.benchmark_limit,
            benchmark_sample_seed=args.benchmark_sample_seed,
            router_seed=args.router_seed,
            stochastic_router=args.stochastic_router,
            public_freshness_path=args.freshness_data_path,
            sequence_repeats=args.sequence_repeats,
            future_value_scale=args.future_value_scale,
        )
    )
    if args.prefetch_popqa:
        result = runner.prefetch_popqa(candidate_limit=args.popqa_candidate_limit)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.export_popqa_bundle:
        export_dir = args.export_dir or f"artifacts/popqa_{args.popqa_limit}"
        result = runner.export_popqa_bundle(
            output_dir=export_dir,
            audit_limit=args.audit_limit,
            run_label=args.run_label or args.popqa_guardrail_mode,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.export_freshness_bundle:
        export_dir = args.export_dir or "artifacts/freshness_v1"
        result = runner.export_freshness_bundle(
            output_dir=export_dir,
            audit_limit=args.audit_limit,
            run_label=args.run_label or "freshness_v1",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.validate_public_freshness:
        if not args.freshness_data_path:
            raise ValueError("--validate-public-freshness requires --freshness-data-path")
        rows = []
        with Path(args.freshness_data_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        print(json.dumps(validate_public_freshness_rows(rows), indent=2, sort_keys=True))
        return
    if args.check_model_loads:
        model_names = [item.strip() for item in (args.model_names or args.model_name).split(",") if item.strip()]
        statuses = assert_models_available(model_names, attempt_load=True)
        print(json.dumps(statuses, indent=2, sort_keys=True))
        return
    if args.export_journal_bundle:
        output_dir = args.export_dir or f"artifacts/journal_{args.benchmark}"
        model_names = [item.strip() for item in (args.model_names or args.model_name).split(",") if item.strip()]
        seeds = [int(item.strip()) for item in (args.seeds or str(args.router_seed)).split(",") if item.strip()]
        assert_models_available(model_names, attempt_load=False)
        manifest, run_rows, aggregate_rows = run_journal_matrix(
            base_config=runner.config,
            benchmark_name=args.benchmark,
            model_names=model_names,
            seeds=seeds,
            include_ablations=True,
        )
        result = export_journal_bundle(Path(output_dir), manifest, run_rows, aggregate_rows)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.export_freshqa_future_sweep:
        if args.benchmark != "freshqa_public":
            raise ValueError("--export-freshqa-future-sweep requires --benchmark freshqa_public")
        output_dir = args.export_dir or "artifacts/journal_freshqa_future_scale_sweep"
        model_names = [item.strip() for item in (args.model_names or args.model_name).split(",") if item.strip()]
        seeds = [int(item.strip()) for item in (args.seeds or str(args.router_seed)).split(",") if item.strip()]
        scales = [float(item.strip()) for item in args.future_value_scales.split(",") if item.strip()]
        assert_models_available(model_names, attempt_load=False)
        manifest, rows = run_freshqa_future_scale_sweep(
            base_config=runner.config,
            model_names=model_names,
            seeds=seeds,
            scales=scales,
        )
        result = export_freshqa_future_scale_sweep(Path(output_dir), manifest, rows)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.suite:
        rows = runner.run_baseline_suite()
        if args.benchmark == "freshness":
            print(format_freshness_result_table(rows))
        else:
            print(format_result_table(rows))
        return
    if args.compare_v_future:
        rows = runner.run_router_ablation_suite()
        rows = [row for row in rows if row.mode in {"router_full", "router_no_v_future"}]
        if args.benchmark == "freshness":
            print(format_freshness_result_table(rows))
        else:
            print(format_result_table(rows))
        return
    if args.compare_memory_ablations:
        rows = runner.run_router_ablation_suite()
        if args.benchmark == "freshness":
            print(format_freshness_result_table(rows))
        else:
            print(format_result_table(rows))
        return
    result = runner.run()
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps(result["actions"], indent=2, sort_keys=True))
    print(json.dumps(result["analyses"], indent=2, sort_keys=True))
    print(json.dumps(result["subsets"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
