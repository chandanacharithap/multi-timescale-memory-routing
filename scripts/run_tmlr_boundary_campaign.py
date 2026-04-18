#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multitimescale_memory.journal import export_journal_bundle, run_selected_modes_matrix  # noqa: E402
from multitimescale_memory.runner import RunnerConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the calibrated-router boundary campaign on selected benchmarks and modes.")
    parser.add_argument("--benchmark", required=True, choices=["popqa", "freshqa_public"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-names", required=True, help="Comma-separated model names.")
    parser.add_argument("--seeds", required=True, help="Comma-separated integer seeds.")
    parser.add_argument("--modes", default="always_retrieve,router,router_calibrated", help="Comma-separated baseline modes.")
    parser.add_argument("--freshness-data-path", default=None)
    parser.add_argument("--sequence-repeats", type=int, default=1)
    parser.add_argument("--popqa-limit", type=int, default=100)
    parser.add_argument("--future-value-scale", type=float, default=1.0)
    parser.add_argument("--include-ablations", action="store_true")
    args = parser.parse_args()

    config = RunnerConfig(
        benchmark_name=args.benchmark,
        model_name=args.model_names.split(",")[0].strip(),
        public_freshness_path=args.freshness_data_path,
        sequence_repeats=args.sequence_repeats,
        popqa_limit=args.popqa_limit,
        future_value_scale=args.future_value_scale,
    )
    manifest, run_rows, aggregate_rows = run_selected_modes_matrix(
        base_config=config,
        benchmark_name=args.benchmark,
        model_names=[item.strip() for item in args.model_names.split(",") if item.strip()],
        seeds=[int(item.strip()) for item in args.seeds.split(",") if item.strip()],
        modes=[item.strip() for item in args.modes.split(",") if item.strip()],
        include_ablations=args.include_ablations,
    )
    result = export_journal_bundle(Path(args.output_dir), manifest, run_rows, aggregate_rows)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
