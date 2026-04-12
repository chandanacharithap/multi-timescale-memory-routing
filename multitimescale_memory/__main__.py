from __future__ import annotations

import argparse
import json

from .runner import ExperimentRunner, RunnerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-timescale memory controller MVP.")
    parser.add_argument(
        "--mode",
        default="router",
        choices=[
            "router",
            "param_only",
            "always_retrieve",
            "retrieve_gate",
            "three_way_gate",
            "memory_only",
            "fast_adapt_only",
            "oracle",
            "coverage_probe",
        ],
        help="Policy or baseline to run.",
    )
    parser.add_argument("--trace-path", default=None, help="Optional JSONL trace output path.")
    parser.add_argument(
        "--benchmark",
        default="demo",
        choices=["demo", "coverage"],
        help="Benchmark fixture to run.",
    )
    args = parser.parse_args()
    runner = ExperimentRunner(
        RunnerConfig(
            trace_path=args.trace_path,
            baseline_mode=args.mode,
            benchmark_name=args.benchmark,
        )
    )
    result = runner.run()
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps(result["actions"], indent=2, sort_keys=True))
    print(json.dumps(result["analyses"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
