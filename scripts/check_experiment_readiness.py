#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multitimescale_memory.readiness import DEFAULT_TARGET_MODELS, deep_model_runtime_probe, workspace_readiness_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether this copied workspace is ready for the next TMLR experiment phase.")
    parser.add_argument("--root", default=str(ROOT), help="Repository root to inspect.")
    parser.add_argument("--models", default=",".join(DEFAULT_TARGET_MODELS), help="Comma-separated model names to validate.")
    parser.add_argument("--run-tests", action="store_true", help="Also run the unit test suite with the current Python.")
    parser.add_argument(
        "--deep-runtime-probe",
        action="store_true",
        help="Also run a timeout-bounded import/load probe for the active Hugging Face backends.",
    )
    parser.add_argument(
        "--probe-timeout-seconds",
        type=int,
        default=20,
        help="Timeout for the deep runtime probe.",
    )
    args = parser.parse_args()

    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    report = workspace_readiness_report(args.root, model_names=model_names)
    if args.deep_runtime_probe:
        report["deep_model_runtime"] = deep_model_runtime_probe(
            model_names,
            timeout_seconds=args.probe_timeout_seconds,
            python_executable=sys.executable,
        )
    if args.run_tests:
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
        completed = subprocess.run(command, cwd=args.root, capture_output=True, text=True, check=False)
        report["tests"] = {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "stdout_tail": "\n".join(completed.stdout.strip().splitlines()[-20:]),
            "stderr_tail": "\n".join(completed.stderr.strip().splitlines()[-20:]),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
