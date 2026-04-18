from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .modeling import model_runtime_status, resolve_model_spec


DEFAULT_TARGET_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
]

CORE_SOURCE_PATHS = [
    "multitimescale_memory/runner.py",
    "multitimescale_memory/journal.py",
    "multitimescale_memory/modeling.py",
    "multitimescale_memory/freshqa_export.py",
    "tests/test_mvp.py",
]

REQUIRED_DATA_PATHS = [
    "data/freshqa_public_main_v2.jsonl",
    "data/freshqa_public_sequence_v2.jsonl",
]

JOURNAL_ARTIFACT_PATHS = [
    "artifacts/journal_freshqa_main_calibrated_v1/manifest.json",
    "artifacts/journal_freshqa_sequence_calibrated_v1/manifest.json",
    "artifacts/journal_popqa_broad_100_calibrated_v1/manifest.json",
    "artifacts/public_freshqa_provenance_v1/manifest.json",
]

BENCHMARK_INPUT_GROUPS = {
    "mquake": [
        "data/benchmarks/mquake/MQuAKE-CF.json",
        "data/benchmarks/mquake/MQuAKE-CF-3k.json",
        "data/benchmarks/mquake/MQuAKE-CF-3k-v2.json",
        "data/benchmarks/mquake/MQuAKE-T.json",
    ],
    "knowedit": [
        "data/benchmarks/knowedit/README.md",
        "data/benchmarks/knowedit/WikiBio",
        "data/benchmarks/knowedit/ZsRE",
        "data/benchmarks/knowedit/wiki_counterfact",
        "data/benchmarks/knowedit/wiki_recent",
    ],
    "uniedit": [
        "data/benchmarks/uniedit/README.md",
        "data/benchmarks/uniedit/test",
        "data/benchmarks/uniedit/train",
        "data/benchmarks/uniedit/manifest.json",
    ],
}


def _readable_path_report(root: Path, relative_path: str) -> dict[str, object]:
    path = root / relative_path
    exists = path.exists()
    readable = False
    kind = "missing"
    size_bytes = 0
    if exists:
        kind = "dir" if path.is_dir() else "file"
        try:
            if path.is_dir():
                next(path.iterdir(), None)
            else:
                with path.open("rb") as handle:
                    handle.read(1)
            readable = True
        except OSError:
            readable = False
        if path.is_file():
            size_bytes = path.stat().st_size
    return {
        "path": relative_path,
        "exists": exists,
        "readable": readable,
        "kind": kind,
        "size_bytes": size_bytes,
    }


def benchmark_input_status(root: str | Path) -> dict[str, dict[str, object]]:
    root_path = Path(root)
    result: dict[str, dict[str, object]] = {}
    for name, entries in BENCHMARK_INPUT_GROUPS.items():
        reports = [_readable_path_report(root_path, item) for item in entries]
        result[name] = {
            "present": all(item["exists"] and item["readable"] for item in reports),
            "paths": reports,
        }
    return result


def workspace_readiness_report(
    root: str | Path,
    *,
    model_names: list[str] | None = None,
) -> dict[str, object]:
    root_path = Path(root)
    requested_models = model_names or list(DEFAULT_TARGET_MODELS)
    core_reports = [_readable_path_report(root_path, item) for item in CORE_SOURCE_PATHS]
    data_reports = [_readable_path_report(root_path, item) for item in REQUIRED_DATA_PATHS]
    artifact_reports = [_readable_path_report(root_path, item) for item in JOURNAL_ARTIFACT_PATHS]
    benchmark_reports = benchmark_input_status(root_path)
    runtime_statuses = [model_runtime_status(model_name) for model_name in requested_models]
    return {
        "cwd": str(root_path.resolve()),
        "python": sys.executable,
        "core_source_ready": all(item["exists"] and item["readable"] for item in core_reports),
        "required_data_ready": all(item["exists"] and item["readable"] for item in data_reports),
        "journal_artifacts_ready": all(item["exists"] and item["readable"] for item in artifact_reports),
        "core_source": core_reports,
        "required_data": data_reports,
        "journal_artifacts": artifact_reports,
        "benchmark_inputs": benchmark_reports,
        "model_runtime": {
            "target_models": requested_models,
            "all_loadable": all(bool(item["loadable"]) for item in runtime_statuses),
            "statuses": runtime_statuses,
        },
    }


def deep_model_runtime_probe(
    model_names: list[str],
    *,
    timeout_seconds: int = 20,
    python_executable: str | None = None,
) -> dict[str, object]:
    python = python_executable or sys.executable
    backend_commands: dict[str, list[str]] = {
        "hf_seq2seq": [
            "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM",
            "AutoTokenizer.from_pretrained('google/flan-t5-small', local_files_only=True)",
            "AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small', local_files_only=True)",
            "print('ok')",
        ],
        "hf_causal": [
            "from transformers import AutoTokenizer, AutoModelForCausalLM",
            "AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-360M-Instruct', local_files_only=True)",
            "AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-360M-Instruct', local_files_only=True)",
            "print('ok')",
        ],
    }
    probes: dict[str, dict[str, object]] = {}
    backend_seen: set[str] = set()
    for model_name in model_names:
        spec = resolve_model_spec(model_name)
        if spec.backend == "frozen":
            probes[model_name] = {
                "model_name": model_name,
                "backend": spec.backend,
                "healthy": True,
                "detail": "frozen backend does not require external runtime imports",
            }
            continue
        if spec.backend in backend_seen:
            continue
        backend_seen.add(spec.backend)
        command = backend_commands.get(spec.backend)
        if command is None:
            probes[model_name] = {
                "model_name": model_name,
                "backend": spec.backend,
                "healthy": False,
                "detail": f"no deep probe command configured for backend {spec.backend}",
            }
            continue
        try:
            completed = subprocess.run(
                [python, "-c", "; ".join(command)],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
            healthy = completed.returncode == 0
            probes[model_name] = {
                "model_name": model_name,
                "backend": spec.backend,
                "healthy": healthy,
                "returncode": completed.returncode,
                "stdout_tail": "\n".join(completed.stdout.strip().splitlines()[-10:]),
                "stderr_tail": "\n".join(completed.stderr.strip().splitlines()[-10:]),
                "detail": "probe completed" if healthy else "probe failed",
            }
        except subprocess.TimeoutExpired as exc:
            probes[model_name] = {
                "model_name": model_name,
                "backend": spec.backend,
                "healthy": False,
                "detail": f"probe timed out after {timeout_seconds}s",
                "stdout_tail": "\n".join((exc.stdout or "").strip().splitlines()[-10:]),
                "stderr_tail": "\n".join((exc.stderr or "").strip().splitlines()[-10:]),
            }
    return {
        "python": python,
        "timeout_seconds": timeout_seconds,
        "all_healthy": all(bool(item["healthy"]) for item in probes.values()),
        "probes": list(probes.values()),
    }


def workspace_readiness_json(
    root: str | Path,
    *,
    model_names: list[str] | None = None,
    indent: int = 2,
) -> str:
    return json.dumps(workspace_readiness_report(root, model_names=model_names), indent=indent, sort_keys=True)
