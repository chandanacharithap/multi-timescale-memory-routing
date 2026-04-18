#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".md",
    ".tex",
    ".json",
    ".jsonl",
    ".txt",
    ".bib",
    ".svg",
    ".yml",
    ".yaml",
}

BANNED_PATTERNS = {
    "frozen-parametric": re.compile(r"frozen-parametric", re.IGNORECASE),
    "smoke": re.compile(r"smoke", re.IGNORECASE),
    "scaffold": re.compile(r"scaffold", re.IGNORECASE),
    "demo_fixture": re.compile(r"\bdemo\b", re.IGNORECASE),
    "coverage_probe": re.compile(r"coverage_probe", re.IGNORECASE),
    "self_rag_like": re.compile(r"self_rag_like"),
    "memllm_like": re.compile(r"memllm_like"),
    "wise_like": re.compile(r"wise_like"),
    "melo_like": re.compile(r"melo_like"),
    "mello_like": re.compile(r"mello_like"),
}


def _iter_text_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in TEXT_SUFFIXES or candidate.name in {"README", "README.md"}:
            files.append(candidate)
    return files


def _scan_file(path: Path) -> list[dict[str, object]]:
    if path.name == "paper_run_manifest_index.json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        findings: list[dict[str, object]] = []
        entries = list(payload.get("reportable_artifacts", [])) + list(payload.get("tmlr_official_reportable", [])) + list(payload.get("legacy_reportable_support", []))
        for entry in entries:
            artifact_root = str(entry.get("artifact_root", "") or "")
            for name, pattern in BANNED_PATTERNS.items():
                if pattern.search(artifact_root):
                    findings.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "pattern": name,
                            "line": 0,
                            "snippet": artifact_root,
                        }
                    )
        return findings
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    findings: list[dict[str, object]] = []
    for name, pattern in BANNED_PATTERNS.items():
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            findings.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "pattern": name,
                    "line": line_no,
                    "snippet": text.splitlines()[line_no - 1][:240] if text.splitlines() else "",
                }
            )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed audit for submission-facing files.")
    parser.add_argument("--path", action="append", default=[], help="Relative path to scan. May be passed multiple times.")
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    args = parser.parse_args()

    paths = [ROOT / item for item in args.path]
    if not paths:
        raise SystemExit("No --path values provided.")

    findings: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        for file_path in _iter_text_files(path):
            findings.extend(_scan_file(file_path))

    report = {
        "root": str(ROOT),
        "paths": args.path,
        "passed": not findings,
        "finding_count": len(findings),
        "findings": findings,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if findings:
        sys.exit(1)


if __name__ == "__main__":
    main()
