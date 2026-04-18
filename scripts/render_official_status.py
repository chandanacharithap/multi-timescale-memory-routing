#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _official_rows(registry: dict) -> list[dict]:
    if "tmlr_official_reportable" in registry:
        return list(registry["tmlr_official_reportable"])
    return [
        row
        for row in registry.get("reportable_artifacts", [])
        if str(row.get("artifact_root", "")).startswith("artifacts/tmlr_official/")
    ]


def main() -> None:
    registry = json.loads((ROOT / "paper_run_manifest_index.json").read_text(encoding="utf-8"))
    reportable = _official_rows(registry)
    pending = registry.get("pending_required_runs", [])

    official_real = [
        row for row in reportable
        if str(row.get("artifact_root", "")).startswith("artifacts/tmlr_official/")
        and row.get("status") == "frozen_real_run"
    ]
    official_support = [
        row for row in reportable
        if str(row.get("artifact_root", "")).startswith("artifacts/tmlr_official/")
        and row.get("status") == "frozen_support_bundle"
    ]

    lines = [
        "# TMLR Official Status",
        "",
        "## Frozen real-model bundles now available",
        "",
    ]
    for row in official_real:
        lines.append(f"- `{row['run_id']}`:")
        lines.append(f"  - `{row['artifact_root']}/`")
    lines.extend(["", "## Frozen official support bundles now available", ""])
    for row in official_support:
        lines.append(f"- `{row['run_id']}`:")
        lines.append(f"  - `{row['artifact_root']}/`")
    lines.extend(["", "## Not yet frozen in the official namespace", ""])
    for row in pending:
        lines.append(f"- `{row['run_id']}`")
    lines.append("")
    (ROOT / "artifacts" / "tmlr_official" / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
