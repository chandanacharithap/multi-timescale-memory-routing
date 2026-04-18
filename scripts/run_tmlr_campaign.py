#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multitimescale_memory.campaign import run_future_sweep_campaign, run_manifest_campaign  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a manifest-driven TMLR benchmark campaign.")
    parser.add_argument("--manifest", required=True, help="Path to a JSON manifest.")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    campaign_type = str(manifest.get("campaign_type", "matrix"))
    if campaign_type == "freshqa_future_sweep":
        result = run_future_sweep_campaign(manifest)
    else:
        result = run_manifest_campaign(manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
