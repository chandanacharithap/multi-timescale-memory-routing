from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multitimescale_memory.freshqa_export import build_public_freshqa_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a public FreshQA JSONL export from weekly CSV snapshots.")
    parser.add_argument("snapshots", nargs="+", help="Ordered or unordered FreshQA CSV snapshot paths.")
    parser.add_argument("--output", required=True, help="Output JSONL path for the derived FreshQA benchmark export.")
    parser.add_argument("--min-snapshots-per-question", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=24, help="Use 0 to include all qualifying changed TEST questions.")
    parser.add_argument("--include-non-test", action="store_true", help="Include non-TEST rows from the public snapshots.")
    parser.add_argument("--slice-mode", choices=["main", "sequence"], default="main")
    args = parser.parse_args()
    manifest = build_public_freshqa_export(
        args.snapshots,
        args.output,
        min_snapshots_per_question=args.min_snapshots_per_question,
        max_questions=args.max_questions,
        test_split_only=not args.include_non_test,
        slice_mode=args.slice_mode,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
