# Journal Upgrade Reproducibility Notes

This project now includes a completed second journal evidence cycle in addition to the frozen preprint bundles.

## Canonical journal artifact roots

- `artifacts/journal_popqa_broad_100_v2/`
- `artifacts/journal_freshqa_main_v2/`
- `artifacts/journal_freshqa_sequence_v2/`
- `artifacts/journal_freshqa_future_scale_sweep_v1/`
- `artifacts/journal_popqa_broad_100_calibrated_v1/`
- `artifacts/journal_freshqa_main_calibrated_v1/`
- `artifacts/journal_freshqa_sequence_calibrated_v1/`
- `artifacts/journal_freshqa_main_cross_family_v1/`
- `artifacts/public_freshqa_provenance_v1/`
- `artifacts/public_freshqa_audit_v1/`

The earlier `journal_freshqa_main_small` and `journal_popqa_broad_100` roots should be treated as pre-fix checkpoints.

## Production runtime

Use the clean production environment for real Hugging Face runs:

```bash
bash scripts/bootstrap_production_env.sh
.venv-prod/bin/python scripts/check_experiment_readiness.py --deep-runtime-probe --probe-timeout-seconds 20
```

The default system `python3` on this machine does not carry the required model stack. Submission-facing benchmark campaigns should run from `.venv-prod/bin/python`, not from the default Python 3.13 environment.

## Public FreshQA-derived inputs

- Raw weekly public CSV snapshots:
  - `data/freshqa_2025-02-17.csv`
  - `data/freshqa_2025-02-24.csv`
  - `data/freshqa_2025-03-03.csv`
  - `data/freshqa_2025-03-10.csv`
  - `data/freshqa_2025-03-17.csv`
  - `data/freshqa_2025-03-24.csv`
  - `data/freshqa_2025-03-31.csv`
  - `data/freshqa_2025-04-07.csv`
- Derived public tracks:
  - `data/freshqa_public_main_v2.jsonl`
  - `data/freshqa_public_sequence_v2.jsonl`

The sample file `data/freshqa_public_sample.jsonl` is an internal schema/debug fixture and is not a submission-facing artifact.

## Main journal commands

Public FreshQA main v2:

```bash
.venv-journal/bin/python -m multitimescale_memory \
  --benchmark freshqa_public \
  --model-name google/flan-t5-small \
  --model-names google/flan-t5-small,google/flan-t5-base \
  --freshness-data-path data/freshqa_public_main_v2.jsonl \
  --seeds 0,1,2 \
  --export-journal-bundle \
  --export-dir artifacts/journal_freshqa_main_v2
```

Public FreshQA sequence v2:

```bash
.venv-journal/bin/python -m multitimescale_memory \
  --benchmark freshqa_public \
  --model-name google/flan-t5-small \
  --model-names google/flan-t5-small,google/flan-t5-base \
  --freshness-data-path data/freshqa_public_sequence_v2.jsonl \
  --sequence-repeats 2 \
  --seeds 0,1,2 \
  --export-journal-bundle \
  --export-dir artifacts/journal_freshqa_sequence_v2
```

PopQA broad 100 v2:

```bash
.venv-journal/bin/python -m multitimescale_memory \
  --benchmark popqa \
  --model-name google/flan-t5-small \
  --model-names google/flan-t5-small,google/flan-t5-base \
  --popqa-limit 100 \
  --popqa-cached-only \
  --seeds 0,1,2 \
  --export-journal-bundle \
  --export-dir artifacts/journal_popqa_broad_100_v2
```

Public FreshQA delayed-utility calibration sweep:

```bash
.venv-journal/bin/python -m multitimescale_memory \
  --benchmark freshqa_public \
  --model-name google/flan-t5-small \
  --model-names google/flan-t5-small,google/flan-t5-base \
  --freshness-data-path data/freshqa_public_main_v2.jsonl \
  --seeds 0,1,2 \
  --export-freshqa-future-sweep \
  --future-value-scales 0.0,0.25,0.5,0.75,1.0 \
  --export-dir artifacts/journal_freshqa_future_scale_sweep_v1
```

## Submission-facing baseline policy

- Main-table evidence must come only from runs listed in `paper_run_manifest_index.json`.
- Main tables may contain only faithful or clearly official baselines.
- Proxy-family baselines are internal or appendix-only and are excluded from submission-facing default matrices.
- The reproducibility package should include only official frozen bundles, manifests, statistics, and audits referenced by the claims map and the official run registry.

## Boundary-paper commands

Selected-mode calibrated public main bundle:

```bash
.venv-journal/bin/python scripts/run_tmlr_boundary_campaign.py \
  --benchmark freshqa_public \
  --output-dir artifacts/journal_freshqa_main_calibrated_v1 \
  --model-names google/flan-t5-small,google/flan-t5-base \
  --seeds 0,1,2 \
  --modes always_retrieve,router,router_calibrated \
  --freshness-data-path data/freshqa_public_main_v2.jsonl \
  --include-ablations
```

Selected-mode calibrated public sequence bundle:

```bash
.venv-journal/bin/python scripts/run_tmlr_boundary_campaign.py \
  --benchmark freshqa_public \
  --output-dir artifacts/journal_freshqa_sequence_calibrated_v1 \
  --model-names google/flan-t5-small,google/flan-t5-base \
  --seeds 0,1,2 \
  --modes always_retrieve,router,router_calibrated \
  --freshness-data-path data/freshqa_public_sequence_v2.jsonl \
  --sequence-repeats 2 \
  --include-ablations
```

Selected-mode PopQA regression bundle:

```bash
.venv-journal/bin/python scripts/run_tmlr_boundary_campaign.py \
  --benchmark popqa \
  --output-dir artifacts/journal_popqa_broad_100_calibrated_v1 \
  --model-names google/flan-t5-small,google/flan-t5-base \
  --seeds 0,1,2 \
  --modes always_retrieve,router,router_calibrated \
  --include-ablations \
  --popqa-limit 100
```

Cross-family public-main bundle:

```bash
.venv-journal/bin/python scripts/run_tmlr_boundary_campaign.py \
  --benchmark freshqa_public \
  --output-dir artifacts/journal_freshqa_main_cross_family_v1 \
  --model-names google/flan-t5-small,google/flan-t5-base,HuggingFaceTB/SmolLM2-360M-Instruct \
  --seeds 0,1,2 \
  --modes always_retrieve,router,router_calibrated \
  --freshness-data-path data/freshqa_public_main_v2.jsonl
```

Public provenance + audit bundle:

```bash
python3 scripts/build_tmlr_boundary_bundle.py \
  --snapshot data/freshqa_2025-02-17.csv \
  --snapshot data/freshqa_2025-02-24.csv \
  --snapshot data/freshqa_2025-03-03.csv \
  --snapshot data/freshqa_2025-03-10.csv \
  --snapshot data/freshqa_2025-03-17.csv \
  --snapshot data/freshqa_2025-03-24.csv \
  --snapshot data/freshqa_2025-03-31.csv \
  --snapshot data/freshqa_2025-04-07.csv \
  --main-jsonl data/freshqa_public_main_v2.jsonl \
  --sequence-jsonl data/freshqa_public_sequence_v2.jsonl \
  --router-trace artifacts/public_freshqa_audit_v1/router_full_trace.jsonl \
  --calibrated-trace artifacts/public_freshqa_audit_v1/router_calibrated_trace.jsonl \
  --provenance-dir artifacts/public_freshqa_provenance_v1 \
  --audit-dir artifacts/public_freshqa_audit_v1
```

## Output bundle contract

Every main journal bundle includes:

- `manifest.json`
- `run_rows.jsonl`
- `aggregate_rows.json`
- `aggregate_table.md`
- `frontier_table.md`
- `stale_answer_table.md`
- `rollback_report.md`
- `forgetting_report.md`
- `action_distribution.json`

The future-scale sweep bundle includes:

- `manifest.json`
- `run_rows.jsonl`
- `rows.json`
- `aggregate_table.md`

## Constraints and interpretation

- The frozen `PopQA` and bundled `freshness_v1` preprint bundles remain intact and are not overwritten by journal runs.
- `journal_popqa_broad_100_v2` is a positive external-validation result for recurring-memory routing.
- The public FreshQA v2 bundles are robustness studies: they show that the public track is currently retrieval-dominant and that delayed utility is the main miscalibrated component.
- The evidence-backed next tuning direction after Cycle 2 is delayed-utility calibration on public changing-answer tracks.
- The boundary-paper update shows that regime-aware delayed-utility calibration materially improves the public FreshQA-derived main slice while leaving the positive PopQA regime unchanged.
