# Claims To Artifacts Map

Only artifact roots listed in `paper_run_manifest_index.json` are paper-facing and reportable. The TMLR paper uses only `tmlr_official_reportable`. The arXiv paper uses the same official surface plus one clearly labeled bundled controlled-update artifact.

## TMLR official claims

### PopQA recurring-memory

- Claim: delayed utility is useful in the stable recurring-memory regime because the router matches retrieval quality while reducing retrieval calls.
  Source: `artifacts/tmlr_official/popqa_recurring_flan_base_v1/aggregate_table.md`
- Claim: the router and retrieval are statistically tied on PopQA, while suppressing delayed utility is significantly worse.
  Source: `artifacts/tmlr_official/popqa_recurring_flan_base_v1/significance.json`

### FreshQA public main and sequence

- Claim: public changing-answer and correction-chain regimes are retrieval-dominant across the official model matrix.
  Source: `artifacts/tmlr_official/freshqa_public_main_flan_small_v1/aggregate_table.md`, `artifacts/tmlr_official/freshqa_public_main_flan_base_v1/aggregate_table.md`, `artifacts/tmlr_official/freshqa_public_main_qwen_1p5b_v1/aggregate_table.md`, `artifacts/tmlr_official/freshqa_public_main_smollm2_360m_v1/aggregate_table.md`, `artifacts/tmlr_official/freshqa_public_sequence_flan_small_v1/aggregate_table.md`, `artifacts/tmlr_official/freshqa_public_sequence_flan_base_v1/aggregate_table.md`, `artifacts/tmlr_official/freshqa_public_sequence_qwen_1p5b_v1/aggregate_table.md`, `artifacts/tmlr_official/freshqa_public_sequence_smollm2_360m_v1/aggregate_table.md`
- Claim: calibration partially recovers the public regime, but suppressing delayed utility is more reliable than soft calibration.
  Source: `artifacts/tmlr_official/freshqa_public_main_flan_small_v1/significance.json`, `artifacts/tmlr_official/freshqa_public_main_flan_base_v1/significance.json`, `artifacts/tmlr_official/freshqa_public_sequence_flan_small_v1/significance.json`, `artifacts/tmlr_official/freshqa_public_sequence_flan_base_v1/significance.json`
- Claim: the public benchmark trust story is bounded rather than fully cleared.
  Source: `artifacts/tmlr_official/freshqa_leakage_audit_manual_v1/summary.json`

### MQuAKE

- Claim: propagation-heavy editing is retrieval-dominant in the current setup, and suppressing delayed utility improves over the full router without overtaking retrieval.
  Source: `artifacts/tmlr_official/mquake_flan_base_v1/aggregate_table.md`, `artifacts/tmlr_official/mquake_flan_small_v1/aggregate_table.md`, `artifacts/tmlr_official/mquake_flan_base_v1/benchmark_metric_aggregates.json`, `artifacts/tmlr_official/mquake_flan_small_v1/benchmark_metric_aggregates.json`
- Claim: the MQuAKE parser and snapshot are frozen and auditable.
  Source: `artifacts/tmlr_official/mquake_reliability_note_v1/reliability_note.json`

### UniEdit

- Claim: UniEdit is a near-ceiling sanity check rather than a headline router win.
  Source: `artifacts/tmlr_official/uniedit_official_v1/aggregate_table.md`, `artifacts/tmlr_official/uniedit_official_v1/benchmark_metric_aggregates.json`, `artifacts/tmlr_official/uniedit_official_v1/significance.json`
- Claim: the UniEdit sample and field mapping are frozen and auditable.
  Source: `artifacts/tmlr_official/uniedit_reliability_note_v1/reliability_note.json`

### Cross-benchmark law and statistics

- Claim: the paper's empirical law is derived from the frozen official benchmark surface only.
  Source: `artifacts/tmlr_official/official_statistics_v1/cross_benchmark_summary.md`, `artifacts/tmlr_official/official_statistics_v1/bootstrap_all.json`, `artifacts/tmlr_official/official_statistics_v1/effect_sizes_all.json`, `artifacts/tmlr_official/official_statistics_v1/significance_all.json`

## arXiv-only extra context

- Claim: on the bundled controlled-update artifact, the six-action router improves quality and stale-answer rate over simpler baselines while exercising temporary adaptation, guarded consolidation, and rollback.
  Source: `artifacts/freshness_v1/baseline_comparison_table.md`, `artifacts/freshness_v1/router_summary.json`, `artifacts/freshness_v1/rollback_report.md`
- Claim: on the bundled controlled-update artifact, removing delayed utility or temporary adaptation weakens the positive result.
  Source: `artifacts/freshness_v1/ablation_table.md`
- Claim: remaining bundled controlled-update failures concentrate in volatile updates.
  Source: `artifacts/freshness_v1/error_audit_summary.json`, `artifacts/freshness_v1/error_audit_50.md`
