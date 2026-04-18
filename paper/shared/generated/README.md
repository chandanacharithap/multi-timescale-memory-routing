# Generated Paper Assets

This directory contains paper-facing tables and figures derived from frozen artifact bundles allowed by `paper_run_manifest_index.json`.

## Authoritative sources

- `artifacts/tmlr_official/popqa_recurring_flan_base_v1/aggregate_table.md`
- `artifacts/tmlr_official/freshqa_public_main_flan_small_v1/aggregate_table.md`
- `artifacts/tmlr_official/freshqa_public_main_flan_base_v1/aggregate_table.md`
- `artifacts/tmlr_official/freshqa_public_sequence_flan_small_v1/aggregate_table.md`
- `artifacts/tmlr_official/freshqa_public_sequence_flan_base_v1/aggregate_table.md`
- `artifacts/tmlr_official/freshqa_public_main_qwen_1p5b_v1/aggregate_table.md`
- `artifacts/tmlr_official/freshqa_public_sequence_qwen_1p5b_v1/aggregate_table.md`
- `artifacts/tmlr_official/freshqa_public_main_smollm2_360m_v1/aggregate_table.md`
- `artifacts/tmlr_official/freshqa_public_sequence_smollm2_360m_v1/aggregate_table.md`
- `artifacts/tmlr_official/mquake_flan_base_v1/aggregate_table.md`
- `artifacts/tmlr_official/mquake_flan_small_v1/aggregate_table.md`
- `artifacts/tmlr_official/mquake_flan_base_v1/benchmark_metric_aggregates.json`
- `artifacts/tmlr_official/mquake_flan_small_v1/benchmark_metric_aggregates.json`
- `artifacts/tmlr_official/uniedit_official_v1/aggregate_table.md`
- `artifacts/tmlr_official/uniedit_official_v1/benchmark_metric_aggregates.json`
- `artifacts/tmlr_official/freshqa_leakage_audit_manual_v1/summary.json`
- `artifacts/tmlr_official/official_statistics_v1/cross_benchmark_summary.md`
- `artifacts/tmlr_official/official_statistics_v1/significance_all.json`
- `artifacts/freshness_v1/baseline_comparison_table.md` for the named arXiv-only bundled controlled-update context

## Generated files

- `tab_benchmark_map_arxiv.tex`
- `tab_benchmark_map_tmlr.tex`
- `tab_cross_benchmark_summary_official.tex`
- `tab_popqa_official.tex`
- `tab_freshqa_main_flan_official.tex`
- `tab_freshqa_sequence_flan_official.tex`
- `tab_freshqa_cross_family_official.tex`
- `tab_public_audit_official.tex`
- `tab_mquake_official.tex`
- `tab_uniedit_official.tex`
- `tab_freshness_bundled_official.tex`
- `popqa_retrieval_calls_reduction.svg`
- `fig_system_overview.svg`
- `fig_public_main_calibration_curve.svg`
- `word_assets/popqa_retrieval_calls_reduction.png`
- `word_assets/fig_system_overview.png`
- `word_assets/fig_public_main_calibration_curve.png`

These assets are intentionally copied from frozen results instead of live reruns. Debug, provisional, synthetic, and proxy-only bundles are excluded from cited paper generation.

## Legacy assets retained but not cited by the final papers

The directory still contains older generated files from earlier drafts. They are preserved for traceability but are not cited by the rewritten arXiv or TMLR manuscripts. Reader-facing assets now use the official `tab_*_official.tex`, `tab_benchmark_map_*`, `tab_cross_benchmark_summary_official.tex`, and `fig_*` files listed above.
