# Paper Sources

This directory now contains two paper deliverables built from the same frozen research artifacts:

- `arxiv/main.tex`: the full self-contained preprint
- `workshop/main.tex`: the compressed 8-page workshop version

## Shared assets

- `shared/references.bib`: bibliography used by both papers
- `shared/generated/`: table assets copied from frozen benchmark bundles
- `shared/claims_to_artifacts.md`: mapping from claims to the exact artifact files
- `shared/artifact_availability.tex`: common artifact-release paragraph

## Frozen result roots

- `artifacts/popqa_500/post_fix/`
- `artifacts/freshness_v1/`

## Notes

- The bundled freshness benchmark is described transparently as a deterministic bundled local snapshot, not as a public benchmark run.
- Root-level legacy files from the earlier single-draft scaffold are retained for reference, but the authoritative paper entrypoints are `paper/arxiv/main.tex` and `paper/workshop/main.tex`.
