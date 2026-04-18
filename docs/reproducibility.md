# Reproducibility Notes

This repository is the public code surface for the project. It is intentionally lighter than the full internal research workspace: code, tests, scripts, public documentation, and the public paper stay in git, while heavy raw benchmark trees and large frozen artifact bundles are kept out of git history.

## Public Reproducibility Surfaces

- `docs/paper/paper.pdf`: canonical reader version of the public manuscript
- `docs/paper/paper.docx`: editable manuscript source for the same public release
- GitHub Release assets:
  - `paper-v1.0.0-paper.pdf`
  - `paper-v1.0.0-manuscript.docx`
  - `paper-v1.0.0-results-lite.zip`

The results-lite asset is the public frozen evidence bundle for this GitHub-first release. It contains the curated manifests, aggregate tables, run rows, significance/bootstrap summaries, audit summaries, and reliability notes needed to inspect the paper’s headline claims without shipping the full raw artifact tree in git.

## Install And Run

Lightweight setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Optional Hugging Face runtime support:

```bash
pip install -e ".[dev,hf]"
```

Sanity checks:

```bash
python -m multitimescale_memory --help
pytest tests/test_mvp.py
python -m multitimescale_memory --benchmark demo --mode router
```

## Public Release Assets Versus Heavy Local State

Kept in git:

- code under `multitimescale_memory/`
- tests under `tests/`
- public docs under `docs/`
- manuscript provenance sources under `paper/`

Kept out of git and shipped separately when needed:

- heavy raw benchmark files under `data/`
- large frozen artifact trees under `artifacts/`
- generated delivery/output directories

## Building The Lightweight Results Bundle

Use:

```bash
python scripts/build_results_lite_release.py
```

This creates a small GitHub Release asset zip with the curated frozen summaries/statistics needed for public inspection. The script intentionally excludes raw heavy traces, caches, environments, and large benchmark trees.

## Notes

- The public GitHub release is code-first and PDF-first, not LaTeX-package-first.
- The LaTeX tree remains in the repository as secondary provenance.
- Large frozen bundles and raw benchmark inputs should be attached as release assets or archived later, not committed into git history.
