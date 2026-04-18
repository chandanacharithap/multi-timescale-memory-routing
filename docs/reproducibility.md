# Reproducibility Notes

This repository is the public code surface for the project. It is intentionally lighter than the full internal research workspace: code, tests, scripts, and public documentation stay in git, while heavy raw benchmark trees and large frozen artifact bundles stay out of git history.

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

## Public Repo Versus Heavy Local State

Kept in git:

- code under `multitimescale_memory/`
- tests under `tests/`
- public docs under `docs/`

Kept out of git and managed separately when needed:

- heavy raw benchmark files under `data/`
- large frozen artifact trees under `artifacts/`
- generated delivery/output directories

## Notes

- This repo is code-first and is not the arXiv submission package.
- Large frozen bundles and raw benchmark inputs should be managed separately or archived later, not committed into git history.
