# Multi-Timescale Memory Routing for Language Models

This repository contains the public code and lightweight release documentation for a multi-timescale controller that routes between parametric answering, retrieval, reusable memory, temporary adaptation, and durable consolidation.

## What This Repository Shows

The project studies one routing problem across three regimes:

- recurring-memory access on PopQA
- controlled updated-knowledge routing on a deterministic bundled FreshQA-style benchmark
- public-data-derived changing-answer evaluation built from weekly FreshQA releases

The main boundary result is regime-sensitive rather than uniformly positive:

- delayed utility helps in stable recurring-memory settings
- controlled update settings benefit from routing across retrieval, memory, and temporary adaptation
- volatile public changing-answer and propagation-heavy regimes are more retrieval-dominant
- delayed-utility calibration partially repairs the public failure mode without changing the action inventory

## Manuscript

The public repository does not ship manuscript files. It focuses on code, reproducibility notes, and lightweight release documentation.

## Install

For a lightweight local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

For real Hugging Face model runs, install the optional runtime dependency set:

```bash
pip install -e ".[dev,hf]"
```

After installation, the CLI is available as either:

```bash
python -m multitimescale_memory --help
multitimescale-memory --help
```

## Quickstart

Run the test suite:

```bash
pytest tests/test_mvp.py
```

Run a lightweight local sanity-check benchmark:

```bash
python -m multitimescale_memory --benchmark demo --mode router
```

Export a bundled freshness benchmark artifact locally:

```bash
python -m multitimescale_memory \
  --benchmark freshness \
  --model-name google/flan-t5-small \
  --freshness-limit 15 \
  --export-freshness-bundle \
  --export-dir artifacts/freshness_v1 \
  --run-label freshness_v1 \
  --audit-limit 50
```

## Reproducibility And Release Assets

This GitHub repo is intentionally lean and code-first:

- code, tests, scripts, and public documentation stay in git
- heavy raw benchmark trees and large frozen artifact bundles stay out of git history
- public frozen summaries and statistics are shipped as release assets instead

See:

- [docs/reproducibility.md](docs/reproducibility.md)
- [docs/public_archive_release.md](docs/public_archive_release.md)
- [docs/github_release_checklist.md](docs/github_release_checklist.md)

Once the first GitHub Release is published, the release asset set may include:

- `paper-v1.0.0-results-lite.zip`

## Repository Layout

- `multitimescale_memory/`: routing, memory, retrieval, benchmark, and reporting code
- `tests/`: regression and benchmark-shape tests
- `scripts/`: export, packaging, and campaign helpers
- `docs/`: public-facing reproducibility and release docs

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). GitHub’s citation panel should point readers to the repository.
