# GitHub Release Checklist

This checklist is for the first public GitHub-first release of the project.

## Keep In Git

- `multitimescale_memory/`
- `tests/`
- `scripts/`
- `README.md`
- `LICENSE`
- `CITATION.cff`
- lightweight docs under `docs/`
- manuscript provenance sources under `paper/`

## Keep Out Of Git

- virtual environments
- caches and `__pycache__`
- heavy `artifacts/` trees
- heavy raw benchmark files under `data/`
- generated delivery/output folders
- duplicate local snapshots and temporary office files
- large frozen result bundles that belong in release assets

## Release Assets

- `paper-v1.0.0-results-lite.zip`

## Before Tagging

- confirm the README links resolve
- run `python -m multitimescale_memory --help`
- run `pytest tests/test_mvp.py`
- build `paper-v1.0.0-results-lite.zip`
- confirm no oversized or private local files are staged for git

## Publish Steps

1. Commit the public-release cleanup on a dedicated release branch.
2. Merge the release branch into `main`.
3. Push `main` to GitHub.
4. Create the tag `paper-v1.0.0`.
5. Create a GitHub Release and attach the results-lite zip.
6. Update release notes with any final release URL or DOI once public.
