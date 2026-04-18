# Public GitHub Release

This project now targets a GitHub-first public release rather than a LaTeX-package-first public surface.

## Public Home

- GitHub repository: `https://github.com/chandanacharithap/multi-timescale-memory-routing`
- Planned first public tag: `paper-v1.0.0`
- Zenodo: optional after the first public GitHub release

## Public Release Assets

The first release is expected to attach:

- `paper-v1.0.0-paper.pdf`
- `paper-v1.0.0-manuscript.docx`
- `paper-v1.0.0-results-lite.zip`

## Release Preparation

1. Confirm the repo is lean and code-first.
2. Export the current public manuscript to:
   - `docs/paper/paper.docx`
   - `docs/paper/paper.pdf`
3. Build the lightweight frozen-results bundle:

```bash
python scripts/build_results_lite_release.py
```

4. Run the public checks:

```bash
python -m multitimescale_memory --help
pytest tests/test_mvp.py
python -m multitimescale_memory --benchmark demo --mode router
```

## Publish Steps

1. Push the cleaned `main` branch to GitHub.
2. Create the tag `paper-v1.0.0`.
3. Open a GitHub Release for that tag.
4. Attach the PDF, Word manuscript, and results-lite zip.
5. Update any final release notes or archival metadata after the release is live.

## Important Note

The anonymous TMLR manuscript should remain free of direct GitHub or release links even after the named public GitHub materials are published.
