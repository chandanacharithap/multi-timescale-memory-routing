# Bundled Freshness Snapshot Protocol

This document describes the deterministic `bundled_freshqa_style_snapshot_v1` benchmark used in `artifacts/freshness_v1/`.

## Construction

The benchmark is defined directly in [`multitimescale_memory/freshness.py`](../multitimescale_memory/freshness.py). It is a small, fully deterministic timestamped update stream intended to stress all six controller actions on updated knowledge.

The stream contains:

- stable updates
- confirmations
- volatile updates
- one rollback probe
- two forgetting probes

Each episode is represented with the existing `EpisodeInput` interface, and benchmark-specific details live in `EpisodeInput.metadata`.

## Metadata Fields

Each freshness episode may include:

- `benchmark_source`
- `update_id`
- `update_type`
- `source_support_count`
- `domain_change_rate`
- `source_freshness`
- `stability_class`
- `volatility_class`
- `stale_answers`
- `temporary_patch_horizon`
- `rollback_probe`
- `consolidation_candidate`
- `probe_group`
- `probe_phase`

## Update Types

- `stable_update`: the first observed corrected fact for a stable subject/relation pair
- `confirmation`: later evidence that supports the same stable corrected fact
- `volatile_update`: rapidly changing fact where temporary adaptation is preferred to durable storage
- `rollback_probe`: corrective episode that should invalidate a previously durable answer if the evidence conflicts
- `forgetting_probe`: unaffected control fact used to verify that unrelated knowledge remains intact

## Evidence Visibility

Freshness retrieval is timestamp-aware.

- Only evidence documents with `doc.timestamp <= episode.timestamp` are visible.
- Episode-local `support_docs` are also filtered by timestamp.
- This prevents future evidence leakage and makes the update stream causal.

## Deterministic Split

- There is a single frozen slice in `freshness_v1`.
- Episode order is fixed.
- Corpus documents are bundled and fixed.
- Model name is fixed to `google/flan-t5-small` for the bundled result.

## Regeneration

To regenerate the exact benchmark stream:

1. Load the episode definitions in `multitimescale_memory/freshness.py`.
2. Preserve the listed timestamps and document timestamps exactly.
3. Preserve the metadata fields and their values exactly.
4. Run the freshness export command recorded in [`docs/reproducibility.md`](./reproducibility.md).

## Transparency Note

This benchmark is currently a bundled local `FreshQA`-style artifact included with the repository. It should be described transparently as such in the paper until a public-benchmark replacement is added.
