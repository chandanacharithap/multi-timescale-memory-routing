# Public Freshness JSONL Schema

Each line represents one timestamped update episode.

Required top-level fields:

- `episode_id`
- `question`
- `gold_answer`
- `timestamp`
- `subject`
- `relation`
- `support_docs`

Recommended top-level fields:

- `dataset_id`
- `domain`
- `parametric_answer`
- `parametric_confidence`
- `popularity_bin`
- `recurrence_hint`
- `stability_hint`
- `volatility_hint`
- `contradiction_hint`
- `freshness`
- `update_expected`
- `metadata`

Each `support_docs` item should contain:

- `doc_id`
- `text`
- `answer`
- `source`
- `timestamp`
- `trust`
- `relevance`

Recommended `metadata` keys:

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

Behavioral requirements:

- episodes must be sorted by non-decreasing `timestamp`
- support documents should obey causal visibility
- repeated subject/relation updates should preserve the original temporal order
- rollback probes and forgetting probes should be marked in `metadata`
