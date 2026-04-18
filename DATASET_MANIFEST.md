# Dataset Manifest

This file records the frozen local dataset inputs that back the official TMLR experiment surface. Only bundles listed in `paper_run_manifest_index.json` are submission-facing and reportable.

## FreshQA Public Snapshot Set

- Source family: weekly public FreshQA CSV snapshots
- Frozen raw snapshots:
  - `data/freshqa_2025-02-17.csv`
    - `sha256: c948ccd749391365b25daf800c55a18c99f9ec1b9db51dfda6e049cdf6d5d48e`
  - `data/freshqa_2025-02-24.csv`
    - `sha256: 07d11ea3092f15304b72a2e9ecd73a267225d8a72dee336bc2aaab9c6acf5aee`
  - `data/freshqa_2025-03-03.csv`
    - `sha256: 06916019335b01fdf1883cfafc20e7f18dcc31fdf126e4d132752908b744c648`
  - `data/freshqa_2025-03-10.csv`
    - `sha256: 7f0fd1455f52a4faf18248ec66874afd49d8fe6eceb69f98cefe276f70470049`
  - `data/freshqa_2025-03-17.csv`
    - `sha256: 6ca05c525eaf3256c801db7d71eadabf3c108e781ee7cb47bfa6bfe34d59911b`
  - `data/freshqa_2025-03-24.csv`
    - `sha256: 4ee3776c4312bb93a90a567182eb7097ac6795874eca8b3d98b92b5fcfbf55e8`
  - `data/freshqa_2025-03-31.csv`
    - `sha256: 683cd979e7e70c46d673c41c565a68ee941958c63ffe7b2efeef23592699ec88`
  - `data/freshqa_2025-04-07.csv`
    - `sha256: 0cafff57ee33965eca0f31a70c97e18aaa21b07043f8f689f3cae9a8cd547080`
  - `data/freshqa_2025-11-24.csv`
    - `sha256: 47f971c5760f286a388f94e671aeee9548d1cbaebdfaf77b66bf27ed2a1cb420`
- Preprocessing code:
  - `multitimescale_memory/freshqa_export.py`
    - `sha256: 479f565217e7f8aa7d81988df4e4304750ad2be68e37759c2b2140675151248c`
  - `scripts/build_tmlr_freshqa_bundle.py`
- Frozen processed tracks:
  - `data/freshqa_public_full_main_v3.jsonl`
    - `sha256: b7001bb10e7c48842e324502dbf3d650a8fd0fcc2b65d65f892da7aa59a65173`
    - `rows: 796`
  - `data/freshqa_public_full_sequence_v3.jsonl`
    - `sha256: be114a5107330dc357898b4022f4e1ed05b50bec7a96a30be2b2d83b083d8bcc`
    - `rows: 796`
- Final official sample count used by current official runs:
  - main track: `796` episodes
  - sequence track: `796` episodes
- Supporting provenance bundle:
  - `artifacts/tmlr_freshqa_full_v1/provenance_manifest.json`
- Supporting audit bundle:
  - `artifacts/tmlr_official/freshqa_leakage_audit_manual_v1`
- Status:
  - official FreshQA main and sequence runs are frozen for FLAN small/base, Qwen 1.5B, and SmolLM2 360M

## MQuAKE Snapshot Set

- Source family: local Princeton NLP MQuAKE benchmark snapshot
- Frozen raw file:
  - `data/benchmarks/mquake/MQuAKE-CF-3k.json`
    - `sha256: ce27a39c39f2983512b9b5578fadea5fbe352e5368f49d64f38d37ce304edc80`
- Additional local raw files retained:
  - `data/benchmarks/mquake/MQuAKE-CF.json`
  - `data/benchmarks/mquake/MQuAKE-CF-3k-v2.json`
  - `data/benchmarks/mquake/MQuAKE-T.json`
- Preprocessing code:
  - `multitimescale_memory/mquake.py`
- Final official sample count used by current official runs:
  - `3000` cases
  - `21000` emitted episodes
  - `3000` update episodes
  - `9000` single-hop probes
  - `9000` multi-hop probes
- Reliability note:
  - `artifacts/tmlr_official/mquake_reliability_note_v1/reliability_note.json`
- Status:
  - official FLAN-T5 Base and FLAN-T5 Small MQuAKE runs are frozen

## UniEdit Snapshot Set

- Source family: `qizhou/UniEdit`
- Frozen local tree:
  - `data/benchmarks/uniedit/README.md`
  - `data/benchmarks/uniedit/test/*.json`
  - `data/benchmarks/uniedit/train/*.json`
- Snapshot manifest:
  - `data/benchmarks/uniedit/manifest.json`
  - `complete: true`
  - `frozen files: 51`
- Preprocessing code:
  - `scripts/fetch_benchmark_inputs.py`
  - `multitimescale_memory/uniedit.py`
- Final official sample configuration:
  - `25` test domains
  - `50` sampled cases per domain
  - `1250` sampled edits
  - `3750` total episodes
  - `sample seed: 2026`
- Reliability note:
  - `artifacts/tmlr_official/uniedit_reliability_note_v1/reliability_note.json`
- Status:
  - snapshot frozen locally
  - `uniedit_official_v1` is the last live official benchmark run and becomes reportable only after the bundle lands and is registered

## Legacy Support Inputs

- `data/benchmarks/knowedit/`
  - retained for development-stage benchmark work only
  - not part of the final locked TMLR main evidence surface
- internal bundled freshness artifacts remain available as legacy support only

## Policy

- `paper_run_manifest_index.json` is the source of truth for which artifact roots are reportable.
- `tmlr_official_reportable` is the final TMLR evidence surface.
- `legacy_reportable_support` is retained for background support only and is not part of the final official TMLR claim surface.
- Provisional, synthetic, sample-fixture, and debug-only data must not appear in submission packages or paper-facing manifests.
