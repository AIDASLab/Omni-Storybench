# Release decisions still required

## Resolved

- All seven research backends are in scope: OpenAI, Claude, vLLM,
  Qwen2.5-Omni, EMOVA, Emu3, and MMaDA.
- Configurations retain one canonical file per distinct backbone, image, and
  speech pipeline. Legacy aliases, superseded tuning copies, and
  device/server-only duplicates were removed.
- Emu3 and MMaDA use official upstream repositories pinned by full commit hash.
  Their checkouts are fetched on demand, while compatibility integration stays
  in this repository. No upstream patch is currently required.
- All backends share one Python 3.10 environment, locked with uv. The common
  compatibility point is PyTorch 2.10.0 and Transformers 4.57.6, with explicit
  overrides documented in `docs/environment.md`.

## Still required

The following changes remain deferred because they can alter the scientific
release, reproducibility contract, or legal terms.

1. Choose the project license and confirm dataset redistribution terms,
   including per-source and per-sample licensing.
2. Choose the canonical public dataset schema: the current parquet loader or
   the existing JSONL release layout.
3. Define the reproducibility policy for model revisions, API model snapshots,
   random seeds, dependency versions, and run manifests.
4. Decide whether dataset construction, annotation, and evaluation utilities
   belong in this repository or in separate release repositories.
