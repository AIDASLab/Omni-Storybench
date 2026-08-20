# Compatibility patches

No upstream patch is currently required: the pinned Emu3 and MMaDA files used
by the baseline match the research snapshot. Import-path and cache-key
integration remains in `src/third_party.py` and `src/orchestrator.py`.

If an upstream modification becomes necessary, store it here as a documented
unified diff and add its filename and SHA-256 digest to `manifest.json`. Do not
edit a fetched checkout in place.
