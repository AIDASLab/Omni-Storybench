# Pinned third-party source

Emu3 and MMaDA are fetched from their official repositories at immutable
commits recorded in `manifest.json`. Their source code is not copied into this
repository.

Fetch both repositories:

```bash
python3 scripts/bootstrap_third_party.py
```

Fetch or verify one repository:

```bash
python3 scripts/bootstrap_third_party.py --backend emu3
python3 scripts/bootstrap_third_party.py --backend mmada --check
```

The default checkouts are `third_party/Emu3` and `third_party/MMaDA`; both are
ignored by Git. Set `OMNI_STORYBENCH_EMU3_ROOT` or
`OMNI_STORYBENCH_MMADA_ROOT` to use verified checkouts elsewhere.

The bootstrap command never overwrites a dirty checkout. It validates the full
commit hash and the files required by the local adapters. It fetches source
only; Python/CUDA packages are installed by `uv sync --frozen` from the root
lockfile.
