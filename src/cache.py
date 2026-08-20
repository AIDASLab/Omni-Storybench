"""Content-addressed cache for artifacts shared between baselines.

A baseline is a combination of a backbone, a T2I model and a T2S model, so runs
overlap heavily: every ``gpt54_*`` baseline shares one backbone pass, and the
``gpt54_flux_*`` pair shares its images on top of that. Each artifact is keyed by
a hash of everything that determines it -- the producing component with its
generation parameters, plus the exact prompt it was given -- so a later run
reuses what an earlier one produced, while a changed prompt, model or parameter
misses instead of silently reusing a stale artifact.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_ROOT = os.environ.get(
    "OMNI_STORYBENCH_CACHE",
    str(PROJECT_ROOT / ".cache" / "artifacts"),
)

STAGES = ("raw", "image", "speech")


class ArtifactCache:
    """Stores artifacts under ``<cache_root>/<stage>/<key[:2]>/<key>/``."""

    def __init__(self, cache_root: str, owner: str = "", enabled: bool = True, refresh: bool = False):
        """
        Args:
            cache_root: Root directory holding every cached artifact.
            owner: Baseline name recorded on entries this run creates.
            enabled: When false, every lookup misses and nothing is stored.
            refresh: Regenerate artifacts and overwrite the entries they came from.
        """
        self.cache_root = cache_root
        self.owner = owner
        self.enabled = enabled
        self.refresh = refresh
        self.stats: Dict[str, Dict[str, int]] = {stage: {"hit": 0, "miss": 0} for stage in STAGES}

    @staticmethod
    def key(payload: Any) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _entry_dir(self, stage: str, key: str) -> str:
        return os.path.join(self.cache_root, stage, key[:2], key)

    @staticmethod
    def _same_file(a: str, b: str) -> bool:
        try:
            return os.path.samefile(a, b)
        except OSError:
            return False

    @classmethod
    def _link_or_copy(cls, src: str, dest: str) -> None:
        """Publish src at dest atomically, preferring a hardlink over a copy."""
        if cls._same_file(src, dest):
            # Already the same inode. Renaming onto itself is a no-op that would
            # leave the temporary link behind.
            return

        tmp = f"{dest}.tmp-{os.getpid()}"
        if os.path.lexists(tmp):
            os.remove(tmp)
        try:
            os.link(src, tmp)
        except OSError:
            # Different filesystem, or a link count limit.
            shutil.copy2(src, tmp)
        os.replace(tmp, dest)
        if os.path.lexists(tmp):
            os.remove(tmp)

    def restore(self, stage: str, key: str, files: Dict[str, str]) -> bool:
        """Publish a cached entry to its destinations. All-or-nothing.

        Args:
            stage: One of STAGES.
            key: Entry key from :meth:`key`.
            files: Maps each file name inside the entry to its destination path.
        """
        if not self.enabled or self.refresh:
            self._record(stage, hit=False)
            return False

        entry_dir = self._entry_dir(stage, key)
        sources = {name: os.path.join(entry_dir, name) for name in files}
        if not all(os.path.isfile(path) for path in sources.values()):
            self._record(stage, hit=False)
            return False

        for name, dest in files.items():
            self._link_or_copy(sources[name], dest)
        self._record(stage, hit=True)
        return True

    def store(self, stage: str, key: str, files: Dict[str, str], payload: Any, adopt: bool = True) -> None:
        """Publish freshly generated files as an entry, optionally adopting it.

        The first writer of a key wins. When ``adopt`` is true, a run that loses
        the race re-links its output to the stored artifact, so baselines sharing
        a key end up byte-identical -- correct for a leaf artifact (image, speech)
        with nothing generated from it.

        Pass ``adopt=False`` for an artifact the caller has ALREADY derived other
        outputs from this run (the ``raw`` backbone output drives image + speech).
        Adopting a racing winner there would leave this run's ``raw``/``text`` from
        one sampling while its image and speech came from another. Keeping the local
        copy preserves within-run consistency; future runs still read the winner via
        :meth:`restore`.
        """
        if not self.enabled:
            return

        entry_dir = self._entry_dir(stage, key)
        os.makedirs(entry_dir, exist_ok=True)

        for name, src in files.items():
            target = os.path.join(entry_dir, name)
            if self.refresh or not os.path.isfile(target):
                self._link_or_copy(src, target)

        meta_path = os.path.join(entry_dir, "meta.json")
        if self.refresh or not os.path.isfile(meta_path):
            meta = {
                "stage": stage,
                "key": key,
                "created_by": self.owner,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "payload": payload,
            }
            tmp = f"{meta_path}.tmp-{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=4, default=str)
            os.replace(tmp, meta_path)

        if adopt:
            for name, dest in files.items():
                self._link_or_copy(os.path.join(entry_dir, name), dest)

    def _record(self, stage: str, hit: bool) -> None:
        self.stats.setdefault(stage, {"hit": 0, "miss": 0})["hit" if hit else "miss"] += 1

    def summary(self) -> str:
        if not self.enabled:
            return "Cache disabled."
        parts = [
            f"{stage} {counts['hit']}/{counts['hit'] + counts['miss']}"
            for stage, counts in self.stats.items()
            if counts["hit"] or counts["miss"]
        ]
        reused = ", ".join(parts) if parts else "nothing generated"
        return f"Cache reuse (hit/total): {reused} [{self.cache_root}]"
