"""Resolution and validation for pinned third-party source checkouts."""

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "third_party" / "manifest.json"


@lru_cache(maxsize=1)
def _repositories() -> dict[str, dict[str, Any]]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    repositories = manifest.get("repositories")
    if not isinstance(repositories, dict):
        raise ValueError(f"Invalid third-party manifest: {MANIFEST_PATH}")
    return repositories


def repository_info(name: str) -> dict[str, Any]:
    try:
        return _repositories()[name]
    except KeyError as exc:
        raise ValueError(f"Unknown third-party repository: {name}") from exc


def repository_revision(name: str) -> str:
    revision = repository_info(name).get("commit")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError(f"Third-party repository {name!r} has no full commit hash")
    return revision


def require_checkout(name: str) -> Path:
    info = repository_info(name)
    environment_variable = info["environment_variable"]
    configured_root = os.environ.get(environment_variable)
    root = (
        Path(configured_root).expanduser()
        if configured_root
        else MANIFEST_PATH.parent / info["directory"]
    ).resolve()

    missing = [
        relative_path
        for relative_path in info["required_paths"]
        if not (root / relative_path).is_file()
    ]
    if missing:
        command = f"python3 scripts/bootstrap_third_party.py --backend {name}"
        raise RuntimeError(
            f"The pinned {name} source checkout is missing or incomplete at {root}. "
            f"Missing: {', '.join(missing)}. Run `{command}` from the repository root "
            f"or set {environment_variable}."
        )
    return root


def prepend_import_path(path: Path) -> None:
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)
