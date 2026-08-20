#!/usr/bin/env python3
"""Fetch or verify immutable Emu3 and MMaDA source checkouts."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "third_party" / "manifest.json"
DEFAULT_TARGET_ROOT = PROJECT_ROOT / "third_party"


def load_repositories() -> dict[str, dict[str, Any]]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("schema_version") != 1:
        raise SystemExit(f"Unsupported third-party manifest: {MANIFEST_PATH}")
    repositories = manifest.get("repositories")
    if not isinstance(repositories, dict) or not repositories:
        raise SystemExit(f"No repositories in {MANIFEST_PATH}")
    return repositories


def run_git(*arguments: str, cwd: Path | None = None, capture: bool = False) -> str:
    command = ["git", *arguments]
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def verify_checkout(name: str, info: dict[str, Any], target: Path) -> list[str]:
    errors = []
    if not (target / ".git").is_dir():
        return [f"{name}: no Git checkout at {target}"]

    try:
        actual_commit = run_git("rev-parse", "HEAD", cwd=target, capture=True)
        dirty = run_git("status", "--porcelain", cwd=target, capture=True)
    except subprocess.CalledProcessError as exc:
        return [f"{name}: Git verification failed at {target}: {exc}"]

    if actual_commit != info["commit"]:
        errors.append(
            f"{name}: expected {info['commit']}, found {actual_commit} at {target}"
        )
    if dirty:
        errors.append(f"{name}: checkout is dirty at {target}")

    for relative_path in info["required_paths"]:
        if not (target / relative_path).is_file():
            errors.append(f"{name}: missing {relative_path} at {target}")
    return errors


def fetch_checkout(name: str, info: dict[str, Any], target: Path) -> None:
    if target.exists():
        if not (target / ".git").is_dir():
            raise SystemExit(
                f"Refusing to replace non-Git path {target}. Move it aside and retry."
            )
        dirty = run_git("status", "--porcelain", cwd=target, capture=True)
        if dirty:
            raise SystemExit(
                f"Refusing to change dirty checkout {target}. Commit, stash, or move it first."
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        run_git(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--depth",
            "1",
            info["url"],
            str(target),
        )

    run_git("fetch", "--depth", "1", "origin", info["commit"], cwd=target)
    run_git("checkout", "--detach", info["commit"], cwd=target)


def parse_args(repository_names: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        action="append",
        choices=repository_names,
        help="Fetch or check only this backend; repeat to select multiple",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify existing checkouts without network access or changes",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print pinned repositories without fetching them",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help="Checkout parent directory (default: third_party/)",
    )
    return parser.parse_args()


def main() -> int:
    repositories = load_repositories()
    args = parse_args(sorted(repositories))
    selected_names = args.backend or sorted(repositories)

    if args.list:
        for name in selected_names:
            info = repositories[name]
            print(f"{name}: {info['url']} @ {info['commit']}")
        return 0

    all_errors = []
    for name in selected_names:
        info = repositories[name]
        target = args.target_root.resolve() / info["directory"]
        if not args.check:
            print(f"Fetching {name} at {info['commit']} into {target}")
            fetch_checkout(name, info, target)
        errors = verify_checkout(name, info, target)
        if errors:
            all_errors.extend(errors)
        else:
            print(f"Verified {name}: {info['commit']}")

    if all_errors:
        for error in all_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
