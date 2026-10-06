#!/usr/bin/env python3
"""Download the latest Omni-StoryBench dataset without third-party packages.

Choose a destination with --local-dir. Failed partial transfers are retained;
reruns verify and skip completed files. No global cache is used.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


REPO_ID = "snu-aidas/Omni-StoryBench"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class IntegrityError(ValueError):
    """A transfer ended without the expected bytes; its partial file is retained."""


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            if urlparse(newurl).scheme != "https":
                raise ValueError("Refusing a non-HTTPS download redirect")
            if urlparse(req.full_url).netloc != urlparse(newurl).netloc:
                redirected.remove_header("Authorization")
        return redirected


def safe_destination(path: Path) -> Path:
    """Accept a user-selected directory while rejecting symlink write targets."""
    absolute = Path(os.path.abspath(path.expanduser()))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Refusing symlink destination: {current}")
    return absolute


def request(opener, url: str):
    headers = {"User-Agent": "Omni-StoryBench-release-downloader/1"}
    token = os.environ.get("HF_TOKEN")
    if token and urlparse(url).netloc == "huggingface.co":
        headers["Authorization"] = f"Bearer {token}"
    return opener.open(Request(url, headers=headers), timeout=60)


def matches(path: Path, entry: dict) -> bool:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    if entry.get("lfs", {}).get("sha256"):
        digest = hashlib.sha256()
        expected = entry["lfs"]["sha256"]
    else:
        digest = hashlib.sha1()
        digest.update(f"blob {entry['size']}\0".encode("ascii"))
        expected = entry["blobId"]
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def retry_wait(exc: Exception, attempt: int) -> float:
    if isinstance(exc, HTTPError):
        if exc.code not in {429, 500, 502, 503, 504}:
            raise exc
        retry_after = exc.headers.get("Retry-After", "")
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass
    return min(60.0, 10.0 * 2 ** attempt)


def fetch_metadata(opener, revision: str, retries: int) -> dict:
    url = (f"https://huggingface.co/api/datasets/{REPO_ID}/revision/"
           f"{quote(revision, safe='')}?blobs=true")
    for attempt in range(retries + 1):
        try:
            with request(opener, url) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, IncompleteRead, ConnectionError) as exc:
            if attempt == retries:
                raise
            time.sleep(retry_wait(exc, attempt))
    raise RuntimeError("Metadata request did not complete")


def download_file(opener, revision: str, entry: dict, output: Path, retries: int) -> str:
    output = safe_destination(output)
    if output.exists():
        if matches(output, entry):
            return "verified-existing"
        raise ValueError(f"Existing file differs from selected snapshot: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://huggingface.co/datasets/{REPO_ID}/resolve/"
           f"{revision}/{quote(entry['rfilename'], safe='/')}")
    for attempt in range(retries + 1):
        partial = safe_destination(output.with_name(output.name + f".partial-{time.time_ns()}"))
        try:
            with request(opener, url) as response, partial.open("xb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
            if not matches(partial, entry):
                raise IntegrityError(f"Size or content hash mismatch; retained {partial}")
            if output.exists():
                raise ValueError(f"Destination appeared during download: {output}")
            safe_destination(output)
            partial.rename(output)
            return "downloaded-and-verified"
        except (HTTPError, URLError, TimeoutError, IncompleteRead, ConnectionError, IntegrityError) as exc:
            if attempt == retries:
                raise
            wait = retry_wait(exc, attempt)
            print(f"Retrying {entry['rfilename']} in {wait:g}s; partial files retained.",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError("Download did not complete")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path,
                        default=PROJECT_ROOT / "evaluation/downloaded_dataset")
    parser.add_argument("--revision", default="main", help="Branch, tag or commit; resolved once to a commit")
    parser.add_argument("--include", action="append", default=[], metavar="GLOB",
                        help="Select a subset for inspection; repeatable. Omit for the complete snapshot.")
    parser.add_argument("--list-only", action="store_true", help="Print selection without downloading files")
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args(argv)
    if args.retries < 0:
        parser.error("--retries must be nonnegative")
    destination = safe_destination(args.local_dir)
    opener = build_opener(SafeRedirect())
    metadata = fetch_metadata(opener, args.revision, args.retries)
    revision = metadata["sha"]
    entries = []
    for entry in metadata["siblings"]:
        name = entry["rfilename"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Invalid repository filename: {name!r}")
        if not args.include or any(fnmatch.fnmatchcase(name, pattern) for pattern in args.include):
            if not isinstance(entry.get("size"), int) or not (
                entry.get("blobId") or entry.get("lfs", {}).get("sha256")
            ):
                raise ValueError(f"Missing remote size/hash metadata for {name}")
            safe_destination(destination / name)
            entries.append(entry)
    if not entries:
        raise ValueError("No repository files match the requested selection")
    summary = {"repo_id": REPO_ID, "revision": revision, "destination": str(destination),
               "complete_snapshot_selection": not bool(args.include),
               "files": len(entries), "bytes": sum(entry["size"] for entry in entries)}
    print(json.dumps(summary, indent=2), flush=True)
    if args.list_only:
        return 0
    records = []
    for index, entry in enumerate(entries, 1):
        status = download_file(opener, revision, entry, destination / entry["rfilename"], args.retries)
        records.append({"path": entry["rfilename"], "status": status, "size": entry["size"],
                        "hash": entry.get("lfs", {}).get("sha256") or entry["blobId"]})
        print(f"[{index}/{len(entries)}] {status}: {entry['rfilename']}", flush=True)
    record_dir = safe_destination(destination / ".download_records")
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = safe_destination(record_dir / f"run-{time.time_ns()}.json")
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump({**summary, "verified_files": records}, stream, indent=2)
        stream.write("\n")
    print(f"Download record: {record_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
