#!/usr/bin/env python3
"""Download and safely unpack the Zeolite files in the Zenodo record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


RECORD_ID = "23736688"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
CHUNK_SIZE = 1024 * 1024


def _with_token(url: str, token: str | None) -> str:
    if not token:
        return url
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query = [(key, value) for key, value in query if key != "token"]
    query.append(("token", token))
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query)))


def _open(url: str, token: str | None):
    request = urllib.request.Request(
        _with_token(url, token), headers={"User-Agent": "zeolite-data-workflow/1.0"}
    )
    try:
        return urllib.request.urlopen(request, timeout=90)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Zenodo returned HTTP {exc.code}; check record access and, if needed, "
            "set ZENODO_ACCESS_TOKEN in the environment."
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Do not include the request URL: it may contain an access token.
        raise RuntimeError(f"Could not reach Zenodo ({type(exc).__name__}).") from None


def _safe_relative_path(name: str) -> Path:
    normalized = name.replace("\\", "/")
    member = PurePosixPath(normalized)
    if member.is_absolute() or not member.parts or any(p in ("", ".", "..") for p in member.parts):
        raise RuntimeError(f"Unsafe path in Zenodo metadata/archive: {name!r}")
    if member.parts[0].endswith(":"):
        raise RuntimeError(f"Unsafe path in Zenodo metadata/archive: {name!r}")
    return Path(*member.parts)


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            h.update(block)
    return h.hexdigest()


def _verify(path: Path, checksum: str | None) -> bool:
    if not checksum or ":" not in checksum:
        return True
    algorithm, expected = checksum.split(":", 1)
    if algorithm not in hashlib.algorithms_available:
        return True
    return _digest(path, algorithm) == expected.lower()


def _download_file(url: str, destination: Path, token: str | None, checksum: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _verify(destination, checksum):
        print(f"Already present: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace an existing, unverifiable file: {destination}")

    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with _open(url, token) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=CHUNK_SIZE)
        if not _verify(temporary, checksum):
            raise RuntimeError(f"Checksum mismatch for downloaded file {destination.name!r}.")
        os.replace(temporary, destination)
    except RuntimeError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        # Network exceptions can contain the full request URL, including its
        # optional access token. Keep diagnostics credential-free.
        raise RuntimeError(f"Download failed for {destination.name!r} ({type(exc).__name__}).") from None
    print(f"Downloaded: {destination}")


def _inside(root: Path, relative: str) -> Path:
    target = (root / _safe_relative_path(relative)).resolve()
    resolved_root = root.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise RuntimeError(f"Archive member escapes extraction directory: {relative!r}")
    return target


def _extract_archive(archive: Path, destination: Path) -> None:
    """Extract supported archives without overwriting different existing files."""
    if not (zipfile.is_zipfile(archive) or tarfile.is_tarfile(archive)):
        return

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".zeolite-extract-", dir=destination) as staging_name:
        staging = Path(staging_name)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as bundle:
                for info in bundle.infolist():
                    target = _inside(staging, info.filename)
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode == stat.S_IFLNK:
                        raise RuntimeError(f"Refusing symlink in archive: {info.filename!r}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=CHUNK_SIZE)
        else:
            with tarfile.open(archive, "r:*") as bundle:
                for info in bundle.getmembers():
                    target = _inside(staging, info.name)
                    if info.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not info.isfile():
                        raise RuntimeError(f"Refusing non-file member in archive: {info.name!r}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = bundle.extractfile(info)
                    if source is None:
                        raise RuntimeError(f"Could not read archive member: {info.name!r}")
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=CHUNK_SIZE)

        for staged in sorted(staging.rglob("*")):
            if not staged.is_file():
                continue
            relative = staged.relative_to(staging)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.is_file() and target.stat().st_size == staged.stat().st_size and _digest(target, "sha256") == _digest(staged, "sha256"):
                    continue
                raise RuntimeError(f"Extraction would overwrite different data: {target}")
            os.replace(staged, target)
    print(f"Extracted: {archive.name} -> {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, type=Path, help="Directory for downloaded files and extracted structures")
    parser.add_argument("--no-extract", action="store_true", help="Download record files but do not unpack archives")
    args = parser.parse_args()

    token = os.environ.get("ZENODO_ACCESS_TOKEN") or None
    download_dir = args.dest / "downloads"
    structures_dir = args.dest / "structures"
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        try:
            with _open(API_URL, token) as response:
                record = json.load(response)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Could not read Zenodo record metadata ({type(exc).__name__}).") from None
        files = record.get("files", [])
        if not files:
            raise RuntimeError("The Zenodo record did not return any downloadable files.")

        print(f"Record: {record.get('metadata', {}).get('title', RECORD_ID)}")
        for item in files:
            relative = _safe_relative_path(item["key"])
            target = download_dir / relative
            download_url = item.get("links", {}).get("download") or item.get("links", {}).get("self")
            if not download_url:
                raise RuntimeError(f"Zenodo did not provide a download URL for {relative.name!r}.")
            _download_file(download_url, target, token, item.get("checksum"))
            if not args.no_extract:
                _extract_archive(target, structures_dir)
    except (KeyError, RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"download_zenodo.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
