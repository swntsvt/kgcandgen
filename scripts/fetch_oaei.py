"""Fetch (optionally) and verify the OAEI benchmark files listed in config/oaei_manifest.csv.

Usage:
    python scripts/fetch_oaei.py --data-root data/oaei_track_cache            # verify only
    python scripts/fetch_oaei.py --data-root data/oaei_track_cache --download # fetch missing

Each manifest row gives a path relative to the data root, its size and SHA-256. Rows with a
``url`` can be downloaded with ``--download``; every file present is verified, and the
script exits non-zero if any file is missing or does not match its checksum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "config" / "oaei_manifest.csv"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--download", action="store_true", help="Download missing files with a url.")
    args = parser.parse_args(argv)

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    missing, mismatched, verified = [], [], 0
    for row in rows:
        path = args.data_root / row["relative_path"]
        if not path.exists() and args.download and row.get("url"):
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {row['url']} -> {path}")
            urllib.request.urlretrieve(row["url"], path)  # noqa: S310 (manifest URLs only)
        if not path.exists():
            missing.append(row["relative_path"])
            continue
        if sha256_of(path) != row["sha256"]:
            mismatched.append(row["relative_path"])
            continue
        verified += 1

    print(f"Verified {verified}/{len(rows)} file(s).")
    for label, items in (("Missing", missing), ("Checksum mismatch", mismatched)):
        for item in items:
            print(f"{label}: {item}", file=sys.stderr)
    return 0 if not missing and not mismatched else 1


if __name__ == "__main__":
    raise SystemExit(main())
