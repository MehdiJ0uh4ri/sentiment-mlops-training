#!/usr/bin/env python
"""Download and extract the UCI Sentiment Labelled Sentences dataset (CC BY 4.0)."""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from sentiment import config


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    targets = {name: config.RAW_DIR / name for name in config.RAW_FILES.values()}

    if not args.force and all(p.exists() for p in targets.values()):
        print("Raw files already present (use --force to re-download).")
        return 0

    print(f"Downloading {config.DATASET_URL}")
    request = urllib.request.Request(
        config.DATASET_URL, headers={"User-Agent": "sentiment-mlops/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except urllib.error.URLError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    print(f"zip size: {len(payload)} bytes, sha256: {sha256_bytes(payload)}")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        by_basename = {
            Path(n).name: n
            for n in zf.namelist()
            if not n.endswith("/") and "__MACOSX" not in n
        }
        for fname, target in targets.items():
            if fname not in by_basename:
                print(f"{fname} not found in zip. Found: {sorted(by_basename)}", file=sys.stderr)
                return 1
            data = zf.read(by_basename[fname])
            target.write_bytes(data)
            print(f"extracted {fname}: {len(data)} bytes, sha256: {sha256_bytes(data)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())