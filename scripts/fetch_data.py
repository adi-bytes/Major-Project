"""Data fetching script for PG-SBD Designer.

Downloads all raw files from the hywong2/Semiconductor_Database repository
with proper URL encoding of '+' characters as '%2B', saves them to data/raw/,
computes SHA-256 checksums, and creates a manifest.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

RAW_BASE_URL = "https://raw.githubusercontent.com/hywong2/Semiconductor_Database/master/GaO_Schottky_Diode_T_WF"

FILES_TO_FETCH = [
    "new_dataset_GaO_WF+T.csv",
    "n9000_4e+16_3X4e+16_377_5014.csv",
    "ds_file_9000_4e+16_3X4e+16_377_5014.cmd",
    "n9000_4e+16_3X4e+16_377_5014_msh.cmd",
    "Readme",
    "TCAD-Augmented_Machine_Learning_With_and_Without_Domain_Expertise.pdf",
]


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_all(dest_dir: Path | str = "data/raw") -> dict[str, dict]:
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}

    for filename in FILES_TO_FETCH:
        # Encode '+' as '%2B' and other characters properly
        encoded_filename = urllib.parse.quote(filename, safe="")
        url = f"{RAW_BASE_URL}/{encoded_filename}"
        target_file = dest_path / filename

        print(f"Downloading {filename} from {url} ...")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PG-SBD-Fetch"},
        )
        with urllib.request.urlopen(req) as resp, open(target_file, "wb") as out_f:
            content = resp.read()
            out_f.write(content)

        file_size = target_file.stat().st_size
        sha256_hash = compute_sha256(target_file)

        manifest[filename] = {
            "source_url": url,
            "filename": filename,
            "size_bytes": file_size,
            "sha256": sha256_hash,
        }
        print(f"  -> Saved {filename} ({file_size} bytes, sha256: {sha256_hash[:16]}...)")

    manifest_path = dest_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest successfully written to {manifest_path}")
    return manifest


if __name__ == "__main__":
    fetch_all()
