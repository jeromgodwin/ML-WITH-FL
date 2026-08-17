"""Download and extract the EMBER 2018_2 dataset.

Official source: https://ember.elastic.co/ember_dataset_2018_2.tar.bz2 (~1.6 GB).
Downloads to <data_dir>/ember_2018_2/ and extracts the .dat arrays.
Supports resuming interrupted downloads (--force to redownload).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tarfile
from pathlib import Path

import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedshield.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)

EMBER_URL = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"
TAR_NAME = "ember_dataset_2018_2.tar.bz2"
EXPECTED_BYTES = 1696539273  # from server Content-Length
EMBER_SHA256 = "b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(tar_path: Path) -> bool:
    """Verify the tarball against the official sha256."""
    if not tar_path.exists():
        return False
    logger.info("Verifying sha256 of %s ...", tar_path.name)
    ok = sha256_file(tar_path) == EMBER_SHA256
    logger.info("sha256 %s", "OK" if ok else "MISMATCH")
    return ok


def download(url: str, dest: Path) -> None:
    """Download with resume support and progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        # Try to resume
        existing = dest.stat().st_size
        request = urllib.request.Request(url, headers={"Range": f"bytes={existing}-"})
        mode = "ab"
        logger.info("Resuming download at %d bytes", existing)
    else:
        request = urllib.request.Request(url)
        mode = "wb"
        existing = 0

    with urllib.request.urlopen(request) as resp:
        if mode == "ab" and resp.status != 206:
            # Server ignored our Range header: start over
            logger.info("Server ignored Range header (status %s); restarting download", resp.status)
            existing = 0
            mode = "wb"
            with open(dest, "wb") as f:
                f.truncate(0)
        total = existing + int(resp.headers.get("Content-Length", 0))
        downloaded = existing
        last_reported = -1
        with open(dest, mode) as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                report_mb = downloaded // (100 * 1024 * 1024)
                if report_mb > last_reported:
                    last_reported = report_mb
                    pct = downloaded * 100 / total if total else 0
                    logger.info("Downloaded %d / %d MB (%.1f%%)",
                                downloaded // (1024 * 1024), total // (1024 * 1024), pct)
                if downloaded % (1024 * 1024 * 1024) == 0:
                    logger.info("Downloaded %d MB ...", downloaded // (1024 * 1024))
    logger.info("Download complete: %s", dest)


def extract(tar_path: Path, target_dir: Path) -> None:
    """Extract tar.bz2 into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s ...", tar_path.name)
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(target_dir)
    logger.info("Extracted into %s", target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EMBER 2018_2 dataset")
    parser.add_argument("--data-dir", default="data", help="root data directory")
    parser.add_argument("--force", action="store_true", help="redownload even if present")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tar_path = data_dir / TAR_NAME
    extract_dir = data_dir / "ember_2018_2"

    if args.force and tar_path.exists():
        tar_path.unlink()

    if not tar_path.exists():
        download(EMBER_URL, tar_path)
    elif tar_path.stat().st_size < EXPECTED_BYTES:
        # Incomplete download: resume from where we left off
        logger.info("Partial download (%d bytes); resuming ...", tar_path.stat().st_size)
        download(EMBER_URL, tar_path)
        if not verify(tar_path):
            logger.error("Resumed download failed verification; deleting and redownloading.")
            tar_path.unlink()
            download(EMBER_URL, tar_path)

    if not verify(tar_path):
        logger.error("Tarball failed verification; deleting and redownloading.")
        tar_path.unlink()
        download(EMBER_URL, tar_path)
        if not verify(tar_path):
            logger.error("Verification failed again; aborting.")
            sys.exit(1)

    # Extract if the expected .dat files are missing
    expected = [extract_dir / "X_train.dat", extract_dir / "X_test.dat"]
    if all(p.exists() for p in expected):
        logger.info("Dataset already extracted; nothing to do.")
        return

    extract(tar_path, extract_dir)
    logger.info("Expected files: %s", [str(p) for p in expected])


if __name__ == "__main__":
    main()
