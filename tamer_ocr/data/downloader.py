"""
Legacy Downloader (kept for backward compatibility).
"""

import os
import hashlib
import logging
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, List
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger("TAMER.Downloader")


class DownloadError(Exception):
    pass


class IntegrityError(DownloadError):
    pass


class DiskSpaceError(DownloadError):
    pass


class AdvDatasetDownloader:
    """Legacy downloader with SHA256 verification."""

    CHUNK_SIZE = 8192
    MAX_RETRIES = 5
    BACKOFF_FACTOR = 2
    TIMEOUT = 300

    def __init__(self, data_dir: str = "./data", mirrors: Optional[List[str]] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mirrors = mirrors or []

    def download_file(self, url: str, dest_path: str, expected_sha256: str = "",
                      expected_size: int = 0, force: bool = False) -> bool:
        dest = Path(dest_path)
        if dest.exists() and not force:
            if self._verify_file_integrity(str(dest), expected_sha256, expected_size):
                return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._do_download(url, str(dest))
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def _do_download(self, url: str, dest_path: str):
        req = Request(url)
        req.add_header('User-Agent', 'TAMER-OCR/2.0')
        with urlopen(req, timeout=self.TIMEOUT) as response:
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(self.CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)

    def _verify_file_integrity(self, filepath: str, expected_sha256: str = "",
                                expected_size: int = 0) -> bool:
        path = Path(filepath)
        if not path.exists():
            return False
        if expected_sha256:
            actual = self._compute_sha256(filepath)
            if actual.lower() != expected_sha256.lower():
                return False
        return True

    @staticmethod
    def _compute_sha256(filepath: str) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def extract_archive(self, archive_path: str, extract_to: str) -> bool:
        try:
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(extract_to)
            elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
                with tarfile.open(archive_path, 'r:gz') as tf:
                    tf.extractall(extract_to)
            elif archive_path.endswith('.tar'):
                with tarfile.open(archive_path, 'r') as tf:
                    tf.extractall(extract_to)
            else:
                return False
            return True
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False

    def get_dataset_dir(self, dataset_name: str) -> Path:
        return self.data_dir / dataset_name


def create_downloader(data_dir: str = "./data") -> AdvDatasetDownloader:
    return AdvDatasetDownloader(data_dir=data_dir)
