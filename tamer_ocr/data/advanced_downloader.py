"""
Advanced Dataset Downloader for TAMER OCR Training.

Enhanced downloader supporting:
- Hugging Face datasets with authentication
- Kaggle datasets with API credentials
- Zenodo direct ZIP downloads with streaming
- GitHub repository cloning
- HTTP/HTTPS proxy configuration
- Resume capability and retry logic
"""

import os
import sys
import time
import logging
import shutil
import zipfile
import tarfile
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("TAMER.Downloader")


class DownloadError(Exception):
    """Custom exception for download-related errors."""
    pass


class IntegrityError(DownloadError):
    """Exception raised when file integrity check fails."""
    pass


class DiskSpaceError(DownloadError):
    """Exception raised when there's insufficient disk space."""
    pass


class AdvDownloader:
    """
    Advanced downloader supporting multiple dataset sources.
    """

    CHUNK_SIZE = 8192
    MAX_RETRIES = 5
    BACKOFF_FACTOR = 2
    TIMEOUT = 300

    def __init__(self, config):
        self.config = config
        self._setup_proxies()
        self._setup_auth()
        self.session = self._create_session()

    def _setup_proxies(self):
        http_proxy = getattr(self.config, 'http_proxy', '') or os.getenv('HTTP_PROXY', '')
        https_proxy = getattr(self.config, 'https_proxy', '') or os.getenv('HTTPS_PROXY', '')
        if http_proxy:
            os.environ['HTTP_PROXY'] = http_proxy
        if https_proxy:
            os.environ['HTTPS_PROXY'] = https_proxy

    def _setup_auth(self):
        self.hf_token = getattr(self.config, 'hf_token', '') or os.getenv('HF_TOKEN', '')
        self.kaggle_api_token = getattr(self.config, 'kaggle_api_token', '') or os.getenv('KAGGLE_API_TOKEN', '')
        self.kaggle_username = getattr(self.config, 'kaggle_username', '') or os.getenv('KAGGLE_USERNAME', '')
        self.kaggle_key = getattr(self.config, 'kaggle_key', '') or os.getenv('KAGGLE_KEY', '')

        if self.kaggle_username and self.kaggle_key:
            os.environ['KAGGLE_USERNAME'] = self.kaggle_username
            os.environ['KAGGLE_KEY'] = self.kaggle_key

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=self.BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({'User-Agent': 'TAMER-OCR/2.0 (Dataset Downloader)'})
        return session

    def get_hf_dataset(self, repo_id: str, split: str = "train"):
        try:
            from datasets import load_dataset
            from huggingface_hub import login

            if self.hf_token:
                try:
                    login(token=self.hf_token, add_to_git_credential=True)
                except Exception:
                    pass

            logger.info(f"Loading Hugging Face dataset: {repo_id} ({split})")
            dataset = load_dataset(repo_id, split=split)
            logger.info(f"Successfully loaded HF dataset: {repo_id} ({len(dataset)} samples)")
            return dataset
        except ImportError:
            logger.error("Hugging Face datasets library not installed. Run: pip install datasets")
            return None
        except Exception as e:
            logger.error(f"Hugging Face dataset load failed for {repo_id}: {e}")
            return None

    def download_kaggle(self, dataset_identifier: str, extract_dir: str):
        if self._is_extracted(extract_dir, required_min_files=10):
            logger.info(f"Kaggle dataset already extracted at {extract_dir}")
            return

        kaggle_key = os.environ.get('KAGGLE_KEY', '')
        if not kaggle_key:
            import getpass
            kaggle_key = getpass.getpass("Enter your Kaggle API Key: ").strip()
            os.environ['KAGGLE_KEY'] = kaggle_key

        if not kaggle_key:
            raise ValueError("Kaggle key cannot be empty!")

        logger.info(f"Downloading Kaggle dataset: {dataset_identifier}...")
        try:
            os.makedirs(extract_dir, exist_ok=True)
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset_identifier, "-p", extract_dir, "--unzip"],
                capture_output=True, text=True, check=True
            )
            logger.info(f"Kaggle dataset downloaded to {extract_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Kaggle CLI download failed. StdErr: {e.stderr}")
            raise DownloadError(f"Kaggle download failed: {e.stderr}")

    def download_zenodo_zip(self, url: str, extract_dir: str, verify_ssl: bool = True):
        os.makedirs(extract_dir, exist_ok=True)
        zip_path = os.path.join(extract_dir, "dataset.zip")

        if self._is_extracted(extract_dir, required_min_files=10):
            logger.info(f"Zenodo dataset already extracted at {extract_dir}")
            return

        logger.info(f"Downloading Zenodo ZIP from {url}...")
        try:
            self._download_large_file_with_requests(url, zip_path, verify_ssl=verify_ssl)
            logger.info("Extracting ZIP...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            if os.path.exists(zip_path):
                os.remove(zip_path)
            logger.info(f"Extraction complete at {extract_dir}")
        except Exception as e:
            logger.error(f"Failed to download/extract Zenodo dataset: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise

    def download_github(self, repo_url: str, extract_dir: str):
        if os.path.exists(os.path.join(extract_dir, ".git")):
            logger.info(f"GitHub repository already cloned at {extract_dir}")
            return

        if not shutil.which("git"):
            raise DownloadError("Git is not installed or not found in PATH.")

        logger.info(f"Cloning GitHub repository from {repo_url} (shallow clone)...")
        try:
            os.makedirs(extract_dir, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, extract_dir],
                capture_output=True, text=True, timeout=3600
            )
            if result.returncode == 0:
                logger.info(f"Clone complete: {extract_dir}")
            else:
                raise DownloadError(f"git clone failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise DownloadError("git clone timed out after 1 hour.")
        except Exception as e:
            raise

    def _download_large_file_with_requests(self, url: str, dest_path: str, verify_ssl: bool = True):
        dest = Path(dest_path)
        resume_pos = 0
        if dest.exists():
            resume_pos = dest.stat().st_size

        headers = {}
        if resume_pos > 0:
            headers['Range'] = f'bytes={resume_pos}-'

        response = self.session.get(url, headers=headers, stream=True, timeout=self.TIMEOUT, verify=verify_ssl)
        response.raise_for_status()

        content_length = int(response.headers.get('Content-Length', 0))
        mode = 'ab' if resume_pos > 0 else 'wb'
        downloaded = resume_pos

        with open(dest_path, mode) as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        logger.info(f"Download complete: {dest_path} ({downloaded // (1024*1024)}MB)")

    def _is_extracted(self, path: str, required_min_files: int = 1) -> bool:
        if not os.path.exists(path):
            return False
        count = 0
        for root, dirs, files in os.walk(path):
            count += len(files)
            if count >= required_min_files:
                return True
        return False

    def extract_archive(self, archive_path: str, extract_to: str) -> bool:
        path = Path(archive_path)
        extract_dir = Path(extract_to)
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(extract_dir)
            elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
                with tarfile.open(archive_path, 'r:gz') as tf:
                    tf.extractall(extract_dir)
            elif archive_path.endswith('.tar'):
                with tarfile.open(archive_path, 'r') as tf:
                    tf.extractall(extract_dir)
            else:
                return False
            return True
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False


def create_downloader(config) -> AdvDownloader:
    return AdvDownloader(config)
