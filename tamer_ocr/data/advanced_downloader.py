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
from pathlib import Path
from typing import Optional, List, Dict, Callable
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

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
    
    Features:
    - Hugging Face datasets with token authentication
    - Kaggle API with credential authentication
    - Zenodo direct ZIP streaming
    - GitHub repository cloning
    - HTTP/HTTPS proxy configuration
    - Download resume and retry logic
    
    Usage:
        downloader = AdvDownloader(config)
        downloader.download_kaggle("user/dataset", "./data/extract")
        downloader.get_hf_dataset("user/dataset", split="train")
        downloader.download_zenodo_zip(url, "./data/extract")
        downloader.download_github(url, "./data/repo")
    """
    
    CHUNK_SIZE = 8192  # 8KB chunks
    MAX_RETRIES = 5
    BACKOFF_FACTOR = 2  # seconds
    TIMEOUT = 300  # 5 minutes timeout for downloads
    
    def __init__(self, config):
        """
        Initialize downloader with configuration.
        
        Args:
            config: Configuration object with optional attributes:
                - hf_token: Hugging Face authentication token
                - kaggle_username: Kaggle API username
                - kaggle_key: Kaggle API key
                - http_proxy: HTTP proxy URL
                - https_proxy: HTTPS proxy URL
        """
        self.config = config
        
        # Configure proxies
        self._setup_proxies()
        
        # Configure authentication
        self._setup_auth()

    def _setup_proxies(self):
        """Configure HTTP/HTTPS proxies from config."""
        http_proxy = getattr(self.config, 'http_proxy', '') or os.getenv('HTTP_PROXY', '')
        https_proxy = getattr(self.config, 'https_proxy', '') or os.getenv('HTTPS_PROXY', '')
        
        if http_proxy:
            os.environ['HTTP_PROXY'] = http_proxy
            logger.info(f"HTTP Proxy configured: {http_proxy}")
        if https_proxy:
            os.environ['HTTPS_PROXY'] = https_proxy
            logger.info(f"HTTPS Proxy configured: {https_proxy}")

    def _setup_auth(self):
        """Configure authentication for external services."""
        # Hugging Face token
        self.hf_token = getattr(self.config, 'hf_token', '') or os.getenv('HF_TOKEN', '')
        if self.hf_token:
            logger.info("Hugging Face token configured")
            
        # Kaggle credentials
        self.kaggle_username = getattr(self.config, 'kaggle_username', '') or os.getenv('KAGGLE_USERNAME', '')
        self.kaggle_key = getattr(self.config, 'kaggle_key', '') or os.getenv('KAGGLE_KEY', '')
        
        if self.kaggle_username and self.kaggle_key:
            os.environ['KAGGLE_USERNAME'] = self.kaggle_username
            os.environ['KAGGLE_KEY'] = self.kaggle_key
            logger.info("Kaggle credentials configured")
        
    # -----------------------------------------------------------------
    # Hugging Face Download
    # -----------------------------------------------------------------
    def get_hf_dataset(self, repo_id: str, split: str = "train"):
        """
        Download and load a Hugging Face dataset.
        
        Args:
            repo_id: Dataset repository ID (e.g., "user/dataset-name")
            split: Dataset split to load ("train", "test", etc.)
            
        Returns:
            Hugging Face Dataset object or None on failure
        """
        try:
            from datasets import load_dataset
            from huggingface_hub import login
            
            # Login if token provided
            if self.hf_token:
                try:
                    login(token=self.hf_token, add_to_git_credential=True)
                    logger.info(f"Logged into Hugging Face with token")
                except Exception as e:
                    logger.warning(f"Hugging Face login failed: {e}")
            
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

    # -----------------------------------------------------------------
    # Kaggle Download
    # -----------------------------------------------------------------
    def download_kaggle(self, dataset_identifier: str, extract_dir: str):
        """
        Download a dataset via the Kaggle API.
        
        Args:
            dataset_identifier: Kaggle dataset identifier (e.g., "user/dataset-name")
            extract_dir: Directory to extract files to
            
        Raises:
            ValueError: If Kaggle credentials are missing
            DownloadError: If download fails
        """
        if self._is_extracted(extract_dir, required_min_files=10):
            logger.info(f"Kaggle dataset already extracted at {extract_dir}")
            return
            
        if not self.kaggle_username or not self.kaggle_key:
            logger.error("KAGGLE_USERNAME and KAGGLE_KEY must be provided in config or env.")
            raise ValueError("Missing Kaggle Credentials. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        
        logger.info(f"Downloading Kaggle dataset: {dataset_identifier}")
        
        try:
            import kaggle
            kaggle.api.authenticate()
            
            os.makedirs(extract_dir, exist_ok=True)
            kaggle.api.dataset_download_files(
                dataset_identifier, 
                path=extract_dir, 
                unzip=True, 
                quiet=False
            )
            logger.info(f"Kaggle dataset downloaded and extracted to {extract_dir}")
            
        except ImportError:
            logger.error("Kaggle API library not installed. Run: pip install kaggle")
            raise DownloadError("Kaggle package not installed")
        except Exception as e:
            logger.error(f"Kaggle download failed: {e}")
            raise DownloadError(f"Kaggle download failed: {e}")

    # -----------------------------------------------------------------
    # Zenodo ZIP Download
    # -----------------------------------------------------------------
    def download_zenodo_zip(self, url: str, extract_dir: str):
        """
        Stream a ZIP from a direct URL (e.g., Zenodo) and extract it.
        
        Args:
            url: Direct download URL for the ZIP file
            extract_dir: Directory to extract files to
        """
        os.makedirs(extract_dir, exist_ok=True)
        zip_path = os.path.join(extract_dir, "dataset.zip")
        
        if self._is_extracted(extract_dir, required_min_files=10):
            logger.info(f"Zenodo dataset already extracted at {extract_dir}")
            return
            
        logger.info(f"Downloading Zenodo ZIP from {url}...")
        
        try:
            self._download_with_progress(url, zip_path)
            
            logger.info("Extracting ZIP...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Clean up ZIP file
            if os.path.exists(zip_path):
                os.remove(zip_path)
                
            logger.info(f"Extraction complete at {extract_dir}")
            
        except Exception as e:
            logger.error(f"Failed to download/extract Zenodo dataset: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise

    # -----------------------------------------------------------------
    # GitHub Clone
    # -----------------------------------------------------------------
    def download_github(self, repo_url: str, extract_dir: str):
        """
        Clone a GitHub repository.
        
        Args:
            repo_url: GitHub repository URL
            extract_dir: Directory to clone into
        """
        if os.path.exists(os.path.join(extract_dir, ".git")):
            logger.info(f"GitHub repository already cloned at {extract_dir}")
            return
            
        logger.info(f"Cloning GitHub repository from {repo_url}...")
        
        try:
            os.makedirs(extract_dir, exist_ok=True)
            # Use git command for cloning
            result = os.system(f"git clone {repo_url} {extract_dir} 2>&1")
            if result == 0:
                logger.info("Clone complete.")
            else:
                raise DownloadError(f"git clone failed with exit code {result}")
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            raise

    # -----------------------------------------------------------------
    # Generic Download with Progress
    # -----------------------------------------------------------------
    def _download_with_progress(self, url: str, dest_path: str):
        """Download a file with progress tracking and retry logic."""
        dest = Path(dest_path)
        
        # Check for resume
        resume_pos = 0
        if dest.exists():
            resume_pos = dest.stat().st_size
            logger.info(f"Resuming download from {resume_pos} bytes")
        
        for retry in range(self.MAX_RETRIES):
            try:
                req = Request(url)
                req.add_header('User-Agent', 'TAMER-OCR/1.0 (Dataset Downloader)')
                
                if resume_pos > 0:
                    req.add_header('Range', f'bytes={resume_pos}-')
                
                with urlopen(req, timeout=self.TIMEOUT) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                    
                    mode = 'ab' if resume_pos > 0 else 'wb'
                    with open(dest_path, mode) as f:
                        downloaded = resume_pos
                        while True:
                            chunk = response.read(self.CHUNK_SIZE)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Log progress every 1MB
                            if downloaded % (1024 * 1024) < self.CHUNK_SIZE:
                                progress = (downloaded / max(1, total_size)) * 100 if total_size > 0 else 0
                                logger.info(f"  Downloaded: {downloaded // (1024*1024)}MB ({progress:.1f}%)")
                                
                logger.info(f"Download complete: {dest_path}")
                return
                
            except (URLError, HTTPError, ConnectionError) as e:
                backoff = self.BACKOFF_FACTOR * (2 ** retry)
                logger.warning(f"Download attempt {retry+1}/{self.MAX_RETRIES} failed: {e}")
                logger.info(f"  Retrying in {backoff}s...")
                time.sleep(backoff)
            except Exception as e:
                logger.error(f"Unexpected error during download: {e}")
                raise
                
        raise DownloadError(f"Failed to download after {self.MAX_RETRIES} attempts")

    # -----------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------
    def _is_extracted(self, path: str, required_min_files: int = 1) -> bool:
        """Check if a directory contains the minimum number of files."""
        if not os.path.exists(path):
            return False
        count = 0
        for root, dirs, files in os.walk(path):
            count += len(files)
            if count >= required_min_files:
                return True
        return False
    
    def extract_archive(self, archive_path: str, extract_to: str) -> bool:
        """Extract an archive (zip, tar, tar.gz)."""
        path = Path(archive_path)
        extract_dir = Path(extract_to)
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting {path.name} to {extract_dir}")
        
        try:
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(extract_dir)
                    logger.info(f"Extracted {len(zf.namelist())} files")
            elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
                with tarfile.open(archive_path, 'r:gz') as tf:
                    tf.extractall(extract_dir)
                    logger.info(f"Extracted {len(tf.getmembers())} files")
            elif archive_path.endswith('.tar'):
                with tarfile.open(archive_path, 'r') as tf:
                    tf.extractall(extract_dir)
                    logger.info(f"Extracted {len(tf.getmembers())} files")
            else:
                logger.warning(f"Unknown archive format: {archive_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False

    def check_disk_space(self, required_bytes: int) -> bool:
        """Check if there's enough disk space for a download."""
        import shutil
        data_dir = getattr(self.config, 'data_dir', './data')
        available = shutil.disk_usage(data_dir).free
        if available < required_bytes * 1.1:  # 10% buffer
            logger.error(
                f"Insufficient disk space! "
                f"Required: {required_bytes // (1024*1024)}MB, "
                f"Available: {available // (1024*1024)}MB"
            )
            return False
        return True


def create_downloader(config) -> AdvDownloader:
    """Factory function to create an AdvDownloader instance."""
    return AdvDownloader(config)