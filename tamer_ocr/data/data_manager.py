"""
Data Manager for TAMER OCR Training.

Orchestrates downloading, parsing, and aggregation of all 4 benchmark datasets:
- CROHME (Zenodo ZIP - Competition Handwritten)
- MathWriting (Hugging Face - Clean Handwritten)
- Im2LaTeX-100K (Kaggle - Printed)
- HME100K (GitHub/HF - Messy Handwritten)

Provides unified access to datasets organized by training stage.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .downloader import AdvDatasetDownloader, AdvDownloader
from .parser import DatasetParser

logger = logging.getLogger("TAMER.DataManager")


class DataManager:
    """
    Orchestrates downloading and parsing across all 4 benchmark datasets.
    
    This manager handles:
    - Download from multiple sources (Zenodo, HuggingFace, Kaggle, GitHub)
    - Authentication for protected sources
    - Proxy configuration for network access
    - Parsing diverse formats into unified structure
    - Caching to avoid re-download/re-parse
    - Error handling and graceful degradation
    
    Usage:
        manager = DataManager(config)
        datasets = manager.load_all_stages()
        # Returns (stage1_data, stage2_data, stage3_data)
    """
    
    def __init__(self, config):
        self.config = config
        self.data_dir = os.path.join(config.data_dir)
        self.parser = DatasetParser()
        
        # Create downloader with proxy/auth support
        self.downloader = AdvDownloader(config)
        
        # Cache directory for parsed samples
        self.cache_dir = os.path.join(self.data_dir, ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Stage-organized datasets
        self._stage1_cache = None
        self._stage2_cache = None
        self._stage3_cache = None
    
    def get_stage1_im2latex(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Stage 1: Printed formula data (Im2LaTeX-100K)
        
        Source: Kaggle dataset
        Returns: List of dicts with 'image' (str path) and 'latex' keys
        """
        if self._stage1_cache is not None and not force_refresh:
            logger.info(f"Using cached Stage 1 data: {len(self._stage1_cache)} samples")
            return self._stage1_cache
            
        logger.info("Loading Stage 1: Im2LaTeX-100K (Printed)")
        
        # Try to load from cache file
        cache_file = os.path.join(self.cache_dir, "stage1_im2latex.json")
        if not force_refresh and os.path.exists(cache_file):
            try:
                self._stage1_cache = self._load_cache(cache_file)
                logger.info(f"Loaded Stage 1 from cache: {len(self._stage1_cache)} samples")
                return self._stage1_cache
            except Exception as e:
                logger.warning(f"Cache load failed, refreshing: {e}")
        
        # Download and parse
        try:
            extract_dir = os.path.join(self.data_dir, "im2latex")
            self.downloader.download_kaggle("shahrukhkhan/im2latex100k", extract_dir)
            samples = self.parser.parse_im2latex(extract_dir)
            
            # Save cache
            self._save_cache(samples, cache_file)
            self._stage1_cache = samples
            
            logger.info(f"Stage 1 complete: {len(samples)} samples")
            return samples
        except Exception as e:
            logger.error(f"Could not load Im2LaTeX. Stage 1 will be empty. Error: {e}")
            self._stage1_cache = []
            return []
    
    def get_stage2_mathwriting(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Stage 2: Clean handwritten data (MathWriting)
        
        Source: Hugging Face dataset
        Returns: List of dicts with 'image' (PIL.Image) and 'latex' keys
        """
        if self._stage2_cache is not None and not force_refresh:
            logger.info(f"Using cached Stage 2 data: {len(self._stage2_cache)} samples")
            return self._stage2_cache
            
        logger.info("Loading Stage 2: MathWriting (Clean Handwritten)")
        
        # Note: HF datasets can't be cached to JSON (PIL Images),
        # but we can cache metadata
        try:
            hf_dataset = self.downloader.get_hf_dataset(
                "deepcopy/MathWriting-human",
                split="train"
            )
            if hf_dataset is None:
                logger.error("Hugging Face returned None dataset")
                self._stage2_cache = []
                return []
                
            samples = self.parser.parse_mathwriting(hf_dataset)
            self._stage2_cache = samples
            
            # Save metadata cache
            meta_file = os.path.join(self.cache_dir, "stage2_mathwriting_meta.json")
            meta = {"count": len(samples), "type": "mathwriting", "has_images": True}
            self._save_cache(meta, meta_file)
            
            logger.info(f"Stage 2 complete: {len(samples)} samples")
            return samples
        except Exception as e:
            logger.error(f"Could not load MathWriting. Stage 2 will be empty. Error: {e}")
            self._stage2_cache = []
            return []
    
    def get_stage3_crohme(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Stage 3: Competition handwritten data (CROHME)
        
        Source: Zenodo ZIP file
        Returns: List of dicts with 'image' (str path) and 'latex' keys
        """
        if self._stage3_cache is not None and not force_refresh:
            # Stage 3 combines CROHME + HME100K
            crohme = self._load_stage3_crohme_cache()
            return crohme
            
        logger.info("Loading Stage 3a: CROHME (Competition Handwritten)")
        
        cache_file = os.path.join(self.cache_dir, "stage3_crohme.json")
        if not force_refresh and os.path.exists(cache_file):
            try:
                samples = self._load_cache(cache_file)
                logger.info(f"Loaded Stage 3 CROHME from cache: {len(samples)} samples")
                return samples
            except Exception as e:
                logger.warning(f"Cache load failed, refreshing: {e}")
        
        # Download and parse
        try:
            extract_dir = os.path.join(self.data_dir, "crohme")
            zenodo_url = "https://zenodo.org/records/8428035/files/CROHME23.zip?download=1"
            self.downloader.download_zenodo_zip(zenodo_url, extract_dir)
            samples = self.parser.parse_crohme(extract_dir)
            
            # Save cache
            self._save_cache(samples, cache_file)
            
            logger.info(f"Stage 3a (CROHME) complete: {len(samples)} samples")
            return samples
        except Exception as e:
            logger.error(f"Could not load CROHME. Stage 3a will be empty. Error: {e}")
            return []
    
    def get_stage3_hme100k(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Stage 3: Messy handwritten data (HME100K)
        
        Sources: Hugging Face mirrors, GitHub fallback
        Returns: List of dicts with 'image' and 'latex' keys
        """
        logger.info("Loading Stage 3b: HME100K (Messy Handwritten)")
        
        # Check for cached HME100K data (path-based samples only)
        cache_file = os.path.join(self.cache_dir, "stage3_hme100k.json")
        if not force_refresh and os.path.exists(cache_file):
            try:
                samples = self._load_cache(cache_file)
                # Verify images still exist
                valid = [s for s in samples if isinstance(s.get('image'), str) and os.path.exists(s['image'])]
                if len(valid) == len(samples):
                    logger.info(f"Loaded Stage 3 HME100K from cache: {len(samples)} samples")
                    return samples
                else:
                    logger.warning(f"Cache has {len(samples) - len(valid)} missing images, refreshing")
            except Exception as e:
                logger.warning(f"Cache load failed, refreshing: {e}")
        
        # Try Hugging Face mirrors first (most reliable)
        hf_mirrors = [
            "Phymond/HME100K",
            "linxy/HME100K",
        ]
        
        for mirror in hf_mirrors:
            try:
                logger.info(f"Attempting HME100K from HF mirror: {mirror}")
                hf_dataset = self.downloader.get_hf_dataset(mirror, split="train")
                if hf_dataset is not None:
                    samples = self.parser.parse_mathwriting(hf_dataset)
                    if samples:
                        # Cache metadata only (PIL images can't be JSON serialized)
                        meta_file = os.path.join(self.cache_dir, "stage3_hme100k_meta.json")
                        self._save_cache({"count": len(samples), "source": mirror}, meta_file)
                        logger.info(f"Stage 3b (HME100K from HF) complete: {len(samples)} samples")
                        return samples
            except Exception as e:
                logger.warning(f"HF mirror {mirror} failed: {e}")
                continue
        
        # Fallback to GitHub clone
        logger.info("Falling back to GitHub clone for HME100K...")
        try:
            extract_dir = os.path.join(self.data_dir, "hme100k")
            self.downloader.download_github(
                "https://github.com/Phymond/HME100K.git",
                extract_dir
            )
            samples = self.parser.parse_hme100k(extract_dir)
            
            # Save cache (path-based samples)
            path_samples = [s for s in samples if isinstance(s.get('image'), str)]
            self._save_cache(path_samples, cache_file)
            
            logger.info(f"Stage 3b (HME100K from GitHub) complete: {len(samples)} samples")
            return samples
        except Exception as e:
            logger.error(f"Could not load HME100K. Stage 3b will be empty. Error: {e}")
            return []
    
    def get_stage3_combined(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get combined Stage 3 data (CROHME + HME100K)."""
        crohme = self.get_stage3_crohme(force_refresh)
        hme100k = self.get_stage3_hme100k(force_refresh)
        combined = crohme + hme100k
        logger.info(f"Stage 3 combined: {len(combined)} samples ({len(crohme)} CROHME + {len(hme100k)} HME100K)")
        return combined
    
    def load_all_stages(self, force_refresh: bool = False) -> Tuple[List, List, List]:
        """
        Load all datasets organized by curriculum stage.
        
        Returns:
            Tuple of (stage1_data, stage2_data, stage3_data) where each is
            a list of dicts with 'image' and 'latex' keys.
        """
        logger.info("=" * 60)
        logger.info("Initializing DataManager: Loading all stages")
        logger.info("=" * 60)
        
        s1 = self.get_stage1_im2latex(force_refresh)
        s2 = self.get_stage2_mathwriting(force_refresh)
        s3 = self.get_stage3_combined(force_refresh)
        
        logger.info("=" * 60)
        logger.info(f"Data Loading Summary:")
        logger.info(f"  Stage 1 (Printed):    {len(s1):>6} samples")
        logger.info(f"  Stage 2 (Clean HW):   {len(s2):>6} samples")
        logger.info(f"  Stage 3 (Comp HW):    {len(s3):>6} samples")
        logger.info(f"  TOTAL:                {len(s1)+len(s2)+len(s3):>6} samples")
        logger.info("=" * 60)
        
        return s1, s2, s3
    
    # -----------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about all datasets."""
        stats = {
            "stages": {},
            "total_samples": 0,
        }
        
        for stage_name, loader in [
            ("stage1_im2latex", self.get_stage1_im2latex),
            ("stage2_mathwriting", self.get_stage2_mathwriting),
            ("stage3_crohme", self.get_stage3_crohme),
            ("stage3_hme100k", self.get_stage3_hme100k),
        ]:
            try:
                data = loader()
                count = len(data)
                stats["stages"][stage_name] = count
                stats["total_samples"] += count
                
                if count > 0:
                    # Calculate average complexity
                    complexities = [self._compute_complexity(s.get('latex', '')) for s in data[:1000]]
                    stats[f"{stage_name}_avg_complexity"] = sum(complexities) / len(complexities) if complexities else 0
            except Exception as e:
                stats["stages"][stage_name] = 0
                logger.warning(f"Could not get stats for {stage_name}: {e}")
        
        return stats
    
    @staticmethod
    def _compute_complexity(latex: str) -> int:
        """Compute structural complexity of a LaTeX formula."""
        return latex.count('{') + latex.count('\\frac') * 2 + latex.count('^') * 2
    
    def _load_cache(self, cache_file: str) -> List[Dict[str, Any]]:
        """Load samples from JSON cache file."""
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_cache(self, samples: List[Dict[str, Any]], cache_file: str):
        """Save samples to JSON cache file."""
        try:
            # Only cache path-based samples (not PIL images)
            cacheable = [
                s for s in samples 
                if isinstance(s.get('image'), str) and os.path.exists(s.get('image', ''))
            ] or samples
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cacheable, f, ensure_ascii=False)
            logger.debug(f"Saved cache: {cache_file} ({len(cacheable)} items)")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _load_stage3_crohme_cache(self) -> List[Dict[str, Any]]:
        """Load cached Stage 3 CROHME data."""
        cache_file = os.path.join(self.cache_dir, "stage3_crohme.json")
        if os.path.exists(cache_file):
            return self._load_cache(cache_file)
        return []
    
    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._stage1_cache = None
        self._stage2_cache = None
        self._stage3_cache = None
        logger.info("Cache cleared")
    
    def verify_datasets(self) -> Dict[str, bool]:
        """Verify all datasets are accessible and have samples."""
        results = {}
        
        for name, loader in [
            ("im2latex", self.get_stage1_im2latex),
            ("mathwriting", self.get_stage2_mathwriting),
            ("crohme", self.get_stage3_crohme),
            ("hme100k", self.get_stage3_hme100k),
        ]:
            try:
                data = loader()
                results[name] = len(data) > 0
                logger.info(f"Verify {name}: {'OK' if results[name] else 'EMPTY'} ({len(data)} samples)")
            except Exception as e:
                results[name] = False
                logger.error(f"Verify {name}: FAILED ({e})")
        
        return results


def create_data_manager(config) -> DataManager:
    """Factory function to create a DataManager instance."""
    return DataManager(config)