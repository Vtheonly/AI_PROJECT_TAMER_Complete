"""
Data Manager for TAMER OCR Training.

Orchestrates downloading, parsing, and normalizing all datasets.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .advanced_downloader import AdvDownloader
from .parser import DatasetParser
from .latex_normalizer import normalize_latex, normalize_corpus

logger = logging.getLogger("TAMER.DataManager")


class DataManager:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.parser = DatasetParser()
        self.downloader = AdvDownloader(config)
        
        self.cache_dir = os.path.join(self.data_dir, ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._stage1_cache = None
        self._stage2_cache = None
        self._stage3_cache = None
    
    def get_stage1_im2latex(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        if self._stage1_cache is not None and not force_refresh:
            return self._stage1_cache
            
        logger.info("Loading Stage 1: Im2LaTeX-100K (Printed)")
        cache_file = os.path.join(self.cache_dir, "stage1_im2latex.json")
        
        if not force_refresh and os.path.exists(cache_file):
            try:
                samples = self._load_cache(cache_file)
                if samples:
                    logger.info(f"Loaded Stage 1 from cache: {len(samples)} samples")
                    self._stage1_cache = samples
                    return samples
            except Exception as e:
                logger.warning(f"Cache load failed, refreshing: {e}")
        
        try:
            extract_dir = os.path.join(self.data_dir, "im2latex-100k")
            
            # Try HuggingFace first
            try:
                logger.info("Attempting Im2LaTeX from HuggingFace mirror...")
                hf_dataset = self.downloader.get_hf_dataset("yuntian-deng/im2latex-100k", split="train")
                if hf_dataset is not None:
                    samples = self.parser.parse_mathwriting(hf_dataset)
                    for s in samples:
                        s['dataset_name'] = 'im2latex'
                    if samples:
                        samples = normalize_corpus(samples)
                        self._save_cache(samples, cache_file)
                        self._stage1_cache = samples
                        return samples
            except Exception as e:
                logger.info(f"HF mirror failed, falling back to Kaggle: {e}")

            self.downloader.download_kaggle("shahrukhkhan/im2latex100k", extract_dir)
            samples = self.parser.parse_im2latex(extract_dir)
            samples = normalize_corpus(samples)
            self._save_cache(samples, cache_file)
            self._stage1_cache = samples
            return samples
        except Exception as e:
            logger.error(f"Could not load Im2LaTeX. Error: {e}")
            self._stage1_cache = []
            return []
    
    def get_stage2_mathwriting(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        if self._stage2_cache is not None and not force_refresh:
            return self._stage2_cache
            
        logger.info("Loading Stage 2: MathWriting (Clean Handwritten)")
        cache_file = os.path.join(self.cache_dir, "stage2_mathwriting.json")
        
        if not force_refresh and os.path.exists(cache_file):
            try:
                samples = self._load_cache(cache_file)
                if samples:
                    self._stage2_cache = samples
                    return samples
            except Exception:
                pass

        try:
            hf_dataset = self.downloader.get_hf_dataset("deepcopy/MathWriting-human", split="train")
            if hf_dataset is None:
                self._stage2_cache = []
                return []
                
            extract_dir = os.path.join(self.data_dir, "mathwriting")
            samples = self.parser.parse_mathwriting(hf_dataset, extract_dir=extract_dir)
            samples = normalize_corpus(samples)
            self._stage2_cache = samples
            self._save_cache(samples, cache_file)
            return samples
        except Exception as e:
            logger.error(f"Could not load MathWriting. Error: {e}")
            self._stage2_cache = []
            return []

    def get_stage3_crohme(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        logger.info("Loading Stage 3a: CROHME (Competition Handwritten)")
        cache_file = os.path.join(self.cache_dir, "stage3_crohme.json")
        
        if not force_refresh and os.path.exists(cache_file):
            try:
                samples = self._load_cache(cache_file)
                if samples:
                    return samples
            except Exception:
                pass
        
        try:
            extract_dir = os.path.join(self.data_dir, "crohme")
            zenodo_url = "https://zenodo.org/records/8428035/files/CROHME23.zip?download=1"
            self.downloader.download_zenodo_zip(zenodo_url, extract_dir)
            samples = self.parser.parse_crohme(extract_dir)
            samples = normalize_corpus(samples)
            self._save_cache(samples, cache_file)
            return samples
        except Exception as e:
            logger.error(f"Could not load CROHME. Error: {e}")
            return []

    def get_stage3_hme100k(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        logger.info("Loading Stage 3b: HME100K (Messy Handwritten)")
        cache_file = os.path.join(self.cache_dir, "stage3_hme100k.json")
        
        if not force_refresh and os.path.exists(cache_file):
            try:
                samples = self._load_cache(cache_file)
                if samples:
                    return samples
            except Exception:
                pass
        
        extract_dir = os.path.join(self.data_dir, "hme100k")
        try:
            self.downloader.download_kaggle("prajwalchy/hme100k-dataset", extract_dir)
            samples = self.parser.parse_hme100k(extract_dir)
            samples = normalize_corpus(samples)
            self._save_cache(samples, cache_file)
            return samples
        except Exception as e:
            logger.error(f"Could not load HME100K. Error: {e}")
            return []

    def get_stage3_combined(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        crohme = self.get_stage3_crohme(force_refresh)
        hme100k = self.get_stage3_hme100k(force_refresh)
        combined = crohme + hme100k
        logger.info(f"Stage 3 combined: {len(combined)} samples ({len(crohme)} CROHME + {len(hme100k)} HME100K)")
        return combined
    
    def load_all_stages(self, force_refresh: bool = False) -> Tuple[List, List, List]:
        logger.info("=" * 60)
        logger.info("Initializing DataManager: Loading all stages")
        logger.info("=" * 60)
        
        s1 = self.get_stage1_im2latex(force_refresh)
        s2 = self.get_stage2_mathwriting(force_refresh)
        s3 = self.get_stage3_combined(force_refresh)
        
        logger.info(f"  Stage 1 (Printed):    {len(s1):>6} samples")
        logger.info(f"  Stage 2 (Clean HW):   {len(s2):>6} samples")
        logger.info(f"  Stage 3 (Comp HW):    {len(s3):>6} samples")
        logger.info(f"  TOTAL:                {len(s1)+len(s2)+len(s3):>6} samples")
        
        return s1, s2, s3
    
    def _load_cache(self, cache_file: str) -> List[Dict[str, Any]]:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_cache(self, samples: List[Dict[str, Any]], cache_file: str):
        try:
            cacheable = [
                s for s in samples 
                if isinstance(s.get('image'), str) and os.path.exists(s.get('image', ''))
            ]
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cacheable, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def clear_cache(self):
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._stage1_cache = None
        self._stage2_cache = None
        self._stage3_cache = None


def create_data_manager(config) -> DataManager:
    return DataManager(config)
