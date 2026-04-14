"""
Data Manager for TAMER OCR Training.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .downloader import AdvDatasetDownloader
from .advanced_downloader import AdvDownloader
from .parser import DatasetParser

logger = logging.getLogger("TAMER.DataManager")


class DataManager:
    def __init__(self, config):
        self.config = config
        self.data_dir = os.path.join(config.data_dir)
        self.parser = DatasetParser()
        self.downloader = AdvDownloader(config)
        
        self.cache_dir = os.path.join(self.data_dir, ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._stage1_cache = None
        self._stage2_cache = None
        self._stage3_cache = None
    
    def get_stage1_im2latex(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        if self._stage1_cache is not None and not force_refresh:
            logger.info(f"Using cached Stage 1 data: {len(self._stage1_cache)} samples")
            return self._stage1_cache
            
        logger.info("Loading Stage 1: Im2LaTeX-100K (Printed)")
        
        cache_file = os.path.join(self.cache_dir, "stage1_im2latex.json")
        if not force_refresh and os.path.exists(cache_file):
            try:
                self._stage1_cache = self._load_cache(cache_file)
                if isinstance(self._stage1_cache, list) and self._stage1_cache:
                    logger.info(f"Loaded Stage 1 from cache: {len(self._stage1_cache)} samples")
                    return self._stage1_cache
            except Exception as e:
                logger.warning(f"Cache load failed, refreshing: {e}")
        
        try:
            extract_dir = os.path.join(self.data_dir, "im2latex-100k")
            
            try:
                logger.info("Attempting Im2LaTeX from HuggingFace mirror (yuntian-deng/im2latex-100k)...")
                hf_dataset = self.downloader.get_hf_dataset("yuntian-deng/im2latex-100k", split="train")
                if hf_dataset is not None:
                    samples = self.parser.parse_mathwriting(hf_dataset)
                    if samples:
                        self._save_cache(samples, cache_file)
                        self._stage1_cache = samples
                        logger.info(f"Stage 1 complete via HF: {len(samples)} samples")
                        return samples
            except Exception as e:
                logger.info(f"HF mirror failed, falling back to Kaggle: {e}")

            self.downloader.download_kaggle("shahrukhkhan/im2latex100k", extract_dir)
            samples = self.parser.parse_im2latex(extract_dir)
            
            self._save_cache(samples, cache_file)
            self._stage1_cache = samples
            
            logger.info(f"Stage 1 complete: {len(samples)} samples")
            return samples
        except Exception as e:
            logger.error(f"Could not load Im2LaTeX. Stage 1 will be empty. Error: {e}")
            self._stage1_cache = []
            return []
    
    def get_stage2_mathwriting(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        if self._stage2_cache is not None and not force_refresh:
            logger.info(f"Using cached Stage 2 data: {len(self._stage2_cache)} samples")
            return self._stage2_cache
            
        logger.info("Loading Stage 2: MathWriting (Clean Handwritten)")
        
        cache_file = os.path.join(self.cache_dir, "stage2_mathwriting.json")
        if not force_refresh and os.path.exists(cache_file):
            try:
                self._stage2_cache = self._load_cache(cache_file)
                if isinstance(self._stage2_cache, list) and self._stage2_cache:
                    logger.info(f"Loaded Stage 2 from cache: {len(self._stage2_cache)} samples")
                    return self._stage2_cache
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")

        try:
            hf_dataset = self.downloader.get_hf_dataset(
                "deepcopy/MathWriting-human",
                split="train"
            )
            if hf_dataset is None:
                logger.error("Hugging Face returned None dataset")
                self._stage2_cache = []
                return []
                
            extract_dir = os.path.join(self.data_dir, "mathwriting")
            samples = self.parser.parse_mathwriting(hf_dataset, extract_dir=extract_dir)
            self._stage2_cache = samples
            
            self._save_cache(samples, cache_file)
            
            logger.info(f"Stage 2 complete: {len(samples)} samples")
            return samples
        except Exception as e:
            logger.error(f"Could not load MathWriting. Stage 2 will be empty. Error: {e}")
            self._stage2_cache = []
            return []
 
    def get_stage3_crohme(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        if self._stage3_cache is not None and not force_refresh:
            crohme = self._load_stage3_crohme_cache()
            return crohme
            
        logger.info("Loading Stage 3a: CROHME (Competition Handwritten)")
        cache_file = os.path.join(self.cache_dir, "stage3_crohme.json")
        
        if not force_refresh and os.path.exists(cache_file):
            try:
                samples = self._load_cache(cache_file)
                if samples:
                    logger.info(f"Loaded Stage 3 CROHME from cache: {len(samples)} samples")
                    return samples
            except Exception as e:
                logger.warning(f"Cache load failed, refreshing: {e}")
        
        try:
            extract_dir = os.path.join(self.data_dir, "crohme")
            zenodo_url = "https://zenodo.org/records/8428035/files/CROHME23.zip?download=1"
            self.downloader.download_zenodo_zip(zenodo_url, extract_dir)
            
            samples = self.parser.parse_crohme(extract_dir)
            
            self._save_cache(samples, cache_file)
            logger.info(f"Stage 3a (CROHME) complete: {len(samples)} samples")
            return samples
        except Exception as e:
            logger.error(f"Could not load CROHME. Stage 3a will be empty. Error: {e}")
            return []
 
    def get_stage3_hme100k(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        logger.info("Loading Stage 3b: HME100K (Messy Handwritten)")
        cache_file = os.path.join(self.cache_dir, "stage3_hme100k.json")
        
        if not force_refresh and os.path.exists(cache_file):
            try:
                samples = self._load_cache(cache_file)
                valid = [s for s in samples if isinstance(s.get('image'), str) and os.path.exists(s['image'])]
                if len(valid) == len(samples) and len(samples) > 0:
                    logger.info(f"Loaded Stage 3 HME100K from cache: {len(samples)} samples")
                    return samples
                elif len(samples) > 0:
                    logger.warning(f"Cache has {len(samples) - len(valid)} missing images, refreshing...")
            except Exception as e:
                logger.warning(f"Cache load failed, refreshing: {e}")
        
        extract_dir = os.path.join(self.data_dir, "hme100k")
        
        logger.info("Attempting to download HME100K from Kaggle mirror (prajwalchy/hme100k-dataset)...")
        try:
            self.downloader.download_kaggle("prajwalchy/hme100k-dataset", extract_dir)
            
            logger.info("Parsing HME100K dataset from Kaggle files...")
            samples = self.parser.parse_hme100k(extract_dir)
            
            if samples:
                self._save_cache(samples, cache_file)
                logger.info(f"Stage 3b (HME100K) complete: {len(samples)} samples")
                return samples
            else:
                logger.error("HME100K downloaded but no valid samples were found during parsing.")
                return []
                
        except Exception as e:
            logger.error(f"Could not load HME100K from Kaggle. Stage 3b will be empty. Error: {e}")
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
        
        logger.info("=" * 60)
        logger.info(f"Data Loading Summary:")
        logger.info(f"  Stage 1 (Printed):    {len(s1):>6} samples")
        logger.info(f"  Stage 2 (Clean HW):   {len(s2):>6} samples")
        logger.info(f"  Stage 3 (Comp HW):    {len(s3):>6} samples")
        logger.info(f"  TOTAL:                {len(s1)+len(s2)+len(s3):>6} samples")
        logger.info("=" * 60)
        
        return s1, s2, s3
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
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
                    complexities = [self._compute_complexity(s.get('latex', '')) for s in data[:1000]]
                    stats[f"{stage_name}_avg_complexity"] = sum(complexities) / len(complexities) if complexities else 0
            except Exception as e:
                stats["stages"][stage_name] = 0
                logger.warning(f"Could not get stats for {stage_name}: {e}")
        
        return stats
    
    @staticmethod
    def _compute_complexity(latex: str) -> int:
        return latex.count('{') + latex.count('\\frac') * 2 + latex.count('^') * 2
    
    def _load_cache(self, cache_file: str) -> List[Dict[str, Any]]:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # FIX: Ensure we always return a list. If old buggy cache is loaded, return []
            if isinstance(data, dict):
                logger.warning(f"Found corrupted/dict cache in {cache_file}. Ignoring.")
                return []
            return data
    
    def _save_cache(self, samples: List[Dict[str, Any]], cache_file: str):
        try:
            cacheable = [
                s for s in samples 
                if isinstance(s.get('image'), str) and os.path.exists(s.get('image', ''))
            ]
            
            # FIX: Never save a dict. Always save a list.
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cacheable, f, ensure_ascii=False)
            logger.debug(f"Saved cache: {cache_file} ({len(cacheable)} items)")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _load_stage3_crohme_cache(self) -> List[Dict[str, Any]]:
        cache_file = os.path.join(self.cache_dir, "stage3_crohme.json")
        if os.path.exists(cache_file):
            return self._load_cache(cache_file)
        return []
    
    def clear_cache(self):
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._stage1_cache = None
        self._stage2_cache = None
        self._stage3_cache = None
        logger.info("Cache cleared")
    
    def verify_datasets(self) -> Dict[str, bool]:
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
    return DataManager(config)