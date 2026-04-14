"""
Dataset Preprocessor — Strict Pipeline:
1. Download datasets
2. Preprocess ALL 4 datasets completely
3. Verify integrity
4. Create/use HuggingFace dataset repo
5. Push full processed dataset to HF
6. ONLY THEN allow training to proceed
"""

import os
import json
import gc
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import Config


class DatasetPreprocessor:
    """Handles the full download → preprocess → verify → push pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.raw_dir = os.path.join(config.data_dir, "raw")
        self.processed_dir = os.path.join(config.data_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    # ─── Public API ────────────────────────────────────────

    def run(self):
        """Execute the full strict pipeline. Training must NOT start before this completes."""
        print("=" * 60)
        print("STARTING STRICT PREPROCESSING PIPELINE")
        print("=" * 60)

        datasets = self.config.datasets  # list of dataset configs

        # Step 1: Download all datasets
        print("\n[STEP 1/5] Downloading datasets...")
        self._download_all(datasets)

        # Step 2: Preprocess all datasets
        print("\n[STEP 2/5] Preprocessing datasets...")
        self._preprocess_all(datasets)

        # Step 3: Verify integrity
        print("\n[STEP 3/5] Verifying processed datasets...")
        self._verify_all(datasets)

        # Step 4 & 5: Push to HuggingFace
        print("\n[STEP 4/5] Pushing to HuggingFace...")
        self._push_to_hf()

        print("\n" + "=" * 60)
        print("PREPROCESSING PIPELINE COMPLETE — Ready for training")
        print("=" * 60)

    # ─── Download ──────────────────────────────────────────

    def _download_all(self, datasets):
        for ds in datasets:
            name = ds.get("name", "unknown")
            print(f"\n  Downloading {name}...")
            ds_raw = os.path.join(self.raw_dir, name)
            os.makedirs(ds_raw, exist_ok=True)

            source_type = ds.get("type", "kaggle")

            if source_type == "kaggle":
                self._download_kaggle(ds, ds_raw)
            elif source_type == "huggingface":
                self._download_hf(ds, ds_raw)
            elif source_type == "url":
                self._download_url(ds, ds_raw)
            else:
                print(f"    Unknown source type '{source_type}' for {name}, skipping download")

            print(f"  ✓ {name} downloaded")

    def _download_kaggle(self, ds: dict, dest: str):
        """Download from Kaggle using kaggle CLI."""
        slug = ds.get("kaggle_slug", "")
        if not slug:
            print("    No kaggle_slug specified, skipping")
            return
        os.system(f"kaggle datasets download -d {slug} -p {dest} --unzip")

    def _download_hf(self, ds: dict, dest: str):
        """Download from HuggingFace datasets."""
        repo = ds.get("hf_repo", "")
        if not repo:
            print("    No hf_repo specified, skipping")
            return
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo, local_dir=dest, repo_type="dataset")
        except Exception as e:
            print(f"    HF download failed: {e}")

    def _download_url(self, ds: dict, dest: str):
        """Download from direct URL."""
        url = ds.get("url", "")
        if not url:
            print("    No url specified, skipping")
            return
        fname = url.split("/")[-1]
        fpath = os.path.join(dest, fname)
        os.system(f"wget -q -O {fpath} {url}")
        if fpath.endswith(".zip"):
            os.system(f"unzip -o {fpath} -d {dest}")
        elif fpath.endswith(".tar.gz") or fpath.endswith(".tgz"):
            os.system(f"tar xzf {fpath} -C {dest}")

    # ─── Preprocess ────────────────────────────────────────

    def _preprocess_all(self, datasets):
        from ..data.latex_normalizer import LaTeXNormalizer
        from ..data.tokenizer import Tokenizer

        normalizer = LaTeXNormalizer()
        tokenizer = Tokenizer(self.config)

        for ds in datasets:
            name = ds.get("name", "unknown")
            print(f"\n  Preprocessing {name}...")
            ds_raw = os.path.join(self.raw_dir, name)
            ds_proc = os.path.join(self.processed_dir, name)
            os.makedirs(ds_proc, exist_ok=True)

            try:
                parser_name = ds.get("parser", name.lower())
                samples = self._parse_dataset(parser_name, ds_raw, normalizer, tokenizer)

                # Save processed samples
                out_file = os.path.join(ds_proc, "data.json")
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(samples, f, ensure_ascii=False, indent=2)

                print(f"  ✓ {name}: {len(samples)} samples preprocessed")
            except Exception as e:
                print(f"  ✗ {name} preprocessing FAILED: {e}")
                raise

            # Free memory between datasets
            gc.collect()

    def _parse_dataset(self, parser_name: str, raw_dir: str, normalizer, tokenizer) -> List[dict]:
        """Parse a dataset using the appropriate parser."""
        try:
            from ..data.parser import CROHMEParser, Im2LaTeXParser, HME100KParser
        except ImportError:
            # Fallback: try importing individually
            from ..data.parser import CROHMEParser, Im2LaTeXParser, HME100KParser

        parsers = {
            "crohme": CROHMEParser,
            "im2latex": Im2LaTeXParser,
            "hme100k": HME100KParser,
        }

        # MathWriting uses same parser as CROHME (INKML format)
        parser_cls = parsers.get(parser_name, CROHMEParser)
        parser = parser_cls(normalizer=normalizer, tokenizer=tokenizer)

        samples = parser.parse(raw_dir)
        return samples

    # ─── Verify ────────────────────────────────────────────

    def _verify_all(self, datasets):
        all_ok = True
        for ds in datasets:
            name = ds.get("name", "unknown")
            ds_proc = os.path.join(self.processed_dir, name)
            data_file = os.path.join(ds_proc, "data.json")

            if not os.path.isfile(data_file):
                print(f"  ✗ {name}: data.json NOT FOUND")
                all_ok = False
                continue

            with open(data_file, "r", encoding="utf-8") as f:
                samples = json.load(f)

            if not isinstance(samples, list) or len(samples) == 0:
                print(f"  ✗ {name}: data.json is empty or invalid")
                all_ok = False
                continue

            # Check each sample has required fields
            bad = [s for s in samples if "image_path" not in s or "tokens" not in s]
            if bad:
                print(f"  ✗ {name}: {len(bad)} samples missing required fields")
                all_ok = False
                continue

            print(f"  ✓ {name}: {len(samples)} valid samples")

        if not all_ok:
            raise RuntimeError("Dataset verification FAILED — fix issues before training")

    # ─── Push to HuggingFace ───────────────────────────────

    def _push_to_hf(self):
        """Create/use HF dataset repo and push processed data."""
        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=self.config.hf_token)
        repo_id = self.config.hf_dataset_repo_id

        # Create repo if it doesn't exist
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", token=self.config.hf_token, exist_ok=True)
            print(f"  HF dataset repo ready: {repo_id}")
        except Exception as e:
            print(f"  Could not create HF repo (may already exist): {e}")

        # Upload processed directory
        try:
            api.upload_folder(
                folder_path=self.processed_dir,
                repo_id=repo_id,
                repo_type="dataset",
                token=self.config.hf_token,
            )
            print(f"  ✓ Processed dataset pushed to {repo_id}")
        except Exception as e:
            print(f"  ✗ Push to HF FAILED: {e}")
            raise