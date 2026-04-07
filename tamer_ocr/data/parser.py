"""
Dataset Parser for TAMER OCR Training.

Parses diverse dataset formats (CROHME, MathWriting, Im2LaTeX-100K, HME100K)
into a unified format that feeds into TreeMathDataset.

Each parser returns a list of dicts with keys:
  - 'image': PIL.Image.Image or str (path to image)
  - 'latex': str (LaTeX formula string)
"""

import os
import glob
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import csv

logger = logging.getLogger("TAMER.Parser")


class DatasetParser:
    """
    Unifies different datasets into List[Dict[str, Any]] containing 'image' and 'latex'.
    
    Supports:
    - CROHME (Zenodo ZIP with images + label files)
    - MathWriting (Hugging Face dataset with PIL images)
    - Im2LaTeX-100K (Kaggle with CSV + images)
    - HME100K (GitHub with images + labels)
    """

    # Common image extensions to search for
    IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')

    def _find_images(self, root_dir: str) -> List[str]:
        """Recursively find all image files in a directory."""
        images = []
        for ext in self.IMG_EXTENSIONS:
            images.extend(glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True))
            images.extend(glob.glob(os.path.join(root_dir, f"**/*{ext.upper()}"), recursive=True))
        return sorted(set(images))

    def _find_label_file(self, root_dir: str, preferred_names: List[str] = None) -> Optional[str]:
        """Find a label/annotation text file."""
        preferred = preferred_names or ['train.txt', 'labels.txt', 'annotation.txt', 'formula.txt']
        
        # Search for preferred names first
        for name in preferred:
            for root, dirs, files in os.walk(root_dir):
                if name.lower() in [f.lower() for f in files]:
                    return os.path.join(root, name)
        
        # Fallback: find any .txt file
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith('.txt'):
                    return os.path.join(root, f)
        return None

    def _safe_read_latex(self, path: str) -> str:
        """Safely read a LaTeX string from a file."""
        try:
            for encoding in ['utf-8', 'latin-1', 'ascii', 'utf-8-sig']:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        return f.read().strip()
                except UnicodeDecodeError:
                    continue
            logger.warning(f"Could not read file with any encoding: {path}")
            return ""
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            return ""

    def _validate_sample(self, img_source: Any, latex: str) -> bool:
        """Check if a sample is valid."""
        if not latex or not latex.strip():
            return False
        if isinstance(img_source, str):
            return os.path.exists(img_source)
        if isinstance(img_source, Image.Image):
            return True
        return False

    # -----------------------------------------------------------------
    # CROHME Parser
    # -----------------------------------------------------------------
    def parse_crohme(self, extract_dir: str) -> List[Dict[str, Any]]:
        """
        Parse CROHME dataset from Zenodo extraction directory.
        
        CROHME typically provides:
        - Images (.png, .inkml) in various subdirectories
        - Label files (.txt) with LaTeX annotations
        - Sometimes images and labels are side-by-side with same basename
        """
        logger.info(f"Parsing CROHME from {extract_dir}")
        samples = []

        if not os.path.exists(extract_dir):
            logger.error(f"CROHME directory not found: {extract_dir}")
            return samples

        # Strategy 1: Find .txt label files and match with images
        label_files = []
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith('.txt'):
                    label_files.append(os.path.join(root, f))

        matched = 0
        for label_file in label_files:
            label_dir = os.path.dirname(label_file)
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Format: filename\tlatex or filename latex or latex-only
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            img_name = parts[0].strip()
                            latex = parts[1].strip()
                        else:
                            img_name = line.split()[0] if line.split() else line
                            latex = ' '.join(line.split()[1:]) if len(line.split()) > 1 else line

                        # Try to find the image
                        img_path = self._find_matching_image(label_dir, img_name)
                        if img_path and latex:
                            samples.append({"image": img_path, "latex": latex})
                            matched += 1
            except Exception as e:
                logger.warning(f"Error parsing label file {label_file}: {e}")

        # Strategy 2: If no labels found, try matching image files with .txt files of same name
        if matched == 0:
            images = self._find_images(extract_dir)
            # Filter to non-inkml images (prefer PNG)
            images = [img for img in images if not img.lower().endswith('.inkml')]
            
            for img_path in images:
                # Find matching .txt file
                base = os.path.splitext(img_path)[0]
                txt_path = base + ".txt"
                if os.path.exists(txt_path):
                    latex = self._safe_read_latex(txt_path)
                    if latex and self._validate_sample(img_path, latex):
                        samples.append({"image": img_path, "latex": latex})
                        matched += 1
                else:
                    # Try searching for any .txt with same basename
                    for root, dirs, files in os.walk(extract_dir):
                        for f in files:
                            if f.lower().endswith('.txt') and os.path.splitext(f)[0].lower() == os.path.splitext(os.path.basename(img_path))[0].lower():
                                txt_path = os.path.join(root, f)
                                latex = self._safe_read_latex(txt_path)
                                if latex and self._validate_sample(img_path, latex):
                                    samples.append({"image": img_path, "latex": latex})
                                    matched += 1
                                break

        # Strategy 3: Check for annotations.json
        if matched == 0:
            samples.extend(self._parse_annotations_json(extract_dir))
            matched = len(samples)

        logger.info(f"CROHME parsed: {matched} valid samples found.")
        return samples

    def _find_matching_image(self, search_dir: str, img_name: str) -> Optional[str]:
        """Find an image file matching the given name."""
        if not img_name:
            return None
        # Direct match
        direct = os.path.join(search_dir, img_name)
        if os.path.exists(direct) and any(direct.lower().endswith(ext) for ext in self.IMG_EXTENSIONS):
            return direct

        # Search recursively
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if f.lower() == img_name.lower():
                    return os.path.join(root, f)
        return None

    def _parse_annotations_json(self, root_dir: str) -> List[Dict[str, Any]]:
        """Parse an annotations.json file if it exists."""
        samples = []
        json_path = os.path.join(root_dir, "annotations.json")
        
        if not os.path.exists(json_path):
            return samples
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            if isinstance(annotations, list):
                for entry in annotations:
                    if isinstance(entry, dict) and 'image_path' in entry and 'latex' in entry:
                        img_path = entry['image_path']
                        if not os.path.isabs(img_path):
                            img_path = os.path.join(root_dir, img_path)
                        if self._validate_sample(img_path, entry['latex']):
                            samples.append({"image": img_path, "latex": entry['latex']})
            elif isinstance(annotations, dict):
                for key, value in annotations.items():
                    if key in ('meta', 'config', 'info', 'metadata'):
                        continue
                    latex = value if isinstance(value, str) else value.get('latex', '')
                    img_path = key if not os.path.isabs(key) else key
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(root_dir, img_path)
                    if self._validate_sample(img_path, latex):
                        samples.append({"image": img_path, "latex": latex})
        except Exception as e:
            logger.warning(f"Error parsing annotations.json: {e}")
            
        return samples

    # -----------------------------------------------------------------
    # Im2LaTeX-100K Parser
    # -----------------------------------------------------------------
    def parse_im2latex(self, extract_dir: str) -> List[Dict[str, Any]]:
        """
        Parse Im2LaTeX-100K from Kaggle extraction directory.
        
        Typical structure:
        - images/ or formula_images/ directory with PNG files
        - CSV file with formula/latex column and image name column
        
        Also supports the original im2latex-100k format:
        - im2latex_formulas.norm.lst (newline-separated LaTeX formulas)
        - im2latex_images/ directory with numbered PNG files
        """
        logger.info(f"Parsing Im2LaTeX from {extract_dir}")
        samples = []

        if not os.path.exists(extract_dir):
            logger.error(f"Im2LaTeX directory not found: {extract_dir}")
            return samples

        # Strategy 1: Try to find and parse CSV
        csv_files = glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)
        if csv_files:
            samples = self._parse_im2latex_csv(csv_files[0], extract_dir)
            if samples:
                logger.info(f"Im2LaTeX parsed (CSV): {len(samples)} valid samples found.")
                return samples

        # Strategy 2: Try im2latex-100k original format (formula list + images)
        formula_files = glob.glob(os.path.join(extract_dir, "**", "*.lst"), recursive=True) + \
                       glob.glob(os.path.join(extract_dir, "**", "*formula*"), recursive=True)
        
        if formula_files:
            for ff in formula_files:
                if os.path.isfile(ff):
                    samples = self._parse_im2latex_formula_list(ff, extract_dir)
                    if samples:
                        logger.info(f"Im2LaTeX parsed (formula list): {len(samples)} valid samples found.")
                        return samples

        # Strategy 3: Check for annotations.json
        samples = self._parse_annotations_json(extract_dir)
        
        logger.info(f"Im2LaTeX parsed: {len(samples)} valid samples found.")
        return samples

    def _parse_im2latex_csv(self, csv_path: str, base_dir: str) -> List[Dict[str, Any]]:
        """Parse Im2LaTeX from CSV format."""
        samples = []
        try:
            # Try different encodings and delimiters
            for encoding in ['utf-8', 'latin-1', 'utf-8-sig']:
                try:
                    with open(csv_path, 'r', encoding=encoding) as f:
                        # Try to detect delimiter
                        sample = f.read(2048)
                        f.seek(0)
                        
                        try:
                            dialect = csv.Sniffer().sniff(sample)
                            reader = csv.DictReader(f, dialect=dialect)
                        except csv.Error:
                            reader = csv.DictReader(f)
                        
                        rows = list(reader)
                        if not rows:
                            continue
                            
                        # Detect columns dynamically
                        columns = list(rows[0].keys())
                        
                        # Find image column
                        img_col = next((c for c in columns if 'image' in c.lower()),
                                      next((c for c in columns if 'file' in c.lower()),
                                          next((c for c in columns if 'path' in c.lower()), None)))
                        
                        # Find latex/formula column
                        latex_col = next((c for c in columns if 'formula' in c.lower() or 'latex' in c.lower()),
                                        next((c for c in columns if 'text' in c.lower()),
                                            next((c for c in columns if 'annotation' in c.lower()), None)))
                        
                        if not latex_col:
                            # If only 2 columns, use the second
                            if len(columns) >= 2:
                                latex_col = columns[-1]
                            elif len(columns) == 1:
                                latex_col = columns[0]
                                img_col = None
                            else:
                                continue
                        
                        # Find image directory
                        img_dir = self._find_image_directory(base_dir)
                        
                        for row in rows:
                            latex = str(row.get(latex_col, '')).strip()
                            if not latex:
                                continue
                                
                            if img_col:
                                img_name = str(row.get(img_col, '')).strip()
                                # Find the image
                                img_path = self._find_image_file(img_name, img_dir, base_dir)
                            else:
                                # Try to generate image name from index or row number
                                img_path = None
                                
                            if img_path and self._validate_sample(img_path, latex):
                                samples.append({"image": img_path, "latex": latex})
                            elif img_path is None:
                                # Image not found, skip
                                pass
                except UnicodeDecodeError:
                    continue
                    
            if samples:
                return samples
                
        except Exception as e:
            logger.warning(f"Error parsing Im2LaTeX CSV {csv_path}: {e}")
            
        return samples

    def _parse_im2latex_formula_list(self, formula_file: str, base_dir: str) -> List[Dict[str, Any]]:
        """Parse Im2LaTeX from a formula list file (one formula per line)."""
        samples = []
        try:
            img_dir = self._find_image_directory(base_dir)
            
            with open(formula_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    latex = line.strip()
                    if not latex:
                        continue
                    
                    # Try to find corresponding image (usually numbered)
                    img_name = f"{idx}.png"
                    img_path = self._find_image_file(img_name, img_dir, base_dir)
                    
                    if img_path and self._validate_sample(img_path, latex):
                        samples.append({"image": img_path, "latex": latex})
        except Exception as e:
            logger.warning(f"Error parsing formula list {formula_file}: {e}")
            
        return samples

    def _find_image_directory(self, base_dir: str) -> str:
        """Find the directory containing images."""
        candidate_names = ['images', 'formula_images', 'img', 'train_images', 'test_images', 
                         'val_images', 'im2latex_images']
        
        # Direct search
        for name in candidate_names:
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                return path
        
        # Recursive search
        for root, dirs, files in os.walk(base_dir):
            for d in dirs:
                if d.lower() in candidate_names:
                    return os.path.join(root, d)
        
        # If no specific directory found, use the base_dir or first directory with images
        for root, dirs, files in os.walk(base_dir):
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                return root
        
        return base_dir

    def _find_image_file(self, img_name: str, img_dir: str, base_dir: str) -> Optional[str]:
        """Find an image file by name."""
        if not img_name:
            return None
            
        # Direct path
        if os.path.isabs(img_name) and os.path.exists(img_name):
            return img_name
            
        paths_to_try = [
            os.path.join(img_dir, img_name),
            os.path.join(img_dir, os.path.basename(img_name)),
            os.path.join(base_dir, img_name),
            os.path.join(base_dir, os.path.basename(img_name)),
        ]
        
        # Add .png extension if not present
        if not any(img_name.lower().endswith(ext) for ext in self.IMG_EXTENSIONS):
            paths_to_try.extend([
                os.path.join(img_dir, img_name + '.png'),
                os.path.join(base_dir, img_name + '.png'),
            ])
        
        for path in paths_to_try:
            if os.path.exists(path):
                return path
        
        return None

    # -----------------------------------------------------------------
    # MathWriting Parser (Hugging Face)
    # -----------------------------------------------------------------
    def parse_mathwriting(self, hf_dataset) -> List[Dict[str, Any]]:
        """
        Parse MathWriting from Hugging Face dataset object.
        
        HF datasets typically provide:
        - 'image' column with PIL.Image objects
        - 'text' or 'latex' column with formula strings
        
        Returns list with PIL.Image objects (not paths).
        """
        logger.info("Parsing MathWriting from Hugging Face dataset object.")
        samples = []

        if hf_dataset is None:
            logger.warning("MathWriting HF dataset is None.")
            return samples

        try:
            # Detect column names dynamically
            if hasattr(hf_dataset, 'column_names'):
                columns = hf_dataset.column_names
            elif hasattr(hf_dataset, 'features'):
                columns = list(hf_dataset.features.keys())
            else:
                columns = []
            
            # Find image column
            img_col = next((c for c in columns if 'image' in c.lower()),
                          next((c for c in columns if 'pixel' in c.lower()), None))
            
            # Find text/latex column
            txt_col = next((c for c in columns if 'latex' in c.lower()),
                          next((c for c in columns if 'text' in c.lower()),
                              next((c for c in columns if 'formula' in c.lower()), None)))
            
            if not txt_col:
                logger.error(f"Could not find text/latex column in MathWriting dataset. Columns: {columns}")
                return samples

            for item in hf_dataset:
                try:
                    latex = str(item.get(txt_col, '')).strip()
                    if not latex:
                        continue
                    
                    if img_col and img_col in item:
                        img = item[img_col]
                        # Handle HF Image object vs PIL Image
                        if hasattr(img, 'convert'):  # PIL Image
                            samples.append({"image": img.convert('L'), "latex": latex})
                        elif isinstance(img, dict) and 'bytes' in img:
                            # HF encoded image
                            import io
                            pil_img = Image.open(io.BytesIO(img['bytes'])).convert('L')
                            samples.append({"image": pil_img, "latex": latex})
                        elif isinstance(img, dict) and 'path' in img:
                            samples.append({"image": img['path'], "latex": latex})
                        else:
                            samples.append({"image": img, "latex": latex})
                    else:
                        # No image column, store latex only (skip if image required)
                        logger.warning(f"No image found for item with latex: {latex[:50]}...")
                except Exception as e:
                    logger.debug(f"Skipping item due to error: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing MathWriting HF dataset: {e}")

        logger.info(f"MathWriting parsed: {len(samples)} valid samples found.")
        return samples

    # -----------------------------------------------------------------
    # HME100K Parser
    # -----------------------------------------------------------------
    def parse_hme100k(self, extract_dir: str) -> List[Dict[str, Any]]:
        """
        Parse HME100K dataset from GitHub clone or download.
        
        Structure varies but typically:
        - images/ directory with formula images
        - train.txt or labels.txt with mappings
        """
        logger.info(f"Parsing HME100K from {extract_dir}")
        samples = []

        if not os.path.exists(extract_dir):
            logger.error(f"HME100K directory not found: {extract_dir}")
            return samples

        # Strategy 1: Find and parse label files
        samples = self._parse_hme100k_labels(extract_dir)
        if samples:
            logger.info(f"HME100K parsed (labels): {len(samples)} valid samples found.")
            return samples

        # Strategy 2: Check for images + formula directories
        img_dirs = self._find_subdirectories(extract_dir, ['images', 'formula_images', 'train', 'data'])
        formula_dirs = self._find_subdirectories(extract_dir, ['labels', 'annotations', 'formula'])
        
        if img_dirs:
            img_dir = img_dirs[0]
            images = self._find_images(img_dir)
            
            for img_path in images:
                # Try to find matching label
                base = os.path.splitext(os.path.basename(img_path))[0]
                latex = self._find_hme100k_label(base, extract_dir, formula_dirs)
                if latex and self._validate_sample(img_path, latex):
                    samples.append({"image": img_path, "latex": latex})

        # Strategy 3: Check for annotations.json
        if not samples:
            samples = self._parse_annotations_json(extract_dir)
            logger.info(f"HME100K parsed (json): {len(samples)} valid samples found.")

        logger.info(f"HME100K parsed: {len(samples)} valid samples found.")
        return samples

    def _parse_hme100k_labels(self, root_dir: str) -> List[Dict[str, Any]]:
        """Parse HME100K label file formats."""
        samples = []
        label_file = self._find_label_file(root_dir)
        
        if not label_file:
            return samples
        
        label_dir = os.path.dirname(label_file)
        
        try:
            for encoding in ['utf-8', 'latin-1', 'utf-8-sig', 'gbk']:
                try:
                    with open(label_file, 'r', encoding=encoding) as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                            
                            # Common formats:
                            # img_name\tlatex
                            # img_name latex
                            # img_name|latex
                            # img_path\tlatex
                            parts = None
                            for sep in ['\t', '|', ',', ':']:
                                parts = line.split(sep)
                                if len(parts) >= 2:
                                    break
                            
                            if parts and len(parts) >= 2:
                                img_name = parts[0].strip()
                                latex = sep.join(parts[1:]).strip()
                            elif parts and len(parts) == 1:
                                # Line is just the latex
                                img_name = None
                                latex = parts[0].strip()
                            else:
                                # Space-separated
                                parts = line.split(None, 1)
                                if len(parts) >= 2:
                                    img_name = parts[0].strip()
                                    latex = parts[1].strip()
                                else:
                                    continue
                            
                            # Resolve image path
                            if img_name:
                                img_path = self._resolve_hme100k_image(img_name, label_dir, root_dir)
                            else:
                                img_path = None
                            
                            if img_path and self._validate_sample(img_path, latex):
                                samples.append({"image": img_path, "latex": latex})
                                
                    if samples:
                        break
                except UnicodeDecodeError:
                    continue
                    
        except Exception as e:
            logger.warning(f"Error parsing HME100K labels: {e}")
        
        return samples

    def _resolve_hme100k_image(self, img_name: str, label_dir: str, root_dir: str) -> Optional[str]:
        """Resolve image path for HME100K."""
        # Direct path
        if os.path.isabs(img_name) and os.path.exists(img_name):
            return img_name
        
        # Relative to label directory
        path = os.path.join(label_dir, img_name)
        if os.path.exists(path):
            return path
        
        # Search in common subdirectories
        search_dirs = [root_dir, label_dir]
        for name in ['images', 'train', 'test', 'val', 'data', 'formula_images']:
            d = os.path.join(root_dir, name)
            if os.path.isdir(d):
                search_dirs.append(d)
        
        for search_dir in search_dirs:
            for sub_root, sub_dirs, sub_files in os.walk(search_dir):
                if img_name.lower() in [f.lower() for f in sub_files]:
                    return os.path.join(sub_root, img_name)
                
                # Also try without extension
                base_no_ext = os.path.splitext(img_name)[0]
                for f in sub_files:
                    if os.path.splitext(f)[0].lower() == base_no_ext.lower():
                        if any(f.lower().endswith(ext) for ext in self.IMG_EXTENSIONS):
                            return os.path.join(sub_root, f)
        
        return None

    def _find_hme100k_label(self, img_base: str, root_dir: str, formula_dirs: List[str]) -> Optional[str]:
        """Try to find a label for a given image base name."""
        # Search in label files
        for formula_dir in formula_dirs:
            for root, dirs, files in os.walk(formula_dir):
                for f in files:
                    if f.lower().endswith(('.txt', '.lst', '.csv')):
                        txt_path = os.path.join(root, f)
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as tf:
                                for line in tf:
                                    parts = line.strip().split('\t')
                                    if len(parts) >= 2 and parts[0].strip() == img_base:
                                        return '\t'.join(parts[1:]).strip()
                                    elif parts[0].strip() == img_base:
                                        return ' '.join(parts[1:]).strip()
                        except Exception:
                            continue
        return None

    def _find_subdirectories(self, root: str, names: List[str]) -> List[str]:
        """Find subdirectories with specific names."""
        found = []
        for name in names:
            for r, dirs, files in os.walk(root):
                for d in dirs:
                    if d.lower() == name.lower():
                        found.append(os.path.join(r, d))
        return found

    # -----------------------------------------------------------------
    # Unified Parser Interface
    # -----------------------------------------------------------------
    def parse_dataset(self, dataset_type: str, source: Any) -> List[Dict[str, Any]]:
        """
        Unified parsing interface for all dataset types.
        
        Args:
            dataset_type: One of 'crohme', 'im2latex', 'mathwriting', 'hme100k'
            source: Directory path (str) or HF dataset object
            
        Returns:
            List of dicts with 'image' and 'latex' keys
        """
        parsers = {
            'crohme': lambda s: self.parse_crohme(s),
            'im2latex': lambda s: self.parse_im2latex(s),
            'mathwriting': lambda s: self.parse_mathwriting(s),
            'hme100k': lambda s: self.parse_hme100k(s),
        }
        
        parser_func = parsers.get(dataset_type.lower())
        if parser_func is None:
            logger.error(f"Unknown dataset type: {dataset_type}")
            return []
            
        return parser_func(source)


def create_parser() -> DatasetParser:
    """Factory function to create a DatasetParser instance."""
    return DatasetParser()