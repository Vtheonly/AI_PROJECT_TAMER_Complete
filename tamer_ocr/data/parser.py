"""
Dataset Parser for TAMER OCR Training.
Converts various dataset formats to a unified structure.
"""

import os
import glob
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image, ImageDraw
import csv
import xml.etree.ElementTree as ET

logger = logging.getLogger("TAMER.Parser")


class DatasetParser:
    IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')

    def _render_inkml(self, inkml_path: str, output_img_path: str, line_width: int = 3) -> Optional[str]:
        try:
            tree = ET.parse(inkml_path)
            root = tree.getroot()

            def strip_ns(tag):
                return tag.split('}', 1)[-1] if '}' in tag else tag

            latex_truth = ""
            traces = []

            for elem in root.iter():
                tag = strip_ns(elem.tag)
                if tag == 'annotation' and elem.attrib.get('type') == 'truth':
                    if elem.text:
                        latex_truth = elem.text.strip()
                elif tag == 'trace':
                    if not elem.text:
                        continue
                    coords = elem.text.strip().split(',')
                    stroke = []
                    for coord in coords:
                        vals = coord.strip().split()
                        if len(vals) >= 2:
                            try:
                                stroke.append((float(vals[0]), float(vals[1])))
                            except ValueError:
                                continue
                    if stroke:
                        traces.append(stroke)

            if not traces or not latex_truth:
                return None

            all_x = [pt[0] for stroke in traces for pt in stroke]
            all_y = [pt[1] for stroke in traces for pt in stroke]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            width = max(max_x - min_x, 1)
            height = max(max_y - min_y, 1)

            padding = 20
            scale = 200.0 / height
            img_w = int(width * scale) + padding * 2
            img_h = int(height * scale) + padding * 2

            img = Image.new('L', (img_w, img_h), color=255)
            draw = ImageDraw.Draw(img)

            for stroke in traces:
                scaled_stroke = [(int((pt[0] - min_x) * scale) + padding,
                                  int((pt[1] - min_y) * scale) + padding) for pt in stroke]
                if len(scaled_stroke) > 1:
                    draw.line(scaled_stroke, fill=0, width=line_width, joint='curve')
                elif len(scaled_stroke) == 1:
                    pt = scaled_stroke[0]
                    draw.ellipse([pt[0]-line_width, pt[1]-line_width,
                                  pt[0]+line_width, pt[1]+line_width], fill=0)

            img.save(output_img_path)
            return latex_truth
        except Exception as e:
            logger.debug(f"Failed to render InkML {inkml_path}: {e}")
            return None

    def _find_images(self, root_dir: str) -> List[str]:
        images = []
        for ext in self.IMG_EXTENSIONS:
            images.extend(glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True))
            images.extend(glob.glob(os.path.join(root_dir, f"**/*{ext.upper()}"), recursive=True))
        return sorted(set(images))

    def _find_label_file(self, root_dir: str, preferred_names: List[str] = None) -> Optional[str]:
        preferred = preferred_names or ['train.txt', 'labels.txt', 'annotation.txt', 'formula.txt']
        for name in preferred:
            for root, dirs, files in os.walk(root_dir):
                if name.lower() in [f.lower() for f in files]:
                    return os.path.join(root, name)
        return None

    def _validate_sample(self, img_source: Any, latex: str) -> bool:
        if not latex or not latex.strip():
            return False
        if isinstance(img_source, str):
            return os.path.exists(img_source)
        if isinstance(img_source, Image.Image):
            return True
        return False

    def parse_im2latex(self, extract_dir: str) -> List[Dict[str, Any]]:
        logger.info(f"Parsing Im2LaTeX from {extract_dir}")
        samples = []
        if not os.path.exists(extract_dir):
            return samples

        csv_files = glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)
        if csv_files:
            samples = self._parse_im2latex_csv(csv_files[0], extract_dir)
            if samples:
                for s in samples:
                    s['dataset_name'] = 'im2latex'
                return samples

        formula_files = glob.glob(os.path.join(extract_dir, "**", "*.lst"), recursive=True) + \
                       glob.glob(os.path.join(extract_dir, "**", "*formula*"), recursive=True)
        if formula_files:
            for ff in formula_files:
                if os.path.isfile(ff):
                    samples = self._parse_im2latex_formula_list(ff, extract_dir)
                    if samples:
                        for s in samples:
                            s['dataset_name'] = 'im2latex'
                        return samples

        samples = self._parse_annotations_json(extract_dir)
        for s in samples:
            s['dataset_name'] = 'im2latex'
        return samples

    def _parse_im2latex_csv(self, csv_path: str, base_dir: str) -> List[Dict[str, Any]]:
        samples = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                img_col = None
                latex_col = None
                if reader.fieldnames:
                    img_col = next((c for c in reader.fieldnames if 'image' in c.lower() or 'file' in c.lower()), None)
                    latex_col = next((c for c in reader.fieldnames if 'formula' in c.lower() or 'latex' in c.lower()), None)
                    if not latex_col and len(reader.fieldnames) >= 2:
                        latex_col = reader.fieldnames[-1]

                img_dir = self._find_image_directory(base_dir)
                for row in reader:
                    latex = str(row.get(latex_col, '')).strip() if latex_col else ''
                    if not latex:
                        continue
                    img_name = str(row.get(img_col, '')).strip() if img_col else ''
                    img_path = self._find_image_file(img_name, img_dir, base_dir) if img_name else None
                    if img_path and self._validate_sample(img_path, latex):
                        samples.append({"image": img_path, "latex": latex})
        except Exception as e:
            logger.warning(f"Error parsing Im2LaTeX CSV: {e}")
        return samples

    def _parse_im2latex_formula_list(self, formula_file: str, base_dir: str) -> List[Dict[str, Any]]:
        samples = []
        try:
            img_dir = self._find_image_directory(base_dir)
            with open(formula_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    latex = line.strip()
                    if not latex:
                        continue
                    img_name = f"{idx}.png"
                    img_path = self._find_image_file(img_name, img_dir, base_dir)
                    if img_path and self._validate_sample(img_path, latex):
                        samples.append({"image": img_path, "latex": latex})
        except Exception as e:
            logger.warning(f"Error parsing formula list: {e}")
        return samples

    def _find_image_directory(self, base_dir: str) -> str:
        best_dir = base_dir
        max_imgs = -1
        for root, dirs, files in os.walk(base_dir):
            count = sum(1 for f in files if f.lower().endswith(self.IMG_EXTENSIONS))
            if count > max_imgs:
                max_imgs = count
                best_dir = root
        return best_dir

    def _find_image_file(self, img_name: str, img_dir: str, base_dir: str) -> Optional[str]:
        if not img_name:
            return None
        for path in [os.path.join(img_dir, img_name), os.path.join(base_dir, img_name)]:
            if os.path.exists(path):
                return path
        if not any(img_name.lower().endswith(ext) for ext in self.IMG_EXTENSIONS):
            for path in [os.path.join(img_dir, img_name + '.png'), os.path.join(base_dir, img_name + '.png')]:
                if os.path.exists(path):
                    return path
        return None

    def parse_crohme(self, extract_dir: str) -> List[Dict[str, Any]]:
        logger.info(f"Parsing CROHME from {extract_dir}")
        samples = []

        inkml_files = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith('.inkml'):
                    inkml_files.append(os.path.join(root, f))

        if inkml_files:
            logger.info(f"Found {len(inkml_files)} .inkml files. Rendering to PNGs...")
            img_out_dir = os.path.join(extract_dir, "images")
            os.makedirs(img_out_dir, exist_ok=True)

            for idx, inkml_path in enumerate(inkml_files):
                base_name = os.path.splitext(os.path.basename(inkml_path))[0]
                out_img = os.path.join(img_out_dir, f"{base_name}.png")
                latex = self._render_inkml(inkml_path, out_img)
                if latex and os.path.exists(out_img):
                    samples.append({"image": out_img, "latex": latex, "dataset_name": "crohme"})
                if (idx + 1) % 1000 == 0:
                    logger.info(f"  Rendered {idx + 1}/{len(inkml_files)} CROHME files...")

            logger.info(f"Successfully rendered {len(samples)} CROHME images from InkML.")
            return samples

        # Fallback: existing images + label files
        image_map = {}
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(self.IMG_EXTENSIONS):
                    image_map[os.path.splitext(f)[0].lower()] = os.path.join(root, f)

        for root, _, files in os.walk(extract_dir):
            for f in files:
                f_lower = f.lower()
                if f_lower.endswith('.txt') and 'readme' not in f_lower:
                    try:
                        with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as tf:
                            for line in tf:
                                parts = line.strip().split('\t') if '\t' in line else line.strip().split(' ', 1)
                                if len(parts) >= 2:
                                    img_id = os.path.splitext(parts[0].strip())[0].lower()
                                    latex = parts[-1].strip()
                                    if img_id in image_map:
                                        samples.append({"image": image_map[img_id], "latex": latex, "dataset_name": "crohme"})
                    except Exception:
                        continue

        unique_samples = list({s['image']: s for s in samples}.values())
        logger.info(f"Matched {len(unique_samples)} CROHME samples.")
        return unique_samples

    def parse_hme100k(self, extract_dir: str) -> List[Dict[str, Any]]:
        logger.info(f"Parsing HME100K from {extract_dir}")
        samples = []
        image_map = {}
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(self.IMG_EXTENSIONS):
                    image_map[os.path.splitext(f)[0].lower()] = os.path.join(root, f)

        for root, _, files in os.walk(extract_dir):
            for f in files:
                f_lower = f.lower()
                if f_lower.endswith('.txt') or 'label' in f_lower:
                    if 'readme' in f_lower:
                        continue
                    try:
                        with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as label_f:
                            for line in label_f:
                                line = line.strip()
                                if not line:
                                    continue
                                parts = line.split('\t') if '\t' in line else line.split(' ', 1)
                                if len(parts) >= 2:
                                    img_id = os.path.splitext(parts[0].strip())[0].lower()
                                    latex = parts[-1].strip()
                                    if img_id in image_map:
                                        samples.append({"image": image_map[img_id], "latex": latex, "dataset_name": "hme100k"})
                    except Exception:
                        continue

        unique_samples = list({s['image']: s for s in samples}.values())
        logger.info(f"Matched {len(unique_samples)} HME100K samples.")
        return unique_samples

    def parse_mathwriting(self, hf_dataset, extract_dir: str = None, max_samples: int = None) -> List[Dict[str, Any]]:
        logger.info(f"Parsing Hugging Face dataset object. Max samples: {max_samples or 'All'}")
        samples = []
        if hf_dataset is None:
            return samples

        img_dir = None
        if extract_dir:
            img_dir = os.path.join(extract_dir, "images")
            os.makedirs(img_dir, exist_ok=True)

        try:
            columns = getattr(hf_dataset, 'column_names', list(getattr(hf_dataset, 'features', {}).keys()))
            img_col = next((c for c in columns if 'image' in c.lower()), None)
            txt_col = next((c for c in columns if 'latex' in c.lower() or 'text' in c.lower() or 'formula' in c.lower()), None)

            if not txt_col:
                logger.error(f"Could not find text/latex column. Columns: {columns}")
                return samples

            for idx in range(len(hf_dataset)):
                if max_samples and idx >= max_samples:
                    break
                try:
                    item = hf_dataset[idx]
                    latex = str(item.get(txt_col, '')).strip()
                    if not latex:
                        continue

                    if img_col and img_col in item:
                        img = item[img_col]
                        pil_img = None
                        if hasattr(img, 'convert'):
                            pil_img = img.convert('L')
                        elif isinstance(img, dict) and 'bytes' in img:
                            import io
                            pil_img = Image.open(io.BytesIO(img['bytes'])).convert('L')
                        elif isinstance(img, dict) and 'path' in img:
                            samples.append({"image": img['path'], "latex": latex, "dataset_name": "mathwriting"})
                            continue
                        else:
                            samples.append({"image": img, "latex": latex, "dataset_name": "mathwriting"})
                            continue

                        if pil_img and img_dir:
                            import hashlib
                            h = hashlib.md5(latex.encode('utf-8')).hexdigest()[:8]
                            img_path = os.path.join(img_dir, f"img_{idx}_{h}.png")
                            if not os.path.exists(img_path):
                                # CRITICAL FIX: Convert to Numpy Array and back to break HF _idat references 
                                # This fully solves the "AttributeError: '_idat' object has no attribute 'fileno'" error
                                safe_img = Image.fromarray(np.array(pil_img))
                                safe_img.save(img_path, format="PNG")
                            samples.append({"image": img_path, "latex": latex, "dataset_name": "mathwriting"})
                        elif pil_img:
                            samples.append({"image": pil_img, "latex": latex, "dataset_name": "mathwriting"})
                except Exception as e:
                    logger.debug(f"Skipping corrupt HF image sample at index {idx}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing HF dataset: {e}")

        logger.info(f"HF dataset parsing complete: {len(samples)} valid samples found.")
        return samples

    def _parse_annotations_json(self, root_dir: str) -> List[Dict[str, Any]]:
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
                    img_path = key if not os.path.isabs(key) else os.path.join(root_dir, key)
                    if self._validate_sample(img_path, latex):
                        samples.append({"image": img_path, "latex": latex})
        except Exception as e:
            logger.warning(f"Error parsing annotations.json: {e}")
        return samples

    def parse_dataset(self, dataset_type: str, source: Any) -> List[Dict[str, Any]]:
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
    return DatasetParser()