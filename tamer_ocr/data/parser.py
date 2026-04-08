"""
Dataset Parser for TAMER OCR Training.
"""

import os
import glob
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import csv
import xml.etree.ElementTree as ET  # <--- NEW IMPORT
from PIL import Image, ImageDraw    # <--- NEW IMPORT


logger = logging.getLogger("TAMER.Parser")


class DatasetParser:
    IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')



    def _render_inkml(self, inkml_path: str, output_img_path: str, line_width: int = 3) -> Optional[str]:
        """Parses an .inkml file, renders pen strokes to a PNG, and returns the LaTeX truth."""
        try:
            tree = ET.parse(inkml_path)
            root = tree.getroot()
            
            # Helper to ignore XML namespaces
            def strip_ns(tag):
                return tag.split('}', 1)[-1] if '}' in tag else tag

            latex_truth = ""
            traces = []

            for elem in root.iter():
                tag = strip_ns(elem.tag)
                
                # 1. Extract LaTeX ground truth
                if tag == 'annotation' and elem.attrib.get('type') == 'truth':
                    if elem.text:
                        latex_truth = elem.text.strip()
                
                # 2. Extract pen strokes (traces)
                elif tag == 'trace':
                    if not elem.text: continue
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

            # 3. Compute Bounding Box
            all_x = [pt[0] for stroke in traces for pt in stroke]
            all_y = [pt[1] for stroke in traces for pt in stroke]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            
            width = max(max_x - min_x, 1)
            height = max(max_y - min_y, 1)

            # 4. Normalize scale and render (keeps math symbols looking natural)
            padding = 20
            scale = 200.0 / height  # Scale handwriting to a fixed 200px height
            
            img_w = int(width * scale) + padding * 2
            img_h = int(height * scale) + padding * 2

            # White background
            img = Image.new('L', (img_w, img_h), color=255)
            draw = ImageDraw.Draw(img)

            # Draw smooth black strokes
            for stroke in traces:
                scaled_stroke = [(int((pt[0] - min_x) * scale) + padding, 
                                  int((pt[1] - min_y) * scale) + padding) for pt in stroke]
                if len(scaled_stroke) > 1:
                    draw.line(scaled_stroke, fill=0, width=line_width, joint='curve')
                elif len(scaled_stroke) == 1: # Draw dot
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
        
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith('.txt'):
                    return os.path.join(root, f)
        return None

    def _safe_read_latex(self, path: str) -> str:
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
        if not latex or not latex.strip():
            return False
        if isinstance(img_source, str):
            return os.path.exists(img_source)
        if isinstance(img_source, Image.Image):
            return True
        return False


    def _find_matching_image(self, search_dir: str, img_name: str) -> Optional[str]:
        if not img_name:
            return None
            
        # CHANGED: Ensure the image extension is a supported one (ignore .inkml directly)
        base_name, ext = os.path.splitext(img_name)
        if ext.lower() not in self.IMG_EXTENSIONS:
            img_name = base_name + ".png"

        direct = os.path.join(search_dir, img_name)
        if os.path.exists(direct) and any(direct.lower().endswith(ext) for ext in self.IMG_EXTENSIONS):
            return direct

        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if f.lower() == img_name.lower():
                    return os.path.join(root, f)
                # Fallback: check if the base name matches any of our supported image extensions
                if os.path.splitext(f)[0].lower() == os.path.splitext(img_name)[0].lower():
                    if any(f.lower().endswith(ext) for ext in self.IMG_EXTENSIONS):
                        return os.path.join(root, f)
        return None

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
                    img_path = key if not os.path.isabs(key) else key
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(root_dir, img_path)
                    if self._validate_sample(img_path, latex):
                        samples.append({"image": img_path, "latex": latex})
        except Exception as e:
            logger.warning(f"Error parsing annotations.json: {e}")
            
        return samples

    def parse_im2latex(self, extract_dir: str) -> List[Dict[str, Any]]:
        logger.info(f"Parsing Im2LaTeX from {extract_dir}")
        samples = []

        if not os.path.exists(extract_dir):
            logger.error(f"Im2LaTeX directory not found: {extract_dir}")
            return samples

        csv_files = glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)
        if csv_files:
            samples = self._parse_im2latex_csv(csv_files[0], extract_dir)
            if samples:
                logger.info(f"Im2LaTeX parsed (CSV): {len(samples)} valid samples found.")
                return samples

        formula_files = glob.glob(os.path.join(extract_dir, "**", "*.lst"), recursive=True) + \
                       glob.glob(os.path.join(extract_dir, "**", "*formula*"), recursive=True)
        
        if formula_files:
            for ff in formula_files:
                if os.path.isfile(ff):
                    samples = self._parse_im2latex_formula_list(ff, extract_dir)
                    if samples:
                        logger.info(f"Im2LaTeX parsed (formula list): {len(samples)} valid samples found.")
                        return samples

        samples = self._parse_annotations_json(extract_dir)
        
        logger.info(f"Im2LaTeX parsed: {len(samples)} valid samples found.")
        return samples

    def _parse_im2latex_csv(self, csv_path: str, base_dir: str) -> List[Dict[str, Any]]:
        samples = []
        try:
            for encoding in ['utf-8', 'latin-1', 'utf-8-sig']:
                try:
                    with open(csv_path, 'r', encoding=encoding) as f:
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
                            
                        columns = list(rows[0].keys())
                        
                        img_col = next((c for c in columns if 'image' in c.lower()),
                                      next((c for c in columns if 'file' in c.lower()),
                                          next((c for c in columns if 'path' in c.lower()), None)))
                        
                        latex_col = next((c for c in columns if 'formula' in c.lower() or 'latex' in c.lower()),
                                        next((c for c in columns if 'text' in c.lower()),
                                            next((c for c in columns if 'annotation' in c.lower()), None)))
                        
                        if not latex_col:
                            if len(columns) >= 2:
                                latex_col = columns[-1]
                            elif len(columns) == 1:
                                latex_col = columns[0]
                                img_col = None
                            else:
                                continue
                        
                        img_dir = self._find_image_directory(base_dir)
                        
                        for row in rows:
                            latex = str(row.get(latex_col, '')).strip()
                            if not latex:
                                continue
                                
                            if img_col:
                                img_name = str(row.get(img_col, '')).strip()
                                img_path = self._find_image_file(img_name, img_dir, base_dir)
                            else:
                                img_path = None
                                
                            if img_path and self._validate_sample(img_path, latex):
                                samples.append({"image": img_path, "latex": latex})
                            elif img_path is None:
                                pass
                except UnicodeDecodeError:
                    continue
                    
            if samples:
                return samples
                
        except Exception as e:
            logger.warning(f"Error parsing Im2LaTeX CSV {csv_path}: {e}")
            
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
            logger.warning(f"Error parsing formula list {formula_file}: {e}")
            
        return samples

    def _find_image_directory(self, base_dir: str) -> str:
        """Deep search for the folder containing the most images."""
        best_dir = base_dir
        max_imgs = -1
        for root, dirs, files in os.walk(base_dir):
            count = sum(1 for f in files if f.lower().endswith(self.IMG_EXTENSIONS))
            if count > max_imgs:
                max_imgs = count
                best_dir = root
        return best_dir






    def parse_crohme(self, extract_dir: str) -> List[Dict[str, Any]]:
        logger.info(f"Power-Parsing CROHME from {extract_dir}")
        samples = []
        
        # Look for existing images first
        image_map = {}
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(self.IMG_EXTENSIONS):
                    image_map[os.path.splitext(f)[0].lower()] = os.path.join(root, f)

        # Look for .inkml files to render
        inkml_files = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith('.inkml'):
                    inkml_files.append(os.path.join(root, f))

        # If we have .inkml files, render them directly!
        if inkml_files:
            logger.info(f"Found {len(inkml_files)} .inkml files. Rendering to high-quality PNGs...")
            img_out_dir = os.path.join(extract_dir, "images")
            os.makedirs(img_out_dir, exist_ok=True)
            
            rendered_count = 0
            for idx, inkml_path in enumerate(inkml_files):
                base_name = os.path.splitext(os.path.basename(inkml_path))[0]
                out_img = os.path.join(img_out_dir, f"{base_name}.png")
                
                # Render the image and extract the embedded LaTeX truth
                latex = self._render_inkml(inkml_path, out_img)
                
                if latex and os.path.exists(out_img):
                    samples.append({"image": out_img, "latex": latex})
                    rendered_count += 1
                    
                if (idx + 1) % 1000 == 0:
                    logger.info(f"  Rendered {idx + 1}/{len(inkml_files)} CROHME files...")
                    
            logger.info(f"Successfully rendered {rendered_count} CROHME images from InkML.")
            return samples

        # --- FALLBACK: If user provided a dataset that already has text mappings ---
        for root, _, files in os.walk(extract_dir):
            for f in files:
                f_lower = f.lower()
                if f_lower.endswith('.txt') or 'label' in f_lower or 'truth' in f_lower or 'gt' in f_lower:
                    if 'readme' in f_lower: continue
                    try:
                        with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as tf:
                            for line in tf:
                                line = line.strip()
                                if not line: continue
                                parts = line.split('\t') if '\t' in line else line.split(' ', 1)
                                if len(parts) < 2: continue
                                
                                img_id = os.path.splitext(parts[0].strip())[0].lower()
                                latex = parts[-1].strip()
                                if img_id in image_map:
                                    samples.append({"image": image_map[img_id], "latex": latex})
                    except Exception: continue
                    
        unique_samples = list({s['image']: s for s in samples}.values())
        logger.info(f"Matched {len(unique_samples)} CROHME samples.")
        return unique_samples




    def parse_hme100k(self, extract_dir: str) -> List[Dict[str, Any]]:
        logger.info(f"Power-Parsing HME100K from {extract_dir}")
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
                    if 'readme' in f_lower: continue
                    try:
                        with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as label_f:
                            for line in label_f:
                                line = line.strip()
                                if not line: continue
                                parts = line.split('\t') if '\t' in line else line.split(' ', 1)
                                if len(parts) >= 2:
                                    img_id = os.path.splitext(parts[0].strip())[0].lower()
                                    latex = parts[-1].strip()
                                    if img_id in image_map:
                                        samples.append({"image": image_map[img_id], "latex": latex})
                    except Exception:
                        continue
                        
        # Remove duplicates
        unique_samples = list({s['image']: s for s in samples}.values())
        logger.info(f"Matched {len(unique_samples)} HME100K samples.")
        return unique_samples

    # Make sure to REMOVE any other older versions of `parse_hme100k` located further down the file.







  
    def _find_image_file(self, img_name: str, img_dir: str, base_dir: str) -> Optional[str]:
        if not img_name:
            return None
            
        if os.path.isabs(img_name) and os.path.exists(img_name):
            return img_name
            
        paths_to_try = [
            os.path.join(img_dir, img_name),
            os.path.join(img_dir, os.path.basename(img_name)),
            os.path.join(base_dir, img_name),
            os.path.join(base_dir, os.path.basename(img_name)),
        ]
        
        if not any(img_name.lower().endswith(ext) for ext in self.IMG_EXTENSIONS):
            paths_to_try.extend([
                os.path.join(img_dir, img_name + '.png'),
                os.path.join(base_dir, img_name + '.png'),
            ])
        
        for path in paths_to_try:
            if os.path.exists(path):
                return path
        
        return None

    def parse_mathwriting(self, hf_dataset, extract_dir: str = None, max_samples: int = None) -> List[Dict[str, Any]]:
        logger.info(f"Parsing Hugging Face dataset object. Max samples: {max_samples or 'All'}")
        samples = []

        if hf_dataset is None:
            logger.warning("HF dataset is None.")
            return samples

        img_dir = None
        if extract_dir:
            img_dir = os.path.join(extract_dir, "images")
            os.makedirs(img_dir, exist_ok=True)

        try:
            columns = getattr(hf_dataset, 'column_names', list(getattr(hf_dataset, 'features', {}).keys()))
            
            img_col = next((c for c in columns if 'image' in c.lower()),
                          next((c for c in columns if 'pixel' in c.lower()), None))
            
            txt_col = next((c for c in columns if 'latex' in c.lower()),
                          next((c for c in columns if 'text' in c.lower()),
                              next((c for c in columns if 'formula' in c.lower()), None)))
            
            if not txt_col:
                logger.error(f"Could not find text/latex column. Columns: {columns}")
                return samples

            for idx, item in enumerate(hf_dataset):
                if max_samples and idx >= max_samples:
                    logger.info(f"Reached max_samples limit ({max_samples}). Stopping parse.")
                    break
                    
                try:
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
                            samples.append({"image": img['path'], "latex": latex})
                            continue
                        else:
                            samples.append({"image": img, "latex": latex})
                            continue
                            
                        if pil_img and img_dir:
                            import hashlib
                            h = hashlib.md5(latex.encode('utf-8')).hexdigest()[:8]
                            img_path = os.path.join(img_dir, f"img_{idx}_{h}.png")
                            if not os.path.exists(img_path):
                                pil_img.save(img_path)
                            samples.append({"image": img_path, "latex": latex})
                            
                            if len(samples) % 10000 == 0:
                                logger.info(f"  Parsed & saved {len(samples)} images...")
                                
                        elif pil_img:
                            samples.append({"image": pil_img, "latex": latex})
                            
                except Exception as e:
                    logger.debug(f"Skipping item {idx} due to error: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing HF dataset: {e}")

        logger.info(f"HF dataset parsing complete: {len(samples)} valid samples found.")
        return samples

    def parse_hme100k(self, extract_dir: str) -> List[Dict[str, Any]]:
        logger.info(f"Parsing HME100K from {extract_dir}")
        samples = []

        if not os.path.exists(extract_dir):
            logger.error(f"HME100K directory not found: {extract_dir}")
            return samples

        samples = self._parse_hme100k_labels(extract_dir)
        if samples:
            logger.info(f"HME100K parsed (labels): {len(samples)} valid samples found.")
            return samples

        img_dirs = self._find_subdirectories(extract_dir, ['images', 'formula_images', 'train', 'data'])
        formula_dirs = self._find_subdirectories(extract_dir, ['labels', 'annotations', 'formula'])
        
        if img_dirs:
            img_dir = img_dirs[0]
            images = self._find_images(img_dir)
            
            for img_path in images:
                base = os.path.splitext(os.path.basename(img_path))[0]
                latex = self._find_hme100k_label(base, extract_dir, formula_dirs)
                if latex and self._validate_sample(img_path, latex):
                    samples.append({"image": img_path, "latex": latex})

        if not samples:
            samples = self._parse_annotations_json(extract_dir)
            logger.info(f"HME100K parsed (json): {len(samples)} valid samples found.")

        logger.info(f"HME100K parsed: {len(samples)} valid samples found.")
        return samples

    def _parse_hme100k_labels(self, root_dir: str) -> List[Dict[str, Any]]:
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
                            
                            parts = None
                            for sep in ['\t', '|', ',', ':']:
                                parts = line.split(sep)
                                if len(parts) >= 2:
                                    break
                            
                            if parts and len(parts) >= 2:
                                img_name = parts[0].strip()
                                latex = sep.join(parts[1:]).strip()
                            elif parts and len(parts) == 1:
                                img_name = None
                                latex = parts[0].strip()
                            else:
                                parts = line.split(None, 1)
                                if len(parts) >= 2:
                                    img_name = parts[0].strip()
                                    latex = parts[1].strip()
                                else:
                                    continue
                            
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
        if os.path.isabs(img_name) and os.path.exists(img_name):
            return img_name
        
        path = os.path.join(label_dir, img_name)
        if os.path.exists(path):
            return path
        
        search_dirs = [root_dir, label_dir]
        for name in ['images', 'train', 'test', 'val', 'data', 'formula_images']:
            d = os.path.join(root_dir, name)
            if os.path.isdir(d):
                search_dirs.append(d)
        
        for search_dir in search_dirs:
            for sub_root, sub_dirs, sub_files in os.walk(search_dir):
                if img_name.lower() in [f.lower() for f in sub_files]:
                    return os.path.join(sub_root, img_name)
                
                base_no_ext = os.path.splitext(img_name)[0]
                for f in sub_files:
                    if os.path.splitext(f)[0].lower() == base_no_ext.lower():
                        if any(f.lower().endswith(ext) for ext in self.IMG_EXTENSIONS):
                            return os.path.join(sub_root, f)
        
        return None

    def _find_hme100k_label(self, img_base: str, root_dir: str, formula_dirs: List[str]) -> Optional[str]:
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
        found = []
        for name in names:
            for r, dirs, files in os.walk(root):
                for d in dirs:
                    if d.lower() == name.lower():
                        found.append(os.path.join(r, d))
        return found

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