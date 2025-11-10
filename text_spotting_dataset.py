import torch
from torchvision import transforms
from PIL import Image
import json
import os
import random
import numpy as np
import torch.nn.functional as F

class TextFileDataset(torch.utils.data.Dataset):
    """
    Dataset loader for text files with polygon annotations.
    Format: x1,y1,x2,y2,x3,y3,x4,y4,language,text
    Filters out Arabic text during training.
    """
    def __init__(self, gt_dir, img_dir, transform=None, 
                 max_recognition_seq_len=25, padding_value=96,
                 char_to_id=None, filter_languages=None):
        """
        Args:
            gt_dir: Directory containing ground truth .txt files
            img_dir: Directory containing images
            transform: Image transformations
            max_recognition_seq_len: Maximum sequence length for recognition
            padding_value: Padding index for recognition sequences
            char_to_id: Dictionary mapping characters to IDs
            filter_languages: List of languages to exclude (e.g., ['Arabic'])
        """
        self.gt_dir = gt_dir
        self.img_dir = img_dir
        self.transform = transform
        self.max_recognition_seq_len = max_recognition_seq_len
        self.padding_value = padding_value
        self.char_to_id = char_to_id
        self.filter_languages = filter_languages if filter_languages else []
        
        # Get all txt files
        self.gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
        self.gt_files.sort()
        
        print(f"\n{'='*80}")
        print(f"Loading TextFile Dataset")
        print(f"{'='*80}")
        print(f"Ground truth directory: {gt_dir}")
        print(f"Image directory: {img_dir}")
        print(f"Total annotation files: {len(self.gt_files)}")
        print(f"Filtering out languages: {self.filter_languages}")
        print(f"{'='*80}\n")
    
    def __len__(self):
        return len(self.gt_files)
    
    def _parse_annotation_line(self, line):
        """
        Parse a single annotation line.
        Format: x1,y1,x2,y2,x3,y3,x4,y4,language,text
        Returns: polygon_points, language, text
        """
        parts = line.strip().split(',')
        if len(parts) < 10:
            return None, None, None
        
        # Extract polygon coordinates (8 values)
        try:
            coords = [float(parts[i]) for i in range(8)]
        except ValueError:
            return None, None, None
        
        # Extract language and text
        language = parts[8]
        text = ','.join(parts[9:])  # Join remaining parts (text may contain commas)
        
        return coords, language, text
    
    def _text_to_char_ids(self, text):
        """Convert text string to list of character IDs."""
        if self.char_to_id is None:
            # If no char_to_id provided, use ASCII mapping (0-94 for chars)
            char_ids = []
            for char in text:
                # Map to 0-94 range (ASCII 32-126)
                char_id = ord(char) - 32
                if 0 <= char_id <= 94:
                    char_ids.append(char_id)
                else:
                    char_ids.append(95)  # Unknown character
            return char_ids
        else:
            # Use provided mapping
            char_ids = []
            for char in text:
                char_ids.append(self.char_to_id.get(char, 95))  # 95 for unknown
            return char_ids
    
    def __getitem__(self, idx):
        gt_filename = self.gt_files[idx]
        gt_path = os.path.join(self.gt_dir, gt_filename)
        
        # Derive image filename (assuming same name with image extension)
        img_basename = os.path.splitext(gt_filename)[0]
        # Try common image extensions
        img_filename = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_path = os.path.join(self.img_dir, img_basename + ext)
            if os.path.exists(potential_path):
                img_filename = img_basename + ext
                break
        
        if img_filename is None:
            raise FileNotFoundError(f"Image file not found for {gt_filename}")
        
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert('RGB')
        original_width, original_height = image.size
        
        # Parse annotations
        gt_bboxes = []
        gt_polygons = []
        gt_recs = []
        gt_labels = []
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                coords, language, text = self._parse_annotation_line(line)
                
                if coords is None:
                    continue
                
                # Filter out unwanted languages
                if language in self.filter_languages:
                    continue
                
                # Convert polygon to normalized coordinates
                normalized_poly = []
                for i in range(0, len(coords), 2):
                    normalized_poly.append(coords[i] / original_width)
                    normalized_poly.append(coords[i+1] / original_height)
                
                # Calculate bounding box from polygon
                x_coords = [coords[i] for i in range(0, 8, 2)]
                y_coords = [coords[i] for i in range(1, 8, 2)]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Convert to normalized cxcywh format
                width = x_max - x_min
                height = y_max - y_min
                cx = (x_min + width / 2) / original_width
                cy = (y_min + height / 2) / original_height
                w = width / original_width
                h = height / original_height
                
                gt_bboxes.append([cx, cy, w, h])
                gt_polygons.append(torch.tensor(normalized_poly, dtype=torch.float32))
                
                # Convert text to character IDs
                char_ids = self._text_to_char_ids(text)
                rec_tensor = torch.tensor(char_ids, dtype=torch.long)
                
                # Truncate or pad to max_recognition_seq_len
                if len(rec_tensor) > self.max_recognition_seq_len:
                    rec_tensor = rec_tensor[:self.max_recognition_seq_len]
                
                padded_rec_tensor = F.pad(
                    rec_tensor, 
                    (0, self.max_recognition_seq_len - len(rec_tensor)), 
                    value=self.padding_value
                )
                gt_recs.append(padded_rec_tensor)
                gt_labels.append(0)  # 'text' class
        
        # Convert to tensors
        gt_bboxes_tensor = torch.tensor(gt_bboxes, dtype=torch.float32) if gt_bboxes else torch.empty((0, 4), dtype=torch.float32)
        gt_labels_tensor = torch.tensor(gt_labels, dtype=torch.long) if gt_labels else torch.empty((0,), dtype=torch.long)
        gt_recs_tensor = torch.stack(gt_recs) if gt_recs else torch.empty((0, self.max_recognition_seq_len), dtype=torch.long)
        
        # Apply image transformations
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        
        # Create target dictionary
        target = {
            'image_id': idx,
            'file_name': img_filename,
            'original_size': torch.tensor([original_height, original_width]),
            'size': torch.tensor(image_tensor.shape[-2:]),
            'boxes': gt_bboxes_tensor,
            'labels': gt_labels_tensor,
            'recognition': gt_recs_tensor,
            'polygons': gt_polygons,
            'area': torch.tensor([bbox[2] * bbox[3] * original_width * original_height 
                                 for bbox in gt_bboxes], dtype=torch.float32),
            'iscrowd': torch.zeros(len(gt_bboxes), dtype=torch.bool)
        }
        
        return image_tensor, target
    
def build_vocabulary_from_text_files(gt_dir, filter_languages=None, save_vocab_path=None):
    """
    Build vocabulary from text annotation files.
    
    Args:
        gt_dir: Directory containing ground truth .txt files
        filter_languages: List of languages to exclude (e.g., ['Arabic'])
        save_vocab_path: Path to save vocabulary
    
    Returns: id_to_char dict, char_to_id dict, vocab_size, padding_idx
    """
    filter_languages = filter_languages if filter_languages else []
    all_chars = set()
    sample_texts = []
    
    # Read all text files
    txt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
    
    print(f"\n{'='*80}")
    print(f"VOCABULARY BUILDING FROM TEXT FILES")
    print(f"{'='*80}")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Total files: {len(txt_files)}")
    print(f"Filtering languages: {filter_languages}")
    
    for txt_file in txt_files:
        with open(os.path.join(gt_dir, txt_file), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 10:
                    continue
                
                language = parts[8]
                text = ','.join(parts[9:])
                
                # Skip filtered languages
                if language in filter_languages:
                    continue
                
                all_chars.update(text)
                if len(sample_texts) < 10:
                    sample_texts.append(text)
    
    print(f"\nTotal unique characters found: {len(all_chars)}")
    print(f"Sample texts:")
    for i, text in enumerate(sample_texts[:5]):
        print(f"  {i+1}: '{text}'")
    
    # Create character mapping (0-94 for printable ASCII)
    id_to_char = {}
    char_to_id = {}
    
    # Sort characters for consistent mapping
    sorted_chars = sorted(list(all_chars))
    
    for idx, char in enumerate(sorted_chars):
        if idx <= 94:  # Keep within 0-94 range
            id_to_char[idx] = char
            char_to_id[char] = idx
    
    # Add special tokens
    id_to_char[95] = '<unk>'
    id_to_char[96] = '<pad>'
    char_to_id['<unk>'] = 95
    char_to_id['<pad>'] = 96
    
    vocab_size = 97
    padding_idx = 96
    
    print(f"\n{'='*80}")
    print(f"VOCABULARY SUMMARY")
    print(f"{'='*80}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Padding index: {padding_idx}")
    print(f"Character mapping (first 30):")
    for i, (char_id, char) in enumerate(list(id_to_char.items())[:30]):
        print(f"  ID {char_id}: '{char}'")
    print(f"{'='*80}\n")
    
    # Save vocabulary
    if save_vocab_path:
        vocab_data = {
            'id_to_char': id_to_char,
            'char_to_id': char_to_id,
            'vocab_size': vocab_size,
            'padding_idx': padding_idx
        }
        import pickle
        with open(save_vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {save_vocab_path}")
    
    return id_to_char, char_to_id, vocab_size, padding_idx


def build_vocabulary_from_json_v2(json_path, save_vocab_path=None):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_char_ids = set()
    sample_recs = []
    
    # Collect all character IDs
    for ann in data['annotations']:
        if 'rec' in ann and ann['rec']:
            all_char_ids.update(ann['rec'])
            if len(sample_recs) < 10:
                sample_recs.append(ann['rec'])
    
    print(f"\n{'='*80}")
    print(f"VOCABULARY BUILDING - IMPROVED")
    print(f"{'='*80}")
    print(f"\nAll unique character IDs found: {sorted(list(all_char_ids))}")
    print(f"Total unique IDs: {len(all_char_ids)}")
    
    # Sample sequences for debugging
    print(f"\nSample recognition sequences:")
    for i, rec in enumerate(sample_recs[:5]):
        display_rec = rec[:15] if len(rec) > 15 else rec
        print(f"  Sample {i+1}: {display_rec}")
    
    # Create character mapping
    # Standard MLT format: IDs 0-94 map to ASCII 32-126 (printable characters)
    # ID 95: unknown/placeholder
    # ID 96: padding
    
    id_to_char = {}
    
    # Map printable ASCII characters (space to tilde)
    for char_id in sorted(list(all_char_ids)):
        if 0 <= char_id <= 94:
            # Map to printable ASCII: ID 0 → space (ASCII 32), ID 94 → tilde (ASCII 126)
            mapped_char = chr(char_id + 32)
            id_to_char[char_id] = mapped_char
        elif char_id == 95:
            id_to_char[char_id] = '<unk>'  # Unknown/placeholder
        elif char_id == 96:
            id_to_char[char_id] = '<pad>'  # Padding
        else:
            # Handle any unexpected IDs
            id_to_char[char_id] = f'<char_{char_id}>'
    
    # Reverse mapping
    char_to_id = {v: k for k, v in id_to_char.items()}
    
    # Vocab size should be max_id + 1
    vocab_size = max(all_char_ids) + 1 if all_char_ids else 97
    padding_idx = 96
    
    print(f"\n{'='*80}")
    print(f"VOCABULARY SUMMARY")
    print(f"{'='*80}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Padding index: {padding_idx}")
    print(f"Unique character IDs: {len(all_char_ids)}")
    
    # Display character mapping
    print(f"\nCharacter Mapping (showing all non-padding/placeholder):")
    meaningful_ids = sorted([i for i in all_char_ids if i not in [95, 96]])
    
    if meaningful_ids:
        print(f"\n  {'ID':<5} {'Char':<10} {'ASCII':<10} {'Description'}")
        print(f"  {'-'*50}")
        for char_id in meaningful_ids[:30]:  # Show first 30
            char = id_to_char[char_id]
            ascii_val = ord(char) if len(char) == 1 else '-'
            desc = get_char_description(char)
            print(f"  {char_id:<5} '{char}'  {str(ascii_val):<10} {desc}")
        
        if len(meaningful_ids) > 30:
            print(f"  ... and {len(meaningful_ids) - 30} more characters")
    
    # Test decoding some sample sequences
    print(f"\n{'='*80}")
    print(f"SAMPLE DECODED SEQUENCES")
    print(f"{'='*80}")
    for i, rec_ids in enumerate(sample_recs[:5]):
        decoded = ''
        for char_id in rec_ids:
            if char_id == 96:  # Padding
                break
            if char_id == 95:  # Unknown
                decoded += '<?>'
            elif char_id in id_to_char:
                decoded += id_to_char[char_id]
            else:
                decoded += f'<{char_id}>'
        
        print(f"Sample {i+1}:")
        print(f"  IDs: {rec_ids[:15]}")
        print(f"  Text: '{decoded}'")
    
    print(f"\n{'='*80}\n")
    
    # Optionally save vocabulary
    if save_vocab_path:
        vocab_data = {
            'id_to_char': id_to_char,
            'char_to_id': char_to_id,
            'vocab_size': vocab_size,
            'padding_idx': padding_idx
        }
        import pickle
        with open(save_vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {save_vocab_path}")
    
    return id_to_char, char_to_id, vocab_size, padding_idx


def get_char_description(char):
    """Helper function to describe characters"""
    if char == ' ':
        return 'space'
    elif char == '<unk>':
        return 'unknown/placeholder'
    elif char == '<pad>':
        return 'padding'
    elif char.isalpha():
        return 'letter'
    elif char.isdigit():
        return 'digit'
    elif char in '.,;:!?':
        return 'punctuation'
    elif char in '+-*/=':
        return 'math operator'
    elif char in '()[]{}':
        return 'bracket'
    else:
        return 'special'


def build_vocabulary_from_json(json_path):
    """
    Extracts all unique character IDs from the MLT dataset and creates a vocabulary.
    Handles the specific format where:
    - Character IDs range from 0-94 (actual characters)
    - ID 95 is typically unknown/placeholder
    - ID 96 is padding
    
    Returns: id_to_char dict, char_to_id dict, vocab_size, padding_idx
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_char_ids = set()
    sample_recs = []
    
    for ann in data['annotations']:
        if 'rec' in ann and ann['rec']:
            all_char_ids.update(ann['rec'])
            if len(sample_recs) < 5:
                sample_recs.append(ann['rec'])
    
    print(f"\n=== Vocabulary Analysis ===")
    print(f"All unique character IDs found: {sorted(list(all_char_ids))}")
    print(f"Sample recognition sequences:")
    for i, rec in enumerate(sample_recs[:5]):
        print(f"  Sample {i+1}: {rec[:10]}..." if len(rec) > 10 else f"  Sample {i+1}: {rec}")
    
    # Create character mapping
    id_to_char = {}
    
    # Sort IDs for consistent mapping
    char_ids_sorted = sorted(list(all_char_ids))
    
    for char_id in char_ids_sorted:
        if char_id == 96:
            id_to_char[char_id] = '<pad>'  # Padding token
        elif char_id == 95:
            id_to_char[char_id] = '<unk>'  # Unknown/placeholder token
        elif 0 <= char_id <= 94:
            # Map to printable ASCII characters
            # Common mapping: ID 0-94 maps to ASCII 32-126 (space to ~)
            # Adjust offset based on your specific encoding
            try:
                # Try standard ASCII mapping with space as first char
                mapped_char = chr(char_id + 32)
                # Ensure it's printable
                if 32 <= ord(mapped_char) <= 126:
                    id_to_char[char_id] = mapped_char
                else:
                    id_to_char[char_id] = f'<char_{char_id}>'
            except:
                id_to_char[char_id] = f'<char_{char_id}>'
        else:
            # Handle any unexpected IDs
            id_to_char[char_id] = f'<char_{char_id}>'
    
    char_to_id = {v: k for k, v in id_to_char.items()}
    
    # Vocab size should accommodate the maximum ID + 1
    vocab_size = max(all_char_ids) + 1 if all_char_ids else 97
    padding_idx = 96
    
    print(f"\nVocabulary built successfully:")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Padding index: {padding_idx}")
    print(f"  - Number of unique character IDs: {len(all_char_ids)}")
    print(f"Sample character mappings (first 15):")
    for i, (char_id, char) in enumerate(list(id_to_char.items())[:15]):
        print(f"  ID {char_id}: '{char}'")
    print("===========================\n")
    
    return id_to_char, char_to_id, vocab_size, padding_idx


class AdaptiveResize:
    """
    Adaptive resizing that maintains aspect ratio.
    Shorter side is resized to range [min_size, max_size], 
    and longer side is capped at max_long_side.
    Used during training for data augmentation.
    """
    def __init__(self, min_size=640, max_size=896, max_long_side=1600):
        self.min_size = min_size
        self.max_size = max_size
        self.max_long_side = max_long_side
    
    def __call__(self, img):
        w, h = img.size
        shorter_side = min(w, h)
        longer_side = max(w, h)
        
        # Randomly choose target size for shorter side during training
        target_shorter = random.randint(self.min_size, self.max_size)
        scale = target_shorter / shorter_side
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Cap the longer side
        if max(new_w, new_h) > self.max_long_side:
            scale = self.max_long_side / max(new_w, new_h)
            new_w = int(new_w * scale)
            new_h = int(new_h * scale)
        
        return img.resize((new_w, new_h), Image.BILINEAR)


class AdaptiveResizeTest:
    """
    Fixed resizing for testing/inference - uses fixed shorter side size.
    """
    def __init__(self, shorter_size=1000, max_long_side=1824):
        self.shorter_size = shorter_size
        self.max_long_side = max_long_side
    
    def __call__(self, img):
        w, h = img.size
        shorter_side = min(w, h)
        scale = self.shorter_size / shorter_side
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if max(new_w, new_h) > self.max_long_side:
            scale = self.max_long_side / max(new_w, new_h)
            new_w = int(new_w * scale)
            new_h = int(new_h * scale)
        
        return img.resize((new_w, new_h), Image.BILINEAR)


class TotalTextDataset(torch.utils.data.Dataset):
    """
    Dataset loader for MLT-format text spotting datasets.
    Supports both 'bezier_pts' (MLT format) and 'segmentation' (standard COCO format).
    """
    def __init__(self, json_path, img_dir, transform=None, 
                 max_recognition_seq_len=25, padding_value=96,
                 check_recognition_quality=True):
        self.img_dir = img_dir
        self.transform = transform
        self.max_recognition_seq_len = max_recognition_seq_len
        self.padding_value = padding_value
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.images = self.data['images']
        self.annotations = self.data['annotations']
        
        # Map image_id to annotations
        self.img_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.img_id_to_annotations:
                self.img_id_to_annotations[image_id] = []
            self.img_id_to_annotations[image_id].append(ann)
            
        self.image_infos = {img['id']: img for img in self.images}
        
        # Check recognition data quality
        if check_recognition_quality:
            self._check_recognition_quality()

    def _check_recognition_quality(self):
        """Analyze the quality of recognition annotations."""
        print("\n=== Recognition Data Quality Check ===")
        total_annotations = len(self.annotations)
        meaningful_count = 0
        placeholder_count = 0
        empty_count = 0
        
        for ann in self.annotations:
            if 'rec' not in ann or not ann['rec']:
                empty_count += 1
            else:
                # Check if recognition contains non-placeholder characters
                non_placeholder = [x for x in ann['rec'] if x not in [95, 96]]
                if len(non_placeholder) > 0:
                    meaningful_count += 1
                else:
                    placeholder_count += 1
        
        print(f"Total annotations: {total_annotations}")
        print(f"  - With meaningful text: {meaningful_count} ({100*meaningful_count/total_annotations:.1f}%)")
        print(f"  - Placeholder only (95/96): {placeholder_count} ({100*placeholder_count/total_annotations:.1f}%)")
        print(f"  - Empty/missing: {empty_count} ({100*empty_count/total_annotations:.1f}%)")
        
        if meaningful_count < total_annotations * 0.1:
            print("\n⚠️  WARNING: Less than 10% of annotations have meaningful recognition data!")
            print("   Consider training without recognition loss initially:")
            print("   losses_to_compute = ['labels', 'boxes', 'polygons', 'cardinality']")
        
        print("======================================\n")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_path = os.path.join(self.img_dir, img_filename)

        image = Image.open(img_path).convert('RGB')
        
        annotations = self.img_id_to_annotations.get(img_id, [])
        
        gt_bboxes = []
        gt_polygons = []
        gt_recs = []
        gt_labels = []
        gt_areas = []
        gt_iscrowds = []
        
        original_width, original_height = image.size
        
        for ann in annotations:
            # Bbox format in COCO is [x_min, y_min, width, height]
            x_min, y_min, width, height = ann['bbox']
            
            # Convert bbox to normalized cxcywh format [cx, cy, w, h]
            cx = (x_min + width / 2) / original_width
            cy = (y_min + height / 2) / original_height
            w = width / original_width
            h = height / original_height
            gt_bboxes.append([cx, cy, w, h])
            
            # Handle polygon data - support both 'bezier_pts' (MLT) and 'segmentation' (COCO)
            polygon_added = False
            
            # First, try 'bezier_pts' (MLT dataset format)
            if 'bezier_pts' in ann and ann['bezier_pts']:
                # bezier_pts is a flat list: [x1, y1, x2, y2, ..., x16, y16]
                bezier_pts = ann['bezier_pts']
                if len(bezier_pts) >= 2:  # At least one point
                    normalized_poly = []
                    for i in range(0, len(bezier_pts), 2):
                        if i + 1 < len(bezier_pts):
                            normalized_poly.append(bezier_pts[i] / original_width)
                            normalized_poly.append(bezier_pts[i+1] / original_height)
                    
                    if normalized_poly:
                        gt_polygons.append(torch.tensor(normalized_poly, dtype=torch.float32))
                        polygon_added = True
            
            # Fallback to 'segmentation' (standard COCO format)
            if not polygon_added and 'segmentation' in ann and ann['segmentation']:
                seg = ann['segmentation']
                if isinstance(seg, list) and len(seg) > 0:
                    # Handle polygon segmentation (list of coordinates)
                    if isinstance(seg[0], list):
                        poly = seg[0]  # Take first polygon if multiple
                    else:
                        poly = seg
                    
                    if len(poly) >= 2:
                        normalized_poly = []
                        for i in range(0, len(poly), 2):
                            if i + 1 < len(poly):
                                normalized_poly.append(poly[i] / original_width)
                                normalized_poly.append(poly[i+1] / original_height)
                        
                        if normalized_poly:
                            gt_polygons.append(torch.tensor(normalized_poly, dtype=torch.float32))
                            polygon_added = True
            
            # If still no polygon, create empty tensor
            if not polygon_added:
                gt_polygons.append(torch.empty((0,), dtype=torch.float32))

            # Handle recognition tokens
            if 'rec' in ann and ann['rec']:
                rec_list = ann['rec']
                rec_tensor = torch.tensor(rec_list, dtype=torch.long)
            else:
                # If no recognition data, create padding-only sequence
                rec_tensor = torch.full((1,), self.padding_value, dtype=torch.long)
            
            # Truncate or pad to max_recognition_seq_len
            if len(rec_tensor) > self.max_recognition_seq_len:
                rec_tensor = rec_tensor[:self.max_recognition_seq_len]
            
            padded_rec_tensor = F.pad(
                rec_tensor, 
                (0, self.max_recognition_seq_len - len(rec_tensor)), 
                value=self.padding_value
            )
            gt_recs.append(padded_rec_tensor)
            
            gt_labels.append(0)  # 'text' class is 0 (foreground)
            gt_areas.append(ann['area'])
            gt_iscrowds.append(ann['iscrowd'])
        
        # Convert to tensors
        gt_bboxes_tensor = torch.tensor(gt_bboxes, dtype=torch.float32) if gt_bboxes else torch.empty((0, 4), dtype=torch.float32)
        gt_labels_tensor = torch.tensor(gt_labels, dtype=torch.long) if gt_labels else torch.empty((0,), dtype=torch.long)
        gt_recs_tensor = torch.stack(gt_recs) if gt_recs else torch.empty((0, self.max_recognition_seq_len), dtype=torch.long)
        gt_areas_tensor = torch.tensor(gt_areas, dtype=torch.float32) if gt_areas else torch.empty((0,), dtype=torch.float32)
        gt_iscrowds_tensor = torch.tensor(gt_iscrowds, dtype=torch.bool) if gt_iscrowds else torch.empty((0,), dtype=torch.bool)

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        
        # Target dict contains all ground truth information in normalized format
        target = {
            'image_id': img_id,
            'file_name': img_filename,
            'original_size': torch.tensor([original_height, original_width]),  # H, W
            'size': torch.tensor(image_tensor.shape[-2:]),  # H, W after transform
            'boxes': gt_bboxes_tensor,  # Normalized cxcywh format
            'labels': gt_labels_tensor,  # Class labels (0 for text)
            'recognition': gt_recs_tensor,  # Padded sequence of char IDs
            'polygons': gt_polygons,  # List of tensors with normalized coordinates
            'area': gt_areas_tensor,
            'iscrowd': gt_iscrowds_tensor
        }

        return image_tensor, target


def collate_fn(batch):
    """
    Custom collate function for batching.
    Currently designed for batch_size=1 but can be extended.
    
    For DETR-style training, proper batching requires:
    - NestedTensor for images (handling different sizes)
    - List of target dicts (one per image)
    """
    # For batch_size=1
    images_tensor = batch[0][0].unsqueeze(0)  # (1, C, H, W)
    targets = [batch[0][1]]  # List of one target dict

    return images_tensor, targets


def analyze_dataset(json_path):
    """
    Utility function to analyze dataset statistics.
    Useful for understanding your data before training.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n=== Dataset Analysis ===")
    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    
    # Annotation per image statistics
    img_to_ann_count = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_to_ann_count[img_id] = img_to_ann_count.get(img_id, 0) + 1
    
    ann_counts = list(img_to_ann_count.values())
    print(f"\nAnnotations per image:")
    print(f"  - Min: {min(ann_counts)}")
    print(f"  - Max: {max(ann_counts)}")
    print(f"  - Average: {sum(ann_counts)/len(ann_counts):.2f}")
    
    # Check polygon format
    has_bezier = sum(1 for ann in data['annotations'] if 'bezier_pts' in ann)
    has_segmentation = sum(1 for ann in data['annotations'] if 'segmentation' in ann)
    
    print(f"\nPolygon format:")
    print(f"  - With 'bezier_pts': {has_bezier}")
    print(f"  - With 'segmentation': {has_segmentation}")
    
    # Image size statistics
    widths = [img['width'] for img in data['images']]
    heights = [img['height'] for img in data['images']]
    
    print(f"\nImage dimensions:")
    print(f"  - Width range: {min(widths)} - {max(widths)}")
    print(f"  - Height range: {min(heights)} - {max(heights)}")
    print(f"  - Average size: {sum(widths)/len(widths):.0f} x {sum(heights)/len(heights):.0f}")
    
    print("========================\n")


# Example usage
if __name__ == "__main__":
    # Analyze your dataset first
    json_path = '/content/drive/MyDrive/dataset_ts/mlt2017_sample/train.json'
    analyze_dataset(json_path)
    
    # Build vocabulary
    id_to_char, char_to_id, vocab_size, padding_idx = build_vocabulary_from_json(json_path)
    
    # Create dataset
    img_dir = '/content/drive/MyDrive/dataset_ts/mlt2017_sample/img'
    
    transform = transforms.Compose([
        AdaptiveResize(min_size=640, max_size=896, max_long_side=1600),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TotalTextDataset(
        json_path=json_path, 
        img_dir=img_dir, 
        transform=transform,
        max_recognition_seq_len=25,
        padding_value=padding_idx
    )
    
    print(f"Dataset created with {len(dataset)} images")
    
    # Test loading a sample
    print("\nTesting sample loading...")
    img_tensor, target = dataset[0]
    print(f"Image tensor shape: {img_tensor.shape}")
    print(f"Number of annotations: {len(target['boxes'])}")
    print(f"Boxes shape: {target['boxes'].shape}")
    print(f"Recognition shape: {target['recognition'].shape}")
    print(f"Number of polygons: {len(target['polygons'])}")