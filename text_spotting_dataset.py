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
                 char_to_id=None, require_language=None):
        """
        Args:
            gt_dir: Directory containing ground truth .txt files
            img_dir: Directory containing images
            transform: Image transformations
            max_recognition_seq_len: Maximum sequence length for recognition
            padding_value: Padding index for recognition sequences
            char_to_id: Dictionary mapping characters to IDs
            filter_languages: List of languages to exclude (e.g., ['Arabic'])
            require_language: If set, only include images with at least one annotation in this language (e.g., 'Latin')
        """
        self.gt_dir = gt_dir
        self.img_dir = img_dir
        self.transform = transform
        self.max_recognition_seq_len = max_recognition_seq_len
        self.padding_value = padding_value
        self.char_to_id = char_to_id
        self.require_language = require_language  # NEW
        
        # Get all txt files
        all_gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
        all_gt_files.sort()
        
        # NEW: Filter files based on language requirements
        self.gt_files = []
        skipped_no_latin = 0
        skipped_no_annotations = 0
        
        for gt_file in all_gt_files:
            gt_path = os.path.join(gt_dir, gt_file)
            has_required_language = False
            has_valid_annotations = False
            
            with open(gt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 10:
                        continue
                    
                    language = parts[8]
                    
                    if self.require_language and language != self.require_language:
                        continue
                    
                    has_valid_annotations = True
                    
                    # Check if required language is present
                    if self.require_language:
                        if language == self.require_language:
                            has_required_language = True
                            break  # Found required language, no need to continue
                    else:
                        has_required_language = True  # No requirement, accept all
            
            # Include file only if it meets requirements
            if self.require_language:
                if has_required_language:
                    self.gt_files.append(gt_file)
                else:
                    if has_valid_annotations:
                        skipped_no_latin += 1
                    else:
                        skipped_no_annotations += 1
            else:
                if has_valid_annotations:
                    self.gt_files.append(gt_file)
                else:
                    skipped_no_annotations += 1
        
        print(f"\n{'='*80}")
        print(f"Loading TextFile Dataset")
        print(f"{'='*80}")
        print(f"Ground truth directory: {gt_dir}")
        print(f"Image directory: {img_dir}")
        print(f"Total annotation files found: {len(all_gt_files)}")
        print(f"Filtering out languages: {self.filter_languages}")
        if self.require_language:
            print(f"Requiring language: {self.require_language}")
            print(f"  - Files WITH {self.require_language}: {len(self.gt_files)}")
            print(f"  - Files WITHOUT {self.require_language}: {skipped_no_latin}")
            print(f"  - Files with no valid annotations: {skipped_no_annotations}")
        else:
            print(f"Files with valid annotations: {len(self.gt_files)}")
            print(f"Files skipped (no valid annotations): {skipped_no_annotations}")
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
    
def build_vocabulary_from_text_files(gt_dir, required_language=None, save_vocab_path=None):
    all_chars = set()
    sample_texts = []

    txt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
    for txt_file in txt_files:
        with open(os.path.join(gt_dir, txt_file), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 10:
                    continue

                language = parts[8]
                text = ','.join(parts[9:])

                # Only include text from the required language (if set)
                if required_language and language != required_language:
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

    id_to_char = {i - 32: chr(i) for i in range(32, 127)}  # 0–94 for ASCII
    char_to_id = {chr(i): i - 32 for i in range(32, 127)}

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
    print(f"Character mapping:")
    for i, (char_id, char) in enumerate(list(id_to_char.items())[:]):
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