# text_spotting_dataset.py - Updated for MLT Dataset Compatibility

import torch
from torchvision import transforms
from PIL import Image
import json
import os
import random
import numpy as np
import torch.nn.functional as F

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