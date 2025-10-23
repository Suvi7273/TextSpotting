# text_spotting_dataset.py

import torch
from torchvision import transforms
from PIL import Image
import json
import os
import random
import numpy as np
import torch.nn.functional as F # For padding

class TotalTextDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_dir, transform=None, max_recognition_seq_len=25, padding_value=96):
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_path = os.path.join(self.img_dir, img_filename)

        image = Image.open(img_path).convert('RGB')
        
        annotations = self.img_id_to_annotations.get(img_id, [])
        
        gt_bboxes = [] # [cx, cy, w, h] normalized
        gt_polygons = [] # list of [x1, y1, x2, y2, ...] normalized
        gt_recs = [] # list of recognition token IDs (padded)
        gt_labels = [] # 0 for text (as foreground class), 1 for no_object (background) in DETR.
                       # Here, all actual text instances get label 0.
        gt_areas = [] # Area of the bounding box
        gt_iscrowds = [] # Is the annotation a crowd (for IoU handling)
        
        original_width, original_height = image.size # Original image size
        
        for ann in annotations:
            # Bbox format in COCO is [x_min, y_min, width, height]
            x_min, y_min, width, height = ann['bbox']
            
            # Convert bbox to normalized cxcywh format [cx, cy, w, h] where values are [0,1]
            cx = (x_min + width / 2) / original_width
            cy = (y_min + height / 2) / original_height
            w = width / original_width
            h = height / original_height
            gt_bboxes.append([cx, cy, w, h])
            
            # Normalize polygon points as well
            if ann['segmentation']:
                normalized_poly = []
                for i in range(0, len(ann['segmentation'][0]), 2):
                    normalized_poly.append(ann['segmentation'][0][i] / original_width)
                    normalized_poly.append(ann['segmentation'][0][i+1] / original_height)
                gt_polygons.append(torch.tensor(normalized_poly, dtype=torch.float32))
            else:
                gt_polygons.append(torch.empty((0,), dtype=torch.float32)) # Append empty tensor if no segmentation

            # Tokenize and pad recognition labels
            rec_tensor = torch.tensor(ann['rec'], dtype=torch.long)
            if len(rec_tensor) > self.max_recognition_seq_len:
                rec_tensor = rec_tensor[:self.max_recognition_seq_len]
            padded_rec_tensor = F.pad(rec_tensor, (0, self.max_recognition_seq_len - len(rec_tensor)), value=self.padding_value)
            gt_recs.append(padded_rec_tensor)
            
            gt_labels.append(0) # 'text' class is 0 (foreground)
            gt_areas.append(ann['area'])
            gt_iscrowds.append(ann['iscrowd'])
            
        gt_bboxes_tensor = torch.tensor(gt_bboxes, dtype=torch.float32) if gt_bboxes else torch.empty((0, 4), dtype=torch.float32)
        gt_labels_tensor = torch.tensor(gt_labels, dtype=torch.long) if gt_labels else torch.empty((0,), dtype=torch.long)
        gt_recs_tensor = torch.stack(gt_recs) if gt_recs else torch.empty((0, self.max_recognition_seq_len), dtype=torch.long)
        gt_polygons_tensor = gt_polygons # Already list of tensors from above
        gt_areas_tensor = torch.tensor(gt_areas, dtype=torch.float32) if gt_areas else torch.empty((0,), dtype=torch.float32)
        gt_iscrowds_tensor = torch.tensor(gt_iscrowds, dtype=torch.bool) if gt_iscrowds else torch.empty((0,), dtype=torch.bool)

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            
        # Target dict will contain normalized bboxes, labels, and padded recognition tokens
        target = {
            'image_id': img_id,
            'file_name': img_filename,
            'original_size': torch.tensor([original_height, original_width]), # H, W
            'size': torch.tensor(image_tensor.shape[-2:]), # H, W after transform
            'boxes': gt_bboxes_tensor, # Normalized cxcywh
            'labels': gt_labels_tensor, # Class labels (0 for text)
            'recognition': gt_recs_tensor, # Padded sequence of char IDs
            'polygons': gt_polygons_tensor, # Normalized x1y1x2y2... (list of tensors)
            'area': gt_areas_tensor,
            'iscrowd': gt_iscrowds_tensor
        }

        return image_tensor, target

def collate_fn(batch):
    # This collate_fn is for batch_size=1, but designed to be extendable.
    # For actual batching in DETR, padding images (NestedTensor) and
    # handling variable number of targets is more complex.
    
    images_tensor = batch[0][0].unsqueeze(0) # (1, C, H, W)
    targets = [batch[0][1]] # List of one target dict

    return images_tensor, targets
