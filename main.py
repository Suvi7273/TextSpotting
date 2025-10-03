# Complete VimTS Implementation - All Components
# File: complete_vimts.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as transforms

import numpy as np
import json
import os
from PIL import Image
from tqdm import tqdm
import math
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. MEMORY-OPTIMIZED BACKBONE
# ============================================================================

class MemoryEfficientResNet50(nn.Module):
    """Memory-efficient ResNet50 with gradient checkpointing"""
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision.models as models
        
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels, /4
        self.layer2 = resnet.layer2  # 512 channels, /8  
        self.layer3 = resnet.layer3  # 1024 channels, /16
        self.layer4 = resnet.layer4  # 2048 channels, /32
        
        # Feature pyramid for efficient multi-scale features
        self.fpn = FeaturePyramidNetwork()
        
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers with gradient checkpointing
        if self.training:
            c2 = checkpoint(self.layer1, x, use_reentrant=False)
            c3 = checkpoint(self.layer2, c2, use_reentrant=False)
            c4 = checkpoint(self.layer3, c3, use_reentrant=False)
            c5 = checkpoint(self.layer4, c4, use_reentrant=False)
        else:
            c2 = self.layer1(x)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
        
        # Feature pyramid network
        features = self.fpn([c2, c3, c4, c5])
        
        return features

class FeaturePyramidNetwork(nn.Module):
    """Efficient Feature Pyramid Network"""
    def __init__(self):
        super().__init__()
        
        # Lateral connections
        self.lateral_c5 = nn.Conv2d(2048, 256, 1)
        self.lateral_c4 = nn.Conv2d(1024, 256, 1)
        self.lateral_c3 = nn.Conv2d(512, 256, 1)
        self.lateral_c2 = nn.Conv2d(256, 256, 1)
        
        # Output convolution
        self.output_conv = nn.Conv2d(256, 256, 3, padding=1)
        
    def forward(self, features):
        c2, c3, c4, c5 = features
        
        # Top-down pathway
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        
        # Use P4 as main feature (good balance of resolution and semantics)
        enhanced_features = self.output_conv(p4)
        
        return enhanced_features

# ============================================================================
# 2. EFFICIENT TRANSFORMER COMPONENTS
# ============================================================================

class EfficientTransformerEncoder(nn.Module):
    """Memory-efficient transformer encoder"""
    def __init__(self, d_model=256, nhead=8, num_layers=3, dim_feedforward=1024):
        super().__init__()
        
        self.layers = nn.ModuleList([
            EfficientTransformerLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.register_buffer('pos_embed', self._create_pos_embedding(1024, d_model))
        
    def _create_pos_embedding(self, max_len, d_model):
        """Create sinusoidal positional embeddings"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, features):
        B, C, H, W = features.shape
        
        # Flatten spatial dimensions
        x = features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        seq_len = x.shape[1]
        
        # Add positional encoding
        if seq_len <= self.pos_embed.shape[1]:
            pos_embed = self.pos_embed[:, :seq_len, :]
        else:
            # Interpolate for longer sequences
            pos_embed = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=seq_len, 
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
        
        x = x + pos_embed
        
        # Apply transformer layers
        for layer in self.layers:
            if self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        
        x = self.norm(x)
        
        # Reshape back to spatial format
        enhanced_features = x.permute(0, 2, 1).view(B, C, H, W)
        
        return enhanced_features

class EfficientTransformerLayer(nn.Module):
    """Efficient transformer layer with windowed attention"""
    def __init__(self, d_model, nhead, dim_feedforward, window_size=64):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        self.window_size = window_size
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Windowed attention for long sequences
        if x.shape[1] > 256:  # If sequence too long
            x_attended = self._windowed_attention(x)
        else:
            x_attended, _ = self.self_attn(x, x, x)
        
        # Transformer block
        x = x + self.dropout(x_attended)
        x = self.norm1(x)
        
        x_ffn = self.ffn(x)
        x = x + self.dropout(x_ffn)
        x = self.norm2(x)
        
        return x
    
    def _windowed_attention(self, x):
        """Apply attention in windows to save memory"""
        B, L, D = x.shape
        window_size = min(self.window_size, L)
        
        if L <= window_size:
            attended, _ = self.self_attn(x, x, x)
            return attended
        
        # Process in windows
        num_windows = (L + window_size - 1) // window_size
        attended_parts = []
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min(start_idx + window_size, L)
            
            window = x[:, start_idx:end_idx, :]
            attended_window, _ = self.self_attn(window, window, window)
            attended_parts.append(attended_window)
        
        return torch.cat(attended_parts, dim=1)

# ============================================================================
# 3. QUERY INITIALIZATION
# ============================================================================

class SmartQueryInitialization(nn.Module):
    """Intelligent query initialization based on feature analysis"""
    def __init__(self, feature_dim=256, num_detection=100, num_recognition=25):
        super().__init__()
        
        self.num_detection = num_detection
        self.num_recognition = num_recognition
        self.total_queries = num_detection + num_recognition
        
        # Text detection head for smart query placement
        self.text_detector = nn.Sequential(
            nn.Conv2d(feature_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
        # Learnable query embeddings
        self.detection_queries = nn.Embedding(num_detection, feature_dim)
        self.recognition_queries = nn.Embedding(num_recognition, feature_dim)
        
        # Query enhancement network
        self.query_enhancer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, features):
        B, C, H, W = features.shape
        
        # Detect text regions
        text_prob_map = self.text_detector(features)  # [B, 1, H, W]
        
        # Sample informative features
        informative_features = self._sample_informative_features(features, text_prob_map)
        
        # Get base queries
        det_queries = self.detection_queries.weight.unsqueeze(0).expand(B, -1, -1)
        rec_queries = self.recognition_queries.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Enhance queries with informative features
        enhanced_det = self._enhance_queries(det_queries, informative_features)
        enhanced_rec = self._enhance_queries(rec_queries, informative_features)
        
        # Combine all queries
        all_queries = torch.cat([enhanced_det, enhanced_rec], dim=1)
        
        return all_queries, text_prob_map
    
    def _sample_informative_features(self, features, prob_map, num_samples=50):
        """Sample features from high-probability regions"""
        B, C, H, W = features.shape
        
        # Flatten for sampling
        features_flat = features.view(B, C, -1)  # [B, C, H*W]
        prob_flat = prob_map.view(B, -1)  # [B, H*W]
        
        sampled_features = []
        
        for b in range(B):
            prob_b = prob_flat[b]
            feat_b = features_flat[b]  # [C, H*W]
            
            if prob_b.sum() > 0:
                # Sample based on text probability
                try:
                    indices = torch.multinomial(prob_b + 1e-8, num_samples, replacement=True)
                except:
                    indices = torch.randperm(prob_b.shape[0], device=prob_b.device)[:num_samples]
            else:
                indices = torch.randperm(prob_b.shape[0], device=prob_b.device)[:num_samples]
            
            sampled = feat_b[:, indices].mean(dim=1)  # [C]
            sampled_features.append(sampled)
        
        return torch.stack(sampled_features)  # [B, C]
    
    def _enhance_queries(self, queries, informative_features):
        """Enhance queries with informative features"""
        B, N, C = queries.shape
        
        # Expand informative features
        expanded_features = informative_features.unsqueeze(1).expand(-1, N, -1)
        
        # Concatenate and enhance
        combined = torch.cat([queries, expanded_features], dim=-1)
        enhanced = self.query_enhancer(combined)
        
        return enhanced

# ============================================================================
# 4. DECODER
# ============================================================================

class EfficientDecoder(nn.Module):
    """Memory-efficient decoder"""
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        
        self.layers = nn.ModuleList([
            EfficientDecoderLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, queries, memory):
        """
        Args:
            queries: [B, N, C] query embeddings
            memory: [B, C, H, W] visual features
        """
        # Flatten memory for attention
        B, C, H, W = memory.shape
        memory_flat = memory.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        output = queries
        
        for layer in self.layers:
            if self.training:
                output = checkpoint(layer, output, memory_flat, use_reentrant=False)
            else:
                output = layer(output, memory_flat)
        
        output = self.norm(output)
        
        return output

class EfficientDecoderLayer(nn.Module):
    """Efficient decoder layer"""
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        
        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        
        # Cross-attention with visual features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, tgt, memory):
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        
        # Cross-attention (sample memory if too large)
        if memory.shape[1] > 512:
            indices = torch.randperm(memory.shape[1], device=memory.device)[:512]
            sampled_memory = memory[:, indices, :]
        else:
            sampled_memory = memory
        
        tgt2, _ = self.cross_attn(tgt, sampled_memory, sampled_memory)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        
        # Feed-forward
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))
        
        return tgt

# ============================================================================
# 5. COMPLETE VIMTS MODEL
# ============================================================================

class MemoryOptimizedVimTS(nn.Module):
    """Complete Memory-Optimized VimTS Model"""
    def __init__(self, num_classes=2, vocab_size=100, max_text_len=25):
        super().__init__()
        
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        self.num_detection_queries = 100
        self.num_recognition_queries = 25
        
        # Core modules
        self.backbone = MemoryEfficientResNet50(pretrained=True)
        self.feature_enhancer = EfficientTransformerEncoder(256, 8, 3, 1024)
        self.query_initializer = SmartQueryInitialization(256, 100, 25)
        self.decoder = EfficientDecoder(256, 8, 6, 1024)
        
        # Prediction heads
        self.class_head = nn.Linear(256, num_classes + 1)  # +1 for background
        self.bbox_head = nn.Linear(256, 4)
        self.polygon_head = nn.Linear(256, 16)  # 8 points x 2 coords
        
        # Text recognition head (only for recognition queries)
        self.text_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, max_text_len * vocab_size)
        )
        
    def forward(self, images):
        B = images.shape[0]
        
        # Feature extraction
        backbone_features = self.backbone(images)
        enhanced_features = self.feature_enhancer(backbone_features)
        
        # Query initialization
        queries, coarse_text_map = self.query_initializer(enhanced_features)
        
        # Decoder
        decoded_queries = self.decoder(queries, enhanced_features)
        
        # Predictions
        pred_logits = self.class_head(decoded_queries)
        pred_boxes = self.bbox_head(decoded_queries).sigmoid()
        pred_polygons = self.polygon_head(decoded_queries).sigmoid()
        
        # Text predictions (only for recognition queries)
        recognition_queries = decoded_queries[:, -self.num_recognition_queries:]
        text_logits = self.text_head(recognition_queries)
        text_logits = text_logits.view(B, self.num_recognition_queries, self.max_text_len, self.vocab_size)
        
        # Pad text predictions for all queries
        full_text_logits = torch.zeros(B, decoded_queries.shape[1], self.max_text_len, 
                                     self.vocab_size, device=decoded_queries.device)
        full_text_logits[:, -self.num_recognition_queries:] = text_logits
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'pred_polygons': pred_polygons,
            'pred_texts': full_text_logits,
            'coarse_text_map': coarse_text_map
        }

# ============================================================================
# 6. LOSS FUNCTION
# ============================================================================

class VimTSLoss(nn.Module):
    """Simplified but effective loss function"""
    def __init__(self, weight_class=2.0, weight_bbox=5.0, weight_polygon=1.0, weight_text=1.0):
        super().__init__()
        
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_polygon = weight_polygon
        self.weight_text = weight_text
        
    def forward(self, predictions, targets):
        """Compute loss with simplified matching"""
        device = predictions['pred_logits'].device
        batch_size = len(targets)
        
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        # Classification loss
        class_loss = self._classification_loss(predictions['pred_logits'], targets)
        
        # Bounding box loss
        bbox_loss = self._bbox_loss(predictions['pred_boxes'], targets)
        
        # Polygon loss
        polygon_loss = self._polygon_loss(predictions['pred_polygons'], targets)
        
        # Text loss
        text_loss = self._text_loss(predictions['pred_texts'], targets)
        
        # Combine losses
        total_loss = (self.weight_class * class_loss +
                     self.weight_bbox * bbox_loss +
                     self.weight_polygon * polygon_loss +
                     self.weight_text * text_loss)
        
        loss_dict = {
            'total_loss': total_loss,
            'class_loss': class_loss,
            'bbox_loss': bbox_loss,
            'polygon_loss': polygon_loss,
            'text_loss': text_loss
        }
        
        return total_loss, loss_dict
    
    def _classification_loss(self, pred_logits, targets):
        """Simplified classification loss"""
        B, N, C = pred_logits.shape
        device = pred_logits.device
        
        # Create target tensor (all background initially)
        target_classes = torch.zeros(B, N, dtype=torch.long, device=device)
        
        # Assign positive labels to first few queries for each target
        for b, target in enumerate(targets):
            labels = target['labels']
            if len(labels) > 0:
                num_assign = min(len(labels), N)
                target_classes[b, :num_assign] = labels[:num_assign]
        
        # Cross-entropy loss
        pred_flat = pred_logits.view(-1, C)
        target_flat = target_classes.view(-1)
        
        return F.cross_entropy(pred_flat, target_flat, reduction='mean')
    
    def _bbox_loss(self, pred_boxes, targets):
        """Simplified bbox loss"""
        total_loss = torch.tensor(0.0, device=pred_boxes.device)
        num_pos = 0
        
        for b, target in enumerate(targets):
            boxes = target['boxes']
            if len(boxes) > 0:
                num_boxes = min(len(boxes), pred_boxes.shape[1])
                if num_boxes > 0:
                    pred_b = pred_boxes[b, :num_boxes]
                    target_b = boxes[:num_boxes].to(pred_boxes.device)
                    total_loss += F.l1_loss(pred_b, target_b, reduction='sum')
                    num_pos += num_boxes
        
        if num_pos == 0:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        
        return total_loss / num_pos
    
    def _polygon_loss(self, pred_polygons, targets):
        """Simplified polygon loss"""
        total_loss = torch.tensor(0.0, device=pred_polygons.device)
        num_pos = 0
        
        for b, target in enumerate(targets):
            if 'polygons' in target and len(target['polygons']) > 0:
                polygons = target['polygons']
                num_polygons = min(len(polygons), pred_polygons.shape[1])
                if num_polygons > 0:
                    pred_b = pred_polygons[b, :num_polygons]
                    target_b = polygons[:num_polygons].to(pred_polygons.device)
                    total_loss += F.l1_loss(pred_b, target_b, reduction='sum')
                    num_pos += num_polygons
        
        if num_pos == 0:
            return torch.tensor(0.0, device=pred_polygons.device, requires_grad=True)
        
        return total_loss / num_pos
    
    def _text_loss(self, pred_texts, targets):
        """Simplified text loss"""
        total_loss = torch.tensor(0.0, device=pred_texts.device)
        num_pos = 0
        
        # Only compute loss for recognition queries (last 25)
        pred_rec_texts = pred_texts[:, -25:]  # [B, 25, max_len, vocab_size]
        
        for b, target in enumerate(targets):
            if 'texts' in target and len(target['texts']) > 0:
                texts = target['texts']
                num_texts = min(len(texts), 25)
                if num_texts > 0:
                    pred_b = pred_rec_texts[b, :num_texts]  # [num_texts, max_len, vocab_size]
                    target_b = texts[:num_texts].to(pred_texts.device)  # [num_texts, max_len]
                    
                    # Flatten for cross-entropy
                    pred_flat = pred_b.reshape(-1, pred_b.shape[-1])
                    target_flat = target_b.reshape(-1)
                    
                    # Ignore padding tokens (assume 0 is padding)
                    mask = target_flat != 0
                    if mask.sum() > 0:
                        loss_b = F.cross_entropy(pred_flat[mask], target_flat[mask], reduction='sum')
                        total_loss += loss_b
                        num_pos += mask.sum().item()
        
        if num_pos == 0:
            return torch.tensor(0.0, device=pred_texts.device, requires_grad=True)
        
        return total_loss / num_pos

# ============================================================================
# 7. DATASET LOADING
# ============================================================================

class VimTSDataset(Dataset):
    """VimTS Dataset with memory-efficient loading"""
    def __init__(self, dataset_path, split='train', image_size=(640, 640), 
                 max_text_len=25, vocab_size=100):
        self.dataset_path = dataset_path
        self.split = split
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        
        # Paths
        self.annotation_file = os.path.join(dataset_path, 'totaltext', f'{split}.json')
        self.image_dir = os.path.join(dataset_path, 'totaltext', 'train_images')
        
        # Load annotations
        print(f"Loading {split} annotations from {self.annotation_file}")
        with open(self.annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image ID to filename mapping
        self.images = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Group annotations by image
        self.image_to_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())
        print(f"Loaded {len(self.image_ids)} images for {split}")
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, self.images[image_id])
        
        try:
            image = Image.open(image_path).convert('RGB')
            original_w, original_h = image.size
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create dummy image
            image = torch.randn(3, *self.image_size)
            original_w, original_h = self.image_size
        
        # Get annotations for this image
        annotations = self.image_to_annotations.get(image_id, [])
        
        # Process annotations
        labels = []
        boxes = []
        polygons = []
        texts = []
        
        for ann in annotations:
            # Label (category_id)
            labels.append(ann.get('category_id', 1))
            
            # Bounding box - normalize to [0, 1]
            x, y, w, h = ann['bbox']
            norm_box = [
                x / original_w,
                y / original_h, 
                (x + w) / original_w,
                (y + h) / original_h
            ]
            boxes.append(norm_box)
            
            # Polygon
            if 'segmentation' in ann and len(ann['segmentation']) > 0:
                # Take first polygon
                seg = ann['segmentation'][0]
                poly = np.array(seg, dtype=np.float32).reshape(-1, 2)
                
                # Normalize coordinates
                poly[:, 0] = poly[:, 0] / original_w  # x coordinates
                poly[:, 1] = poly[:, 1] / original_h  # y coordinates
                
                # Flatten and pad/crop to 16 values (8 points)
                poly_flat = poly.flatten()
                if len(poly_flat) >= 16:
                    poly_flat = poly_flat[:16]
                else:
                    poly_flat = np.pad(poly_flat, (0, 16 - len(poly_flat)), mode='constant', constant_values=0)
                
                polygons.append(poly_flat)
            else:
                # Default polygon (corners of bbox)
                x1, y1, x2, y2 = norm_box
                poly_flat = np.array([x1, y1, x2, y1, x2, y2, x1, y2] * 2, dtype=np.float32)[:16]
                polygons.append(poly_flat)
            
            # Text tokens
            text_str = ann.get('rec', '')
            if isinstance(text_str, str):
                # Convert characters to token IDs
                tokens = [min(ord(c), self.vocab_size - 1) for c in text_str[:self.max_text_len]]
            elif isinstance(text_str, list):
                tokens = [min(int(t), self.vocab_size - 1) for t in text_str[:self.max_text_len]]
            else:
                tokens = []
            
            # Pad to max length
            tokens = tokens + [0] * (self.max_text_len - len(tokens))
            texts.append(tokens[:self.max_text_len])
        
        # Convert to tensors
        if len(labels) == 0:
            # If no annotations, create empty tensors
            target = {
                'labels': torch.zeros(0, dtype=torch.long),
                'boxes': torch.zeros(0, 4, dtype=torch.float),
                'polygons': torch.zeros(0, 16, dtype=torch.float),
                'texts': torch.zeros(0, self.max_text_len, dtype=torch.long)
            }
        else:
            target = {
                'labels': torch.tensor(labels, dtype=torch.long),
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'polygons': torch.tensor(np.array(polygons, dtype=np.float32), dtype=torch.float32),
                'texts': torch.tensor(texts, dtype=torch.long)
            }
        
        return image, target

def collate_fn(batch):
    """Custom collate function for batching"""
    images, targets = zip(*batch)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Keep targets as list
    return images, list(targets)

# ============================================================================
# 8. TRAINING UTILITIES
# ============================================================================

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return {
            'allocated_gb': round(allocated, 2),
            'reserved_gb': round(reserved, 2),
            'max_allocated_gb': round(max_allocated, 2)
        }
    return {'status': 'CPU mode'}

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def setup_training():
    """Setup optimizations for training"""
    if torch.cuda.is_available():
        # Enable memory optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear cache
        torch.cuda.empty_cache()
        
        print("GPU training optimizations enabled")
    else:
        print("Using CPU mode")

# ============================================================================
# 9. TRAINING SCRIPT
# ============================================================================

def train_vimts(dataset_path, num_epochs=10, batch_size=1, learning_rate=1e-4,
                gradient_accumulation_steps=4, save_path="/content/drive/MyDrive/"):
    """
    Complete training function for VimTS
    
    Args:
        dataset_path: Path to dataset directory
        num_epochs: Number of training epochs
        batch_size: Batch size (keep small for T4)
        learning_rate: Learning rate
        gradient_accumulation_steps: Steps to accumulate gradients
        save_path: Path to save checkpoints
    """
    
    print("=" * 60)
    print("Starting VimTS Training with Memory Optimization")
    print("=" * 60)
    
    # Setup
    setup_training()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dataset and dataloader
    print("\nLoading dataset...")
    train_dataset = VimTSDataset(
        dataset_path=dataset_path,
        split='train',
        image_size=(640, 640),  # Keep original resolution for accuracy
        max_text_len=25,
        vocab_size=100
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = MemoryOptimizedVimTS(
        num_classes=2,
        vocab_size=100,
        max_text_len=25
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = VimTSLoss(
        weight_class=2.0,
        weight_bbox=5.0,
        weight_polygon=1.0,
        weight_text=1.0
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    model.train()
    train_losses = []
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print("Memory usage before training:")
    print(get_memory_usage())
    
    for epoch in range(num_epochs):
        epoch_losses = []
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            try:
                # Move to device
                images = images.to(device, non_blocking=True)
                targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} 
                          for target in targets]
                
                # Forward pass with mixed precision
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        predictions = model(images)
                        loss, loss_dict = criterion(predictions, targets)
                        loss = loss / gradient_accumulation_steps  # Scale loss
                else:
                    predictions = model(images)
                    loss, loss_dict = criterion(predictions, targets)
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                
                # Track loss
                epoch_losses.append(loss.item() * gradient_accumulation_steps)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'Avg': f'{np.mean(epoch_losses):.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'GPU': f'{get_memory_usage()["allocated_gb"]:.1f}GB'
                })
                
                # Clear memory periodically
                if (batch_idx + 1) % 10 == 0:
                    clear_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM Error at batch {batch_idx}")
                    print("Clearing cache and continuing...")
                    clear_memory()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Memory Usage: {get_memory_usage()}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(save_path, f"vimts_checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Clear memory after epoch
        clear_memory()
    
    # Save final model
    final_path = os.path.join(save_path, "vimts_final_model.pth")
    save_checkpoint(model, optimizer, num_epochs-1, train_losses[-1], final_path)
    
    print(f"\n{'='*60}")
    print("Training Completed Successfully!")
    print(f"Final model saved: {final_path}")
    print(f"Training losses: {train_losses}")
    print("="*60)
    
    return model, train_losses

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': {
            'num_classes': 2,
            'vocab_size': 100,
            'max_text_len': 25
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

# ============================================================================
# 10. TESTING/INFERENCE
# ============================================================================

def test_model(model_path, test_image_path=None, dataset_path=None):
    """Test trained model"""
    print("Testing trained VimTS model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MemoryOptimizedVimTS().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Test with single image
    if test_image_path and os.path.exists(test_image_path):
        print(f"Testing with image: {test_image_path}")
        
        # Load and preprocess image
        image = Image.open(test_image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        
    else:
        print("Using dummy image for testing...")
        image_tensor = torch.randn(1, 3, 640, 640, device=device)
    
    # Run inference
    print("Memory before inference:")
    print(get_memory_usage())
    
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                predictions = model(image_tensor)
        else:
            predictions = model(image_tensor)
    
    print("Memory after inference:")
    print(get_memory_usage())
    
    # Analyze predictions
    print("\nPrediction Results:")
    for key, tensor in predictions.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: {tensor.shape}")
    
    # Check confident detections
    class_logits = predictions['pred_logits'][0]  # [N, num_classes]
    class_probs = torch.softmax(class_logits, dim=-1)
    text_scores = class_probs[:, 1]  # Text class probability
    
    confident_detections = (text_scores > 0.5).sum().item()
    max_confidence = text_scores.max().item()
    
    print(f"\nDetection Analysis:")
    print(f"  Confident detections (>0.5): {confident_detections}")
    print(f"  Maximum confidence: {max_confidence:.3f}")
    print(f"  Average confidence: {text_scores.mean().item():.3f}")
    
    return predictions

# ============================================================================
# 11. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("VimTS - Complete Implementation")
    print("Choose an option:")
    print("1. Train new model")
    print("2. Test existing model") 
    print("3. Train and test")
    print("4. Quick memory test")
    
    choice = input("Enter choice (1/2/3/4): ").strip()
    
    # Default paths (modify these for your setup)
    dataset_path = "/content/drive/MyDrive/totaltext/"  # Adjust this path
    save_path = "/content/drive/MyDrive/"
    
    if choice == "1":
        # Training only
        print("Starting training...")
        model, losses = train_vimts(
            dataset_path=dataset_path,
            num_epochs=10,
            batch_size=1,  # Small batch for T4
            learning_rate=1e-4,
            gradient_accumulation_steps=8,  # Effective batch size 8
            save_path=save_path
        )
        
    elif choice == "2":
        # Testing only
        model_path = input("Enter model checkpoint path: ").strip()
        if os.path.exists(model_path):
            test_image_path = input("Enter test image path (or press Enter for dummy): ").strip()
            test_image_path = test_image_path if test_image_path and os.path.exists(test_image_path) else None
            
            test_model(model_path, test_image_path, dataset_path)
        else:
            print(f"Model not found: {model_path}")
    
    elif choice == "3":
        # Train and test
        print("Training model...")
        model, losses = train_vimts(
            dataset_path=dataset_path,
            num_epochs=10,
            batch_size=1,
            learning_rate=1e-4,
            gradient_accumulation_steps=8,
            save_path=save_path
        )
        
        print("\nTesting trained model...")
        final_model_path = os.path.join(save_path, "vimts_final_model.pth")
        test_model(final_model_path)
    
    elif choice == "4":
        # Quick memory test
        print("Running quick memory test...")
        setup_training()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MemoryOptimizedVimTS().to(device)
        
        print("Model created. Testing forward pass...")
        print("Memory before:", get_memory_usage())
        
        dummy_input = torch.randn(1, 3, 640, 640, device=device)
        
        model.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output = model(dummy_input)
            else:
                output = model(dummy_input)
        
        print("Memory after forward pass:", get_memory_usage())
        print("Forward pass successful!")
        
        for key, tensor in output.items():
            if isinstance(tensor, torch.Tensor):
                print(f"  {key}: {tensor.shape}")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
