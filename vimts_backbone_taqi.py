import torch
import torch.nn as nn
import torchvision.models as models
import math
from torch.nn.init import xavier_uniform_, constant_

# ==============================================================================
# LIGHTWEIGHT BACKBONE - SINGLE SCALE FOR T4
# ==============================================================================

class ResNet50BackboneSingleScale(nn.Module):
    """
    ResNet50 with SINGLE SCALE output (C4 only, stride 16)
    Much more memory efficient for T4 GPU
    """
    def __init__(self, pretrained=True, output_dim=128):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Use layers up to layer3 (C4, stride 16, 1024 channels)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels (C4)
        
        # Don't use layer4 to save memory
        # Channel reduction to lightweight dimension
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(1024, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # 1/16 resolution, 1024 channels
        x = self.channel_reducer(x)  # Reduce to output_dim
        return x


# ==============================================================================
# SIMPLIFIED POSITIONAL ENCODING
# ==============================================================================

class PositionalEncoding2D(nn.Module):
    """Lightweight 2D Positional Encoding"""
    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Create positional grids
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(0).repeat(H, 1)
        
        # Normalize
        y_embed = y_embed / H * self.scale
        x_embed = x_embed / W * self.scale
        
        # Create sinusoidal encodings
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        
        pos = torch.cat((pos_y, pos_x), dim=-1).permute(2, 0, 1).unsqueeze(0)  # (1, 2*num_pos_feats, H, W)
        
        # Match channel dimension
        if pos.shape[1] < C:
            pos = torch.cat([pos, torch.zeros(1, C - pos.shape[1], H, W, device=x.device)], dim=1)
        elif pos.shape[1] > C:
            pos = pos[:, :C, :, :]
        
        return x + pos.expand(B, -1, -1, -1)


# ==============================================================================
# LIGHTWEIGHT REM
# ==============================================================================

class LightweightREM(nn.Module):
    """Simplified Receptive Enhancement Module"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Two branches only
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        out = self.output_conv(out)
        return out + x  # Residual connection


# ==============================================================================
# STANDARD TRANSFORMER ENCODER (NO DEFORMABLE ATTENTION)
# ==============================================================================

class TransformerEncoderLayer(nn.Module):
    """Standard Transformer Encoder Layer - much lighter than deformable"""
    def __init__(self, d_model=128, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerEncoder(nn.Module):
    """Stack of Transformer Encoder Layers"""
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


# ==============================================================================
# SIMPLIFIED QUERY INITIALIZATION
# ==============================================================================

class SimpleQueryInitialization(nn.Module):
    """Lightweight Query Initialization without RoI operations"""
    def __init__(self, feature_dim, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        
        # Coarse bbox prediction
        self.bbox_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 5, kernel_size=1)  # 4 bbox + 1 score
        )
        
        # Learnable query embeddings (simpler than RoI extraction)
        self.detection_queries = nn.Embedding(num_queries, feature_dim)
        self.recognition_queries = nn.Embedding(num_queries, feature_dim)

    def forward(self, features):
        B, C, H, W = features.shape
        
        # Predict coarse bounding boxes
        coarse_pred = self.bbox_predictor(features)
        coarse_pred = coarse_pred.permute(0, 2, 3, 1).reshape(B, H * W, 5)
        coarse_pred = coarse_pred.sigmoid()
        
        coarse_bboxes = coarse_pred[..., :4]
        coarse_scores = coarse_pred[..., 4:5]
        
        # Select top-N
        top_scores, top_indices = torch.topk(coarse_scores.squeeze(-1), self.num_queries, dim=1)
        top_boxes = torch.gather(coarse_bboxes, 1, top_indices.unsqueeze(-1).expand(-1, -1, 4))
        
        # Use learnable embeddings for queries
        det_queries = self.detection_queries.weight.unsqueeze(0).expand(B, -1, -1)
        rec_queries = self.recognition_queries.weight.unsqueeze(0).expand(B, -1, -1)
        
        coarse_bboxes_and_scores = torch.cat([top_boxes, top_scores.unsqueeze(-1)], dim=-1)
        
        return det_queries, rec_queries, coarse_bboxes_and_scores


# ==============================================================================
# MAIN MODULE - T4 OPTIMIZED
# ==============================================================================

class VimTSModule1(nn.Module):
    """
    T4-Optimized VimTS Module 1:
    - Single scale features (no multi-scale)
    - Lightweight dimensions (128 instead of 256)
    - Standard attention (no deformable)
    - Fewer queries (50 instead of 100)
    """
    def __init__(self, resnet_pretrained=True,
                 feature_dim=128,  # Reduced from 256
                 transformer_num_heads=4,  # Reduced from 8
                 transformer_num_layers=2,  # Reduced from 3
                 num_queries=50):  # Reduced from 100
        super().__init__()
        
        # Single-scale lightweight backbone
        self.resnet_backbone = ResNet50BackboneSingleScale(
            pretrained=resnet_pretrained,
            output_dim=feature_dim
        )
        
        # Lightweight REM
        self.receptive_enhancement_module = LightweightREM(
            in_channels=feature_dim,
            out_channels=feature_dim
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(num_pos_feats=feature_dim // 2)
        
        # Standard transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=transformer_num_heads,
            dim_feedforward=feature_dim * 2,  # Reduced FFN dimension
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_num_layers
        )
        
        # Simplified query initialization
        self.query_init = SimpleQueryInitialization(
            feature_dim=feature_dim,
            num_queries=num_queries
        )

    def forward(self, img):
        # Extract single-scale features
        features = self.resnet_backbone(img)  # (B, feature_dim, H/16, W/16)
        
        # Apply REM
        features = self.receptive_enhancement_module(features)
        
        # Add positional encoding
        features_with_pos = self.pos_encoding(features)
        
        # Flatten for transformer
        B, C, H, W = features_with_pos.shape
        features_flat = features_with_pos.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Apply transformer encoder
        encoded_features = self.transformer_encoder(features_flat)
        
        # Initialize queries
        detection_queries, recognition_queries, coarse_bboxes_and_scores = \
            self.query_init(features)

        return {
            "encoded_image_features": encoded_features,
            "detection_queries": detection_queries,
            "recognition_queries": recognition_queries,
            "coarse_bboxes_and_scores": coarse_bboxes_and_scores
        }