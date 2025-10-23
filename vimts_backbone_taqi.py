# vimts_backbone_taqi.py

import torch
import torch.nn as nn
import torchvision.models as models

# --- 1. Feature Extraction Components ---

class ResNet50Backbone(nn.Module):
    """
    ResNet50 backbone for feature extraction.
    Uses layers up to layer3 (conv4_x stage) as described in many detection papers
    to get features before significant downsampling.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Using weights=None to avoid deprecation warning directly when pretrained=False
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights) # Use weights argument
        
        # Features up to layer3 (output stride 16, channels 1024)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

    def forward(self, x):
        return self.features(x)

class ReceptiveEnhancementModule(nn.Module):
    """
    Receptive Enhancement Module (REM) as described in the paper.
    Uses a convolutional layer with a large kernel to enlarge the receptive field.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TransformerEncoder(nn.Module):
    """
    Simplified Transformer Encoder.
    In a full implementation, positional encodings would be crucial.
    """
    def __init__(self, feature_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x is assumed to be (Batch, Channel, Height, Width)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1) # Reshape to (B, H*W, C) for Transformer
        
        # In a real implementation, you'd add positional embeddings here
        # E.g., x = x + self.pos_embed(h, w)
        
        output = self.transformer_encoder(x)
        return output # (B, H*W, C)

# --- 2. Task-aware Query Initialization (TAQI) ---

class TaskAwareQueryInitialization(nn.Module):
    """
    Conceptual Task-aware Query Initialization (TAQI).
    This is highly simplified for demonstration.
    A real TAQI (like in ESTextSpotter) would involve learnable query embeddings
    that interact with image features through attention to propose bounding boxes
    and initialize task-specific queries.
    """
    def __init__(self, feature_dim, num_queries):
        super().__init__()
        self.num_queries = num_queries
        
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        
        self.bbox_coord_head = nn.Linear(feature_dim, 4) # (cx, cy, w, h)
        self.bbox_score_head = nn.Linear(feature_dim, 1) # objectness score

        self.detection_query_project = nn.Linear(feature_dim, feature_dim)
        self.recognition_query_project = nn.Linear(feature_dim, feature_dim)

    def forward(self, encoded_features):
        initial_queries = self.query_embed.weight.unsqueeze(0).repeat(encoded_features.shape[0], 1, 1)

        coarse_bboxes_coords = self.bbox_coord_head(initial_queries).sigmoid()
        coarse_bboxes_scores = self.bbox_score_head(initial_queries).sigmoid()
        
        coarse_bboxes_and_scores = torch.cat([coarse_bboxes_coords, coarse_bboxes_scores], dim=-1)

        detection_queries = self.detection_query_project(initial_queries)
        recognition_queries = self.recognition_query_project(initial_queries)

        return encoded_features, detection_queries, recognition_queries, coarse_bboxes_and_scores

# This is the "Module 1" wrapper
class VimTSModule1(nn.Module):
    def __init__(self, resnet_pretrained=True,
                 rem_in_channels=1024, rem_out_channels=1024,
                 transformer_feature_dim=1024, transformer_num_heads=8, transformer_num_layers=3,
                 num_queries=100):
        super().__init__()
        self.resnet_backbone = ResNet50Backbone(pretrained=resnet_pretrained)
        self.receptive_enhancement_module = ReceptiveEnhancementModule(
            in_channels=rem_in_channels, out_channels=rem_out_channels
        )
        self.transformer_encoder = TransformerEncoder(
            feature_dim=transformer_feature_dim,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers
        )
        self.task_aware_query_init = TaskAwareQueryInitialization(
            feature_dim=transformer_feature_dim,
            num_queries=num_queries
        )

    def forward(self, img):
        resnet_features = self.resnet_backbone(img)
        rem_features = self.receptive_enhancement_module(resnet_features)
        
        encoded_image_features = self.transformer_encoder(rem_features)

        encoded_features_for_decoder, detection_queries, recognition_queries, coarse_bboxes_and_scores = \
            self.task_aware_query_init(encoded_image_features)

        return {
            "encoded_image_features": encoded_features_for_decoder,
            "detection_queries": detection_queries,
            "recognition_queries": recognition_queries,
            "coarse_bboxes_and_scores": coarse_bboxes_and_scores
        }
