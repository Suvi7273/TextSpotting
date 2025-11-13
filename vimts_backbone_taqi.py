import torch
import torch.nn as nn
import torchvision.models as models
import math

# --- 1. Feature Extraction Components ---

class ResNet50Backbone(nn.Module):
    """
    ResNet50 backbone - SIMPLIFIED AND FIXED
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Use layer4 for better features (output stride 32, channels 2048)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,  # Changed to layer4 for better features
        )
        
        # Add 1x1 conv to reduce channels from 2048 to 1024
        self.channel_reducer = nn.Conv2d(2048, 1024, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.channel_reducer(x)
        return x


class ReceptiveEnhancementModule(nn.Module):
    """
    SIMPLIFIED REM - just a few convolutions with different receptive fields
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Multiple branches with different kernel sizes
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.output_conv(out)
        return out


class PositionalEncoding2D(nn.Module):
    """
    DETR-style 2D Positional Encoding - FIXED AND SIMPLIFIED
    """
    def __init__(self, d_model, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, h, w):
        """
        x: (B, H*W, C)
        """
        B, HW, C = x.shape
        device = x.device
        
        # Create coordinate grids
        y_embed = torch.arange(h, dtype=torch.float32, device=device)
        x_embed = torch.arange(w, dtype=torch.float32, device=device)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (h + eps) * self.scale
            x_embed = x_embed / (w + eps) * self.scale
        
        # Create meshgrid
        y_embed = y_embed.unsqueeze(1).repeat(1, w)  # (H, W)
        x_embed = x_embed.unsqueeze(0).repeat(h, 1)  # (H, W)
        
        # Flatten
        y_embed = y_embed.reshape(-1)  # (H*W,)
        x_embed = x_embed.reshape(-1)  # (H*W,)
        
        # Calculate dimensions for encoding
        num_pos_feats = C // 2
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)
        
        # Generate position encodings
        pos_x = x_embed.unsqueeze(1) / dim_t  # (H*W, num_pos_feats)
        pos_y = y_embed.unsqueeze(1) / dim_t  # (H*W, num_pos_feats)
        
        # Apply sin/cos
        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1)
        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1)
        
        # Concatenate
        pos = torch.cat([pos_y, pos_x], dim=1)  # (H*W, C)
        
        # Handle dimension mismatch
        if pos.shape[1] < C:
            pos = torch.cat([pos, torch.zeros(HW, C - pos.shape[1], device=device)], dim=1)
        elif pos.shape[1] > C:
            pos = pos[:, :C]
        
        return x + pos.unsqueeze(0)


class TransformerEncoder(nn.Module):
    """
    SIMPLIFIED Transformer Encoder with proper initialization
    """
    def __init__(self, feature_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        
        # Use standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads, 
            dim_feedforward=feature_dim * 4,
            dropout=dropout, 
            activation='gelu',  # Changed to GELU
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding2D(feature_dim)
        
        # Layer norm before and after
        self.pre_norm = nn.LayerNorm(feature_dim)
        self.post_norm = nn.LayerNorm(feature_dim)
        
        # Initialize weights properly
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        
        # Pre-norm
        x = self.pre_norm(x)
        
        # Add positional encoding
        x = self.pos_encoding(x, h, w)
        
        # Apply transformer
        output = self.transformer_encoder(x)
        
        # Post-norm
        output = self.post_norm(output)
        
        return output


class TaskAwareQueryInitialization(nn.Module):
    """
    SIMPLIFIED Query Initialization - Focus on getting it working first
    """
    def __init__(self, feature_dim, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        
        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        
        # Simple cross-attention
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        
        self.norm = nn.LayerNorm(feature_dim)
        
        # Simple bbox prediction
        self.bbox_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 5)  # 4 coords + 1 score
        )
        
        # Initialize properly
        nn.init.xavier_uniform_(self.query_embed.weight)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for module in [self.bbox_head]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, encoded_features):
        B = encoded_features.shape[0]
        
        # Get queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # Cross-attention with image features
        attended_queries, _ = self.cross_attention(
            queries, encoded_features, encoded_features
        )
        
        # Residual + norm
        queries = self.norm(queries + attended_queries)
        
        # Predict coarse boxes
        coarse_pred = self.bbox_head(queries).sigmoid()
        coarse_bboxes_coords = coarse_pred[..., :4]
        coarse_bboxes_scores = coarse_pred[..., 4:5]
        coarse_bboxes_and_scores = torch.cat([coarse_bboxes_coords, coarse_bboxes_scores], dim=-1)
        
        # Use same queries for both detection and recognition
        detection_queries = queries
        recognition_queries = queries
        
        return encoded_features, detection_queries, recognition_queries, coarse_bboxes_and_scores


class VimTSModule1(nn.Module):
    """
    SIMPLIFIED Module 1 - Focus on stability
    """
    def __init__(self, resnet_pretrained=True,
                 rem_in_channels=1024, rem_out_channels=1024,
                 transformer_feature_dim=1024, transformer_num_heads=8, 
                 transformer_num_layers=3, num_queries=100, 
                 use_pqgm=False, task_id=0):
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
        
        self.use_pqgm = use_pqgm
        self.task_id = task_id

    def forward(self, img):
        # Forward pass
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