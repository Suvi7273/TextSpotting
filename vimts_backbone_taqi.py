import torch
import torch.nn as nn
import torchvision.models as models
import math
from torch.nn.init import xavier_uniform_, constant_, normal_

# ==============================================================================
# 1. MULTI-SCALE FEATURE EXTRACTION (ResNet FPN-style)
# ==============================================================================

class ResNet50BackboneMultiScale(nn.Module):
    """
    ResNet50 with MULTI-SCALE feature extraction
    Returns features at 3 scales: C3, C4, C5 (stride 8, 16, 32)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Layer1: stride 4, channels 256
        # Layer2: stride 8, channels 512 (C3)
        # Layer3: stride 16, channels 1024 (C4)
        # Layer4: stride 32, channels 2048 (C5)
        
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels (C3)
        self.layer3 = resnet.layer3  # 1024 channels (C4)
        self.layer4 = resnet.layer4  # 2048 channels (C5)
        
        # Channel reduction for each scale to unified dimension (256)
        self.channel_reducer_c3 = nn.Conv2d(512, 256, kernel_size=1)
        self.channel_reducer_c4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.channel_reducer_c5 = nn.Conv2d(2048, 256, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)  # Not used but computed
        c3 = self.layer2(c1)  # 1/8 resolution
        c4 = self.layer3(c3)  # 1/16 resolution
        c5 = self.layer4(c4)  # 1/32 resolution
        
        # Reduce channels to unified dimension
        c3 = self.channel_reducer_c3(c3)
        c4 = self.channel_reducer_c4(c4)
        c5 = self.channel_reducer_c5(c5)
        
        return [c3, c4, c5]


# ==============================================================================
# 2. FIXED POSITIONAL ENCODING (DETR-style, exact implementation)
# ==============================================================================

class PositionalEncoding2D(nn.Module):
    """
    DETR-style 2D Positional Encoding - FIXED VERSION
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, C, H, W) with positional encoding added
        """
        B, C, H, W = x.shape
        
        # Create mask (all ones for now, can add padding mask later)
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=x.device)
        
        # Compute cumulative sums (for handling masks, here just indices)
        y_embed = (~mask).cumsum(1, dtype=torch.float32)  # (B, H, W)
        x_embed = (~mask).cumsum(2, dtype=torch.float32)  # (B, H, W)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Shape: (B, H, W, num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Apply sin to even indices, cos to odd indices
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        # Concatenate y and x encodings: (B, H, W, 2*num_pos_feats)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, 2*num_pos_feats, H, W)
        
        # Match channel dimension with input
        if pos.shape[1] < C:
            # Pad with zeros
            pos = torch.cat([pos, torch.zeros(B, C - pos.shape[1], H, W, device=x.device)], dim=1)
        elif pos.shape[1] > C:
            # Truncate
            pos = pos[:, :C, :, :]
        
        return x + pos


# ==============================================================================
# 3. RECEPTIVE ENHANCEMENT MODULE (Paper-style with large kernels)
# ==============================================================================

class ReceptiveEnhancementModule(nn.Module):
    """
    REM with LARGE KERNEL convolutions (7x7) and dilation
    Based on VimTS paper description
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Multiple branches with LARGE kernels and different dilations
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Large kernel with dilation (paper mentions this)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, padding=9, dilation=3),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
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


# ==============================================================================
# 4. MULTI-SCALE DEFORMABLE ATTENTION (Simplified version)
# ==============================================================================

class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention (SIMPLIFIED)
    
    For full implementation, use: 
    from models.ops.modules import MSDeformAttn (from Deformable DETR)
    
    This is a PLACEHOLDER - you need the actual implementation
    """
    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        
        # Sampling offsets
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        
        # Attention weights
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        
        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        constant_(self.sampling_offsets.bias.data, 0.)
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index):
        """
        PLACEHOLDER FORWARD
        Real implementation is complex and requires custom CUDA kernels
        
        For now, use standard multi-head attention as fallback
        """
        # Simplified fallback: standard attention
        N, Len_q, _ = query.shape
        value = self.value_proj(input_flatten)
        
        # Simple attention (NOT deformable, just a placeholder)
        attn_output = torch.bmm(
            torch.softmax(torch.bmm(query, value.transpose(1, 2)) / math.sqrt(self.d_model), dim=-1),
            value
        )
        
        return self.output_proj(attn_output)


# ==============================================================================
# 5. TRANSFORMER ENCODER WITH DEFORMABLE ATTENTION
# ==============================================================================

class DeformableTransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Multi-Scale Deformable Attention
    """
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        
        # Multi-scale deformable attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        # Self attention
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        
        return src


class DeformableTransformerEncoder(nn.Module):
    """
    Transformer Encoder with Multi-Scale Deformable Attention
    """
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        output = src
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index)
        return output


# ==============================================================================
# 6. QUERY INITIALIZATION (Paper-style: coarse boxes + RoI features)
# ==============================================================================

class QueryInitialization(nn.Module):
    """
    Query Initialization following VimTS paper:
    1. Predict coarse bounding boxes
    2. Select top-N based on scores
    3. Extract RoI features for recognition queries
    4. Generate detection queries from box coordinates
    """
    def __init__(self, feature_dim, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        
        # Coarse bbox prediction head (applied to highest resolution features)
        self.bbox_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 5, kernel_size=1)  # 4 bbox coords + 1 score
        )
        
        # Position encoding for detection queries
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim)
        )
        
        # RoI Align for extracting recognition features
        from torchvision.ops import RoIAlign
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2)
        
        # Recognition query encoder (from RoI features)
        self.rec_query_encoder = nn.Sequential(
            nn.Linear(feature_dim * 7 * 7, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0.)

    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: List of [c3, c4, c5] features
        Returns:
            detection_queries, recognition_queries, coarse_bboxes_and_scores
        """
        # Use highest resolution features (C3) for coarse prediction
        c3_features = multi_scale_features[0]  # (B, C, H, W)
        B, C, H, W = c3_features.shape
        
        # Predict coarse bounding boxes
        coarse_pred = self.bbox_predictor(c3_features)  # (B, 5, H, W)
        coarse_pred = coarse_pred.permute(0, 2, 3, 1).reshape(B, H * W, 5)  # (B, H*W, 5)
        
        # Apply sigmoid to get normalized coordinates and scores
        coarse_pred = coarse_pred.sigmoid()
        coarse_bboxes = coarse_pred[..., :4]  # (B, H*W, 4) [cx, cy, w, h]
        coarse_scores = coarse_pred[..., 4:5]  # (B, H*W, 1)
        
        # Select top-N based on scores
        top_scores, top_indices = torch.topk(coarse_scores.squeeze(-1), self.num_queries, dim=1)
        
        # Gather top boxes
        top_boxes = torch.gather(
            coarse_bboxes, 
            1, 
            top_indices.unsqueeze(-1).expand(-1, -1, 4)
        )  # (B, N, 4)
        
        # Generate detection queries from box coordinates
        detection_queries = self.pos_encoder(top_boxes)  # (B, N, feature_dim)
        
        # Extract RoI features for recognition queries
        # Convert normalized boxes to pixel coordinates for RoI Align
        boxes_for_roi = top_boxes.clone()
        boxes_for_roi[:, :, [0, 2]] *= W  # Scale x coords
        boxes_for_roi[:, :, [1, 3]] *= H  # Scale y coords
        
        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
        boxes_xyxy = torch.zeros_like(boxes_for_roi)
        boxes_xyxy[:, :, 0] = boxes_for_roi[:, :, 0] - boxes_for_roi[:, :, 2] / 2
        boxes_xyxy[:, :, 1] = boxes_for_roi[:, :, 1] - boxes_for_roi[:, :, 3] / 2
        boxes_xyxy[:, :, 2] = boxes_for_roi[:, :, 0] + boxes_for_roi[:, :, 2] / 2
        boxes_xyxy[:, :, 3] = boxes_for_roi[:, :, 1] + boxes_for_roi[:, :, 3] / 2
        
        # Add batch indices for RoI Align
        batch_indices = torch.arange(B, device=boxes_xyxy.device).view(-1, 1, 1).expand(-1, self.num_queries, 1)
        rois = torch.cat([batch_indices.float(), boxes_xyxy], dim=-1).reshape(-1, 5)
        
        # Extract RoI features
        roi_features = self.roi_align(c3_features, rois)  # (B*N, C, 7, 7)
        roi_features = roi_features.reshape(B, self.num_queries, C * 7 * 7)
        
        # Generate recognition queries from RoI features
        recognition_queries = self.rec_query_encoder(roi_features)  # (B, N, feature_dim)
        
        # Combine boxes and scores for output
        coarse_bboxes_and_scores = torch.cat([top_boxes, top_scores.unsqueeze(-1)], dim=-1)
        
        return detection_queries, recognition_queries, coarse_bboxes_and_scores


class VimTSModule1(nn.Module):
    """
    FIXED VimTS Module 1 with:
    - Multi-scale feature extraction
    - Proper positional encoding
    - Deformable transformer encoder
    - Correct query initialization
    """
    def __init__(self, resnet_pretrained=True,
                 feature_dim=256,  # Changed from 1024 to 256 (standard for DETR)
                 transformer_num_heads=8, 
                 transformer_num_layers=3, 
                 num_queries=100):
        super().__init__()
        
        # Multi-scale backbone
        self.backbone = ResNet50BackboneMultiScale(pretrained=resnet_pretrained)
        
        # REM for each scale (optional, can apply to C5 only)
        self.rem = ReceptiveEnhancementModule(
            in_channels=feature_dim, 
            out_channels=feature_dim
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(num_pos_feats=feature_dim // 2)
        
        # Transformer encoder with deformable attention
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model=feature_dim,
            n_levels=3,  # 3 scales: C3, C4, C5
            n_heads=transformer_num_heads,
            n_points=4
        )
        self.transformer_encoder = DeformableTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_num_layers
        )
        
        # Query initialization
        self.query_init = QueryInitialization(
            feature_dim=feature_dim,
            num_queries=num_queries
        )

    def forward(self, img):
        # Extract multi-scale features
        multi_scale_features = self.backbone(img)  # [C3, C4, C5]
        
        # Apply REM to highest resolution (C5 can be skipped or applied)
        c3, c4, c5 = multi_scale_features
        c5 = self.rem(c5)
        multi_scale_features = [c3, c4, c5]
        
        # Add positional encoding to each scale
        multi_scale_features_with_pos = []
        for feat in multi_scale_features:
            feat_with_pos = self.pos_encoding(feat)
            multi_scale_features_with_pos.append(feat_with_pos)
        
        # Flatten multi-scale features for transformer encoder
        src_flatten = []
        spatial_shapes = []
        for feat in multi_scale_features:
            B, C, H, W = feat.shape
            spatial_shapes.append((H, W))
            src_flatten.append(feat.flatten(2).transpose(1, 2))  # (B, H*W, C)
        
        src_flatten = torch.cat(src_flatten, dim=1)  # (B, sum(H*W), C)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=img.device)
        level_start_index = torch.cat([
            torch.tensor([0], device=img.device),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ])
        
        # Create reference points (normalized center of each spatial location)
        reference_points = []
        for H, W in spatial_shapes.tolist():
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, device=img.device),
                torch.linspace(0.5, W - 0.5, W, device=img.device)
            )
            ref = torch.stack([ref_x, ref_y], dim=-1).reshape(-1, 2)
            ref = ref / torch.tensor([W, H], device=img.device)
            reference_points.append(ref)
        reference_points = torch.cat(reference_points, dim=0).unsqueeze(0).expand(B, -1, -1)
        
        # Flatten positional encodings
        pos_flatten = []
        for feat_with_pos in multi_scale_features_with_pos:
            pos_flatten.append(feat_with_pos.flatten(2).transpose(1, 2))
        pos_flatten = torch.cat(pos_flatten, dim=1)
        
        # Apply transformer encoder
        memory = self.transformer_encoder(
            src_flatten, 
            pos_flatten, 
            reference_points, 
            spatial_shapes, 
            level_start_index
        )
        
        # Initialize queries
        detection_queries, recognition_queries, coarse_bboxes_and_scores = \
            self.query_init(multi_scale_features)

        return {
            "encoded_image_features": memory,
            "detection_queries": detection_queries,
            "recognition_queries": recognition_queries,
            "coarse_bboxes_and_scores": coarse_bboxes_and_scores,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "reference_points": reference_points
        }