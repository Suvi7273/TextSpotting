import torch
import torch.nn as nn
import torchvision.models as models
import math

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
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for spatial features.
    Uses sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model, temperature=10000):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
    
    def forward(self, x, h, w):
        """
        x: (B, H*W, C)
        Returns: (B, H*W, C) with positional encoding added
        """
        B, HW, C = x.shape
        assert HW == h * w, f"Shape mismatch: HW={HW}, h*w={h*w}"
        
        # Create position indices
        y_pos = torch.arange(h, dtype=torch.float32, device=x.device)
        x_pos = torch.arange(w, dtype=torch.float32, device=x.device)
        
        # Create meshgrid
        y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing='ij')
        
        # Flatten
        y_pos = y_pos.flatten()  # (H*W,)
        x_pos = x_pos.flatten()  # (H*W,)
        
        # Generate encodings
        dim_t = torch.arange(C // 2, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (C // 2))
        
        # Y encodings
        pos_y = y_pos[:, None] / dim_t
        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1)
        
        # X encodings
        pos_x = x_pos[:, None] / dim_t
        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1)
        
        # Concatenate and pad if necessary
        pos = torch.cat([pos_y, pos_x], dim=1)  # (H*W, C)
        
        # Handle odd dimensions
        if pos.shape[1] < C:
            pos = torch.cat([pos, torch.zeros(HW, C - pos.shape[1], device=x.device)], dim=1)
        elif pos.shape[1] > C:
            pos = pos[:, :C]
        
        # Add to features
        return x + pos.unsqueeze(0)


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
        self.pos_encoding = PositionalEncoding2D(feature_dim)  # Remove max_h, max_w parameters

    def forward(self, x):
        # x is assumed to be (Batch, Channel, Height, Width)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1) # Reshape to (B, H*W, C) for Transformer
        
        # Add positional encoding
        x = self.pos_encoding(x, h, w)  # Add positional encoding
        
        output = self.transformer_encoder(x)
        return output # (B, H*W, C)

# --- 2. Task-aware Query Initialization (TAQI) ---
class TaskAwareQueryInitialization(nn.Module):
    def __init__(self, feature_dim, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        
        # Learnable query embeddings - initialized with proper variance
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        nn.init.normal_(self.query_embed.weight, mean=0, std=1.0)
        
        # Cross-attention to make queries image-aware
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads=8, batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Separate heads for bbox prediction
        self.bbox_coord_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 4)
        )
        
        self.bbox_score_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Query projection heads
        self.detection_query_project = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.recognition_query_project = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Initialize weights properly
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with proper variance"""
        for module in [self.bbox_coord_head, self.bbox_score_head, 
                      self.detection_query_project, self.recognition_query_project]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, encoded_features):
        """
        Args:
            encoded_features: (B, H*W, feature_dim) - image features
        Returns:
            encoded_features, detection_queries, recognition_queries, coarse_predictions
        """
        B = encoded_features.shape[0]
        
        # Get static query embeddings
        initial_queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, num_queries, D)
        
        # Make queries image-aware through cross-attention
        attended_queries, attn_weights = self.cross_attention(
            initial_queries,  # queries
            encoded_features, # keys
            encoded_features  # values
        )
        
        # Residual connection and normalization
        queries = self.norm1(initial_queries + attended_queries)
        
        # Predict coarse bounding boxes
        coarse_bboxes_coords = self.bbox_coord_head(queries).sigmoid()  # (B, N, 4)
        coarse_bboxes_scores = self.bbox_score_head(queries).sigmoid()  # (B, N, 1)
        coarse_bboxes_and_scores = torch.cat([coarse_bboxes_coords, coarse_bboxes_scores], dim=-1)
        
        # Project to detection and recognition queries
        detection_queries = self.detection_query_project(queries)
        recognition_queries = self.recognition_query_project(queries)
        
        return encoded_features, detection_queries, recognition_queries, coarse_bboxes_and_scores

# This is the "Module 1" wrapper
class VimTSModule1(nn.Module):
    def __init__(self, resnet_pretrained=True,
                 rem_in_channels=1024, rem_out_channels=1024,
                 transformer_feature_dim=1024, transformer_num_heads=8, transformer_num_layers=3,
                 num_queries=100, use_pqgm=False, 
                 task_id=0):       # (0=word, 1=line, 2=video)
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
        
        if self.use_pqgm:
            # Import here to avoid circular dependency
            from prompt_queries_module import PromptQueriesGenerationModule, PromptAdapter
            
            self.pqgm = PromptQueriesGenerationModule(
                d_model=transformer_feature_dim,
                num_tasks=3,
                num_prompt_queries=10
            )
            self.prompt_adapter = PromptAdapter(d_model=transformer_feature_dim)

    def forward(self, img):
        resnet_features = self.resnet_backbone(img)
        rem_features = self.receptive_enhancement_module(resnet_features)
        
        encoded_image_features = self.transformer_encoder(rem_features)
        
        if self.use_pqgm:
            # Generate prompt queries for current task
            prompt_queries = self.pqgm(task_id=self.task_id)
            
            # Integrate prompt queries into encoded features
            encoded_image_features = self.prompt_adapter(
                encoded_image_features, prompt_queries
            )

        encoded_features_for_decoder, detection_queries, recognition_queries, coarse_bboxes_and_scores = \
            self.task_aware_query_init(encoded_image_features)

        return {
            "encoded_image_features": encoded_features_for_decoder,
            "detection_queries": detection_queries,
            "recognition_queries": recognition_queries,
            "coarse_bboxes_and_scores": coarse_bboxes_and_scores
        }
