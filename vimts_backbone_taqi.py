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
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
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
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        
        # Create 2D positional encoding
        pe = torch.zeros(d_model, max_h, max_w)
        
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * 
                            -(math.log(10000.0) / d_model_half))
        
        # Height encoding
        pos_h = torch.arange(0., max_h).unsqueeze(1)  # (max_h, 1)
        sin_h = torch.sin(pos_h * div_term)  # (max_h, d_model_half//2)
        cos_h = torch.cos(pos_h * div_term)  # (max_h, d_model_half//2)
        
        # Assign height encodings
        for i in range(sin_h.shape[1]):
            pe[2*i, :, :] = sin_h[:, i].unsqueeze(1).repeat(1, max_w)
            pe[2*i + 1, :, :] = cos_h[:, i].unsqueeze(1).repeat(1, max_w)
        
        # Width encoding
        pos_w = torch.arange(0., max_w).unsqueeze(1)  # (max_w, 1)
        sin_w = torch.sin(pos_w * div_term)  # (max_w, d_model_half//2)
        cos_w = torch.cos(pos_w * div_term)  # (max_w, d_model_half//2)
        
        # Assign width encodings
        for i in range(sin_w.shape[1]):
            pe[d_model_half + 2*i, :, :] = sin_w[:, i].unsqueeze(0).repeat(max_h, 1)
            pe[d_model_half + 2*i + 1, :, :] = cos_w[:, i].unsqueeze(0).repeat(max_h, 1)
        
        pe = pe.permute(1, 2, 0)  # (max_h, max_w, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x, h, w):
        """
        x: (B, H*W, C)
        Returns: (B, H*W, C) with positional encoding added
        """
        pos_encoding = self.pe[:h, :w, :].reshape(-1, x.shape[-1])  # (H*W, C)
        return x + pos_encoding.unsqueeze(0)
    

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
        self.pos_encoding = PositionalEncoding2D(feature_dim, max_h=100, max_w=100)

    def forward(self, x):
        # x is assumed to be (Batch, Channel, Height, Width)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1) # Reshape to (B, H*W, C) for Transformer
        
        # In a real implementation, you'd add positional embeddings here
        # E.g., x = x + self.pos_embed(h, w)
        x = self.pos_encoding(x, h, w)  # Add positional encoding
        
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
