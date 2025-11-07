# vimts_decoder_heads.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncodingLearned(nn.Module):
    """
    Learnable positional embeddings for queries.
    """
    def __init__(self, num_pos, d_model):
        super().__init__()
        self.pos_embed = nn.Embedding(num_pos, d_model)
    
    def forward(self, num_queries):
        """Returns positional embeddings for queries."""
        positions = torch.arange(num_queries, device=self.pos_embed.weight.device)
        return self.pos_embed(positions).unsqueeze(0)  # (1, num_queries, d_model)
    

class TaskAwareDecoderLayer(nn.Module):
    """
    A single layer of the Task-aware Transformer Decoder.
    Performs self-attention on queries and cross-attention with image features.
    This structure is typical for DETR-like decoders.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, queries, encoded_features, query_pos=None, feature_pos=None):
        """
        Args:
            queries (Tensor): (B, N_queries, D) - combined detection & recognition queries
            encoded_features (Tensor): (B, H*W, D) - image features from Transformer Encoder
            query_pos (Tensor): (1, N_queries, D) - positional embeddings for queries (optional)
            feature_pos (Tensor): (1, H*W, D) - positional embeddings for image features (optional)
        """
        # For simplicity in this conceptual code, positional embeddings are omitted,
        # but they are CRUCIAL for Transformer performance.
        
        # Self-attention among queries
        # (queries_input + query_pos) is typically used for query, key, value
        q_sa = k_sa = v_sa = queries # Simplified
        queries = self.norm1(queries + self.dropout1(self.self_attn(q_sa, k_sa, v_sa)[0]))

        # Cross-attention (queries attend to image features)
        # (queries_input + query_pos) as query, (encoded_features + feature_pos) as key/value
        q_ca = queries # Simplified
        k_ca = v_ca = encoded_features # Simplified
        queries = self.norm2(queries + self.dropout2(self.multihead_attn(q_ca, k_ca, v_ca)[0]))

        # FFN
        queries = self.norm3(queries + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(queries))))))
        
        return queries

class TaskAwareDecoder(nn.Module):
    """
    Stack of TaskAwareDecoderLayer instances.
    It takes initial detection and recognition queries, concatenates them,
    and refines them using encoded image features.
    """
    def __init__(self, decoder_layer, num_layers, d_model=1024, max_queries=200):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.query_pos_encoding = PositionalEncodingLearned(max_queries, d_model)

    def forward(self, detection_queries, recognition_queries, encoded_image_features):
        """
        Args:
            detection_queries (Tensor): (B, N_det, D)
            recognition_queries (Tensor): (B, N_rec, D)
            encoded_image_features (Tensor): (B, H*W, D)
        """
        # Combine detection and recognition queries for unified processing in the decoder
        combined_queries = torch.cat([detection_queries, recognition_queries], dim=1) # (B, N_det+N_rec, D)

        # Add learnable positional embeddings to queries
        query_pos = self.query_pos_encoding(combined_queries.shape[1])
        combined_queries = combined_queries + query_pos

        intermediate_outputs = []
        for layer in self.layers:
            combined_queries = layer(combined_queries, encoded_image_features)
            intermediate_outputs.append(combined_queries)

        # After all layers, split back into refined detection and recognition queries
        num_det_queries = detection_queries.shape[1]
        refined_detection_queries = combined_queries[:, :num_det_queries, :]
        refined_recognition_queries = combined_queries[:, num_det_queries:, :]

        return refined_detection_queries, refined_recognition_queries, intermediate_outputs


class TaskAwareAdapter(nn.Module):
    """
    Task-aware Adapter for efficient multi-task learning.
    Uses cascaded adapters: one for detection, one for recognition.
    """
    def __init__(self, d_model, bottleneck_dim=None):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = d_model // 4
        
        # Adapter 1: Detection information aggregation
        self.adapter_1_down = nn.Linear(d_model, bottleneck_dim)
        self.adapter_1_up = nn.Linear(bottleneck_dim, d_model)
        self.adapter_1_attn = nn.MultiheadAttention(bottleneck_dim, num_heads=4, batch_first=True)
        
        # Adapter 2: Recognition information aggregation
        self.adapter_2_down = nn.Linear(d_model, bottleneck_dim)
        self.adapter_2_up = nn.Linear(bottleneck_dim, d_model)
        self.adapter_2_attn = nn.MultiheadAttention(bottleneck_dim, num_heads=4, batch_first=True)
        
        self.activation = nn.ReLU()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
    
    def forward(self, queries, task_type='detection'):
        """
        queries: (B, N_queries, d_model)
        task_type: 'detection' or 'recognition'
        """
        residual = queries
        
        # Adapter 1: Detection features
        x = self.adapter_1_down(queries)
        x = self.activation(x)
        # Self-attention to aggregate detection information
        x_attn, _ = self.adapter_1_attn(x, x, x)
        x = x + x_attn  # Residual connection within adapter
        x = self.adapter_1_up(x)
        queries = self.layer_norm_1(residual + x)
        
        # Adapter 2: Recognition features
        residual = queries
        x = self.adapter_2_down(queries)
        x = self.activation(x)
        # Self-attention to aggregate recognition information
        x_attn, _ = self.adapter_2_attn(x, x, x)
        x = x + x_attn
        x = self.adapter_2_up(x)
        queries = self.layer_norm_2(residual + x)
        
        return queries


class PredictionHeads(nn.Module):
    """
    Takes refined queries and outputs final bounding boxes, polygons, and recognition logits.
    """
    def __init__(self, d_model, num_foreground_classes, num_chars, 
                 num_polygon_points=16, max_recognition_seq_len=25,
                 use_adapter=True):
        super().__init__()
        self.max_recognition_seq_len = max_recognition_seq_len
        self.num_foreground_classes = num_foreground_classes # Expected to be 1 (for text)
        self.use_adapter = use_adapter
        
        if self.use_adapter:
            self.detection_adapter = TaskAwareAdapter(d_model)
            self.recognition_adapter = TaskAwareAdapter(d_model)

        # Bounding box head (cx, cy, w, h)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4) # Output 4: cx, cy, w, h (normalized)
        )
        
        # Polygon head (e.g., 16 points for 4 Bezier control points: x1,y1,x2,y2...,x16,y16)
        self.polygon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_polygon_points * 2) # Output 2*N for N polygon points (normalized x, y)
        )
        
        # Classification head: Outputs raw logits for `num_foreground_classes + 1` (background) classes.
        # For text spotting, this usually means 1 foreground class (text) + 1 background class. So, 2 logits.
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_foreground_classes + 1) # Output (1 foreground class + 1 background class)
        )

        # Recognition head: each recognition query predicts a sequence of characters
        self.recognition_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, max_recognition_seq_len * num_chars) # Output logits for (Max_Seq_Len * Num_Chars)
        )

    def forward(self, refined_detection_queries, refined_recognition_queries):
        if self.use_adapter:
            refined_detection_queries = self.detection_adapter(refined_detection_queries, task_type='detection')
            refined_recognition_queries = self.recognition_adapter(refined_recognition_queries, task_type='recognition')
        
        pred_bboxes = self.bbox_head(refined_detection_queries).sigmoid() # Normalize to [0,1]
        pred_polygons = self.polygon_head(refined_detection_queries).sigmoid() # Normalize to [0,1]
        pred_logits = self.class_head(refined_detection_queries) # Raw logits for classification, not sigmoid here.

        pred_chars_logits_flat = self.recognition_head(refined_recognition_queries)
        pred_chars_logits = pred_chars_logits_flat.view(
            pred_chars_logits_flat.shape[0], 
            pred_chars_logits_flat.shape[1], 
            self.max_recognition_seq_len, 
            -1 # This will be num_chars
        )

        return {
            "pred_bboxes": pred_bboxes, # (B, N_det, 4) (cx, cy, w, h)
            "pred_polygons": pred_polygons, # (B, N_det, 2*num_polygon_points)
            "pred_logits": pred_logits, # (B, N_det, num_foreground_classes + 1) (raw logits)
            "pred_chars_logits": pred_chars_logits # (B, N_rec, Max_Seq_Len, Num_chars)
        }

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
