# vimts_decoder_heads.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskAwareDecoderLayer(nn.Module):
    """
    A single layer of the Task-aware Transformer Decoder.
    Performs self-attention on queries and cross-attention with image features.
    This structure is typical for DETR-like decoders.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        # Self-attention for queries (intra-group, inter-group if queries are combined)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Cross-attention (queries to image features)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network
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

    def forward_post_norm(self, query, key, value, query_pos=None, key_pos=None):
        # Apply self-attention
        q = k = query + query_pos if query_pos is not None else query
        query2 = self.self_attn(q, k, value, attn_mask=None, key_padding_mask=None)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # Apply cross-attention
        query2 = self.multihead_attn(query + query_pos if query_pos is not None else query,
                                      key + key_pos if key_pos is not None else key,
                                      value, attn_mask=None, key_padding_mask=None)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # Apply FFN
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)
        return query

    def forward(self, queries, encoded_features, query_pos=None, feature_pos=None):
        """
        Args:
            queries (Tensor): (B, N_queries, D) - combined detection & recognition queries
            encoded_features (Tensor): (B, H*W, D) - image features from Transformer Encoder
            query_pos (Tensor): (1, N_queries, D) - positional embeddings for queries
            feature_pos (Tensor): (1, H*W, D) - positional embeddings for image features
        """
        # For simplicity in this conceptual code, we'll assume query_pos and feature_pos
        # are handled implicitly or are part of the 'queries' and 'encoded_features' for now.
        # In a real DETR, these would be separate learned embeddings.
        
        # Self-attention among queries
        queries = self.norm1(queries + self.dropout1(self.self_attn(
            queries, queries, queries
        )[0]))

        # Cross-attention (queries attend to image features)
        queries = self.norm2(queries + self.dropout2(self.multihead_attn(
            queries, encoded_features, encoded_features
        )[0]))

        # FFN
        queries = self.norm3(queries + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(queries))))))
        
        return queries

class TaskAwareDecoder(nn.Module):
    """
    Stack of TaskAwareDecoderLayer instances.
    It takes initial detection and recognition queries, concatenates them,
    and refines them using encoded image features.
    """
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        # In a real DETR, learnable positional embeddings for the image features
        # (flattened to sequence) and queries would be added here.
        # For this conceptual implementation, we'll omit explicit positional embeddings for simplicity,
        # but they are CRUCIAL for Transformer performance.

    def forward(self, detection_queries, recognition_queries, encoded_image_features):
        """
        Args:
            detection_queries (Tensor): (B, N_det, D)
            recognition_queries (Tensor): (B, N_rec, D)
            encoded_image_features (Tensor): (B, H*W, D)
        """
        # Combine detection and recognition queries for unified processing in the decoder
        # This handles "inter-group self-attention" implicitly within the self-attention of the decoder layers.
        # The paper suggests PQGM might handle this more explicitly, but this is a common DETR approach.
        combined_queries = torch.cat([detection_queries, recognition_queries], dim=1) # (B, N_det+N_rec, D)

        intermediate_outputs = []
        for layer in self.layers:
            # Each decoder layer refines the combined queries
            combined_queries = layer(combined_queries, encoded_image_features)
            intermediate_outputs.append(combined_queries) # Store intermediate outputs if needed for auxiliary losses

        # After all layers, split back into refined detection and recognition queries
        num_det_queries = detection_queries.shape[1]
        refined_detection_queries = combined_queries[:, :num_det_queries, :]
        refined_recognition_queries = combined_queries[:, num_det_queries:, :]

        return refined_detection_queries, refined_recognition_queries, intermediate_outputs


class PredictionHeads(nn.Module):
    """
    Takes refined queries and outputs final bounding boxes, polygons, and recognition logits.
    """
    def __init__(self, d_model, num_chars, num_polygon_points=16): # TotalText uses Bezier, often 16 points for 4 Bezier curves
        super().__init__()
        # Bounding box head (cx, cy, w, h)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4) # Output 4: cx, cy, w, h
        )
        
        # Polygon head (e.g., 16 points for 4 Bezier control points: x1,y1,x2,y2...,x16,y16)
        self.polygon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_polygon_points * 2) # Output 2*N for N polygon points (x, y)
        )
        
        # Classification head (objectness score / text/no-text)
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1) # Output 1: objectness score (or 2 for text/no-text classification)
        )

        # Recognition head (logits for each character in a vocabulary)
        # Assuming fixed max_seq_length for recognition (e.g., 25 from paper, Section IV.A)
        self.recognition_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            # This would typically output logits for each character in a sequence
            # For simplicity, we'll output logits per query, assuming one recognition query -> one text instance
            # and each query's feature can be linearly projected to a sequence of character logits.
            # A more sophisticated head might involve another small Transformer Decoder for sequence generation.
            nn.Linear(d_model, num_chars) # Output logits per character
        )

    def forward(self, refined_detection_queries, refined_recognition_queries):
        """
        Args:
            refined_detection_queries (Tensor): (B, N_det, D)
            refined_recognition_queries (Tensor): (B, N_rec, D)
        """
        # For simplicity, we'll assume each refined detection query corresponds to a bbox/polygon
        # and each refined recognition query corresponds to a text string.
        # In actual DETR, usually object queries are directly linked to detection/recognition.
        # Here, let's use refined_detection_queries for localization and refined_recognition_queries for recognition.

        pred_bboxes = self.bbox_head(refined_detection_queries).sigmoid() # Normalize to [0,1]
        pred_polygons = self.polygon_head(refined_detection_queries).sigmoid() # Normalize to [0,1]
        pred_logits = self.class_head(refined_detection_queries).sigmoid() # Objectness score [0,1]

        # For recognition, the paper states max length of recognition queries is 25 (Section IV.A)
        # This implies the recognition head should output a sequence of character logits.
        # For now, let's make it simple: each recognition query predicts the characters.
        # A more realistic recognition head might take `refined_recognition_queries` and produce (B, N_rec, Max_Len, Num_Chars)
        # For this example, let's keep it (B, N_rec, Num_Chars) as if each query is one token representing character logits.
        pred_chars_logits = self.recognition_head(refined_recognition_queries) # (B, N_rec, Num_chars)

        return {
            "pred_bboxes": pred_bboxes, # (B, N_det, 4) (cx, cy, w, h)
            "pred_polygons": pred_polygons, # (B, N_det, 2*num_polygon_points)
            "pred_logits": pred_logits, # (B, N_det, 1) (objectness score)
            "pred_chars_logits": pred_chars_logits # (B, N_rec, Num_chars)
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
