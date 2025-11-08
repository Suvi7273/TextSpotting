# matcher.py - Fixed version

import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import torch.nn as nn

# Utilities for box format conversion
def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from (cx, cy, w, h) to (x_min, y_min, x_max, y_max).
    All values are normalized [0, 1].
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    """
    Computes IoU between two sets of boxes.
    Boxes are expected in [x0, y0, x1, y1] format.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # [N,M,2]

    wh = (rb - lt).clamp(min=0) # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1] # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-7)  # Add epsilon to avoid division by zero
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://arxiv.org/abs/1902.09630.
    The boxes should be in [x0, y0, x1, y1] format.
    Returns a [N, M] tensor.
    """
    # Force boxes to be [x0, y0, x1, y1] format
    boxes1 = box_cxcywh_to_xyxy(boxes1) if boxes1.shape[-1] == 4 and boxes1.min() >=0 and boxes1.max() <=1 else boxes1
    boxes2 = box_cxcywh_to_xyxy(boxes2) if boxes2.shape[-1] == 4 and boxes2.min() >=0 and boxes2.max() <=1 else boxes2

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1] # Area of the smallest enclosing box

    return iou - (area - union) / (area + 1e-7)  # Add epsilon to avoid division by zero


class HungarianMatcher(nn.Module):
    """This class computes an optimal bipartite matching between the detections and the targets.
    For each query, a set of predictions (bbox, class, recognition) is assigned.
    The costs are weighted sum of:
        - Negative log-probability of the predicted class.
        - L1 distance between predicted and target bounding boxes.
        - GIoU distance between predicted and target bounding boxes.
        - Recognition cost (e.g., CrossEntropy loss or another sequence distance).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_recognition: float = 1, vocab_size: int = 97, max_seq_len: int = 25, padding_idx: int = 96):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_recognition = cost_recognition
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_recognition != 0, "All costs can't be 0"
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching.
        Args:
            outputs (dict): This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits.
                "pred_bboxes": Tensor of dim [batch_size, num_queries, 4] with the normalized box coordinates.
                "pred_chars_logits": Tensor of dim [batch_size, num_queries, max_seq_len, vocab_size] for recognition.

            targets (list[dict]): Targets dicts for each image.
                The keys will be "labels", "boxes", and "recognition".
        Returns:
            A list of size batch_size, where each element is a tuple of (indices_i, indices_j)
            that contains the matched indices of the predictions and the targets.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We compute the cost matrices for each image in the batch, then combine.
        # For batch_size=1, we just take the first item.
        # Outputs:
        # out_prob: (num_queries, num_classes_total) -> (num_queries, 2)
        out_prob = outputs["pred_logits"][0].softmax(-1)  
        out_bbox = outputs["pred_bboxes"][0]  # (num_queries, 4) (cxcywh)
        out_rec_logits = outputs["pred_chars_logits"][0] # (num_queries, max_seq_len, vocab_size)

        # Targets (for the current image in the batch):
        tgt_labels = targets[0]["labels"]  # (num_target_boxes) -> (N_gt) where N_gt is variable per image
        tgt_bbox = targets[0]["boxes"]  # (num_target_boxes, 4) (cxcywh)
        tgt_rec_seq = targets[0]["recognition"] # (num_target_boxes, max_seq_len)

        if tgt_labels.numel() == 0: # Handle case with no ground truth objects in the image
            return [(torch.empty(0, dtype=torch.int64, device=out_prob.device), 
                     torch.empty(0, dtype=torch.int64, device=out_prob.device))]

        # Compute the classification cost.
        # Cost for class is 1 - P(class=0) where class 0 is 'text'.
        # So it's P(class=1) i.e. P(no_object).
        cost_class = out_prob[:, 1] # Probability of being 'no_object'
        cost_class = cost_class.unsqueeze(1).repeat(1, tgt_labels.shape[0]) # (num_queries, num_target_boxes)

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # (num_queries, num_target_boxes)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox) # Negative GIoU as cost to minimize

        # Compute Recognition Cost (Cross-Entropy loss for sequences)
        # Handle cases where recognition sequences are all padding
        num_tgt = tgt_rec_seq.shape[0]
        
        # Check if target sequences have any non-padding tokens
        non_padding_mask = (tgt_rec_seq != self.padding_idx)  # (num_target_boxes, max_seq_len)
        has_text = non_padding_mask.any(dim=1)  # (num_target_boxes,) - True if sequence has any non-padding token
        
        if has_text.any():
            # Compute recognition cost only for targets with actual text
            # out_rec_logits: (num_queries, max_seq_len, vocab_size)
            # tgt_rec_seq: (num_target_boxes, max_seq_len)
            
            # Expand for broadcasting
            out_rec_logits_expanded = out_rec_logits.unsqueeze(1).expand(-1, num_tgt, -1, -1)  # (num_queries, num_tgt, max_seq_len, vocab_size)
            tgt_rec_seq_expanded = tgt_rec_seq.unsqueeze(0).expand(num_queries, -1, -1)  # (num_queries, num_tgt, max_seq_len)
            
            # Reshape for cross entropy
            flat_logits = out_rec_logits_expanded.reshape(-1, self.vocab_size)  # (num_queries * num_tgt * max_seq_len, vocab_size)
            flat_targets = tgt_rec_seq_expanded.reshape(-1)  # (num_queries * num_tgt * max_seq_len)
            
            # Compute cross entropy loss
            ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none', ignore_index=self.padding_idx)
            
            # Reshape back to (num_queries, num_tgt, max_seq_len)
            ce_loss = ce_loss.view(num_queries, num_tgt, self.max_seq_len)
            
            # Sum over sequence length to get cost per query-target pair
            cost_recognition = ce_loss.sum(dim=-1)  # (num_queries, num_tgt)
            
            # For targets without text (all padding), set recognition cost to 0
            cost_recognition[:, ~has_text] = 0.0
            
            # Normalize by the number of non-padding tokens to avoid bias towards shorter sequences
            num_tokens_per_target = non_padding_mask.sum(dim=1).float().clamp(min=1.0)  # (num_tgt,)
            cost_recognition = cost_recognition / num_tokens_per_target.unsqueeze(0)  # (num_queries, num_tgt)
        else:
            # All targets are padding-only, set recognition cost to 0
            cost_recognition = torch.zeros(num_queries, num_tgt, device=out_prob.device)

        # Final cost matrix: C_ij = bbox_cost + class_cost + giou_cost + recognition_cost
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + \
            self.cost_giou * cost_giou + self.cost_recognition * cost_recognition
        
        # Check for invalid values before passing to Hungarian algorithm
        if torch.isnan(C).any() or torch.isinf(C).any():
            print(f"Warning: Cost matrix contains NaN or Inf values!")
            print(f"  cost_bbox contains NaN: {torch.isnan(cost_bbox).any()}")
            print(f"  cost_class contains NaN: {torch.isnan(cost_class).any()}")
            print(f"  cost_giou contains NaN: {torch.isnan(cost_giou).any()}")
            print(f"  cost_recognition contains NaN: {torch.isnan(cost_recognition).any()}")
            
            # Replace NaN and Inf with large finite values
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)
        
        C = C.cpu() # Hungarian algorithm typically runs on CPU

        # Perform Hungarian matching
        indices = linear_sum_assignment(C) # Returns (row_ind, col_ind)
        # row_ind: indices of predicted queries
        # col_ind: indices of target boxes
        return [(torch.as_tensor(indices[0], dtype=torch.int64, device=out_prob.device), 
                 torch.as_tensor(indices[1], dtype=torch.int64, device=out_prob.device))]