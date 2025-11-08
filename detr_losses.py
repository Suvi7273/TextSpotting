# detr_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from matcher import generalized_box_iou, box_cxcywh_to_xyxy, box_iou # Import necessary box ops from matcher

# Focal Loss implementation (from DETR's original codebase, adapted)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (N, num_classes_total) raw logits.
        targets: (N) class indices (0 for foreground, 1 for background).
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        
        # Alpha balancing term
        at = self.alpha
        # If alpha is a scalar, apply to positive targets. For background, it's (1-alpha).
        # In original DETR for binary (foreground/background), alpha is often applied to foreground.
        # For simplicity, if alpha is scalar, we assume it's for foreground.
        if isinstance(self.alpha, (float, int)):
            at_tensor = torch.ones_like(targets, dtype=inputs.dtype)
            at_tensor[targets == 0] = self.alpha # For foreground class (text, index 0)
            at_tensor[targets == 1] = 1 - self.alpha # For background class (no_object, index 1)
        else: # If alpha is a tensor for per-class weighting
            at_tensor = self.alpha.gather(0, targets.long())

        focal_loss = at_tensor * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
    1) compute hungarian matching between targets and outputs
    2) compute the losses bounded by this matching
    """
    def __init__(self, num_foreground_classes, matcher, weight_dict, eos_coef, losses, vocab_size, max_seq_len, padding_idx):
        """ Create the criterion.
        Parameters:
            num_foreground_classes: number of actual foreground object categories (e.g., 1 for 'text')
            matcher: module able to compute a matching between targets and outputs
            weight_dict: dict containing weights for the different loss components
            eos_coef: relative classification weight of the no-object class
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            vocab_size: size of the character vocabulary for recognition loss
            max_seq_len: max sequence length for recognition loss
            padding_idx: index used for padding in recognition sequences
        """
        super().__init__()
        self.num_foreground_classes = num_foreground_classes # e.g., 1 (for 'text')
        self.num_total_classes = num_foreground_classes + 1 # Total classes: foreground + background
        self.background_class_idx = num_foreground_classes # Index for 'no_object' (e.g., 1 if 'text' is 0)
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef # Weight for the 'no_object' class in classification loss
        self.losses = losses
        
        # Set up FocalLoss weights
        # alpha for foreground, (1-alpha) for background.
        # DETR's alpha for FocalLoss (usually 0.25) is for foreground class.
        # For background class, the weight is (1-alpha).
        # Here we directly provide alpha as a float. FocalLoss will handle `at`.
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        
    def _get_src_permutation_idx(self, indices):
        # Permute predictions following indices. For batch_size=1, indices is [(src_idx, tgt_idx)]
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    # Target permutation idx also useful if you need to reorder targets for some losses
    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (Focal Loss).
        targets dicts must contain the key "labels".
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # This is on CUDA

        idx = self._get_src_permutation_idx(indices)
        
        target_labels = torch.full(src_logits.shape[:2], self.background_class_idx,
                                   dtype=torch.int64, device=src_logits.device) # <--- ENSURE device argument is present
        
        # Assign foreground label (class 0 'text') to matched predictions
        target_labels[idx] = 0 # As per our TotalTextDataset, 'text' is class 0

        # Focal Loss expects raw logits and target class indices
        loss_fct = FocalLoss(alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction='mean')
        loss_labels = loss_fct(src_logits.squeeze(0), target_labels.squeeze(0)) # Remove batch dim (1) for FocalLoss if batch=1

        losses = {'loss_ce': loss_labels}

        if log:
            # Calculate classification error based on argmax (text vs. no_object)
            class_pred_indices = src_logits.argmax(-1) # (B, N_queries)
            # Correct matches are where predicted label matches target_labels
            accuracy = (class_pred_indices == target_labels).float().mean()
            losses['class_error'] = (1 - accuracy) * 100 # Error percentage

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, log=True):
        """ Compute the total number of predicted objects.
        This is a DETR-specific loss to penalize the model if it predicts too many objects.
        It directly regresses the number of ground truth objects.
        """
        pred_logits = outputs['pred_logits'] # (B, N_queries, num_total_classes)
        
        # We assume pred_logits last dimension has `num_foreground_classes` (text)
        # and then the background class. So class 0 is text.
        # We need to compute the probability of being foreground.
        prob_foreground = F.softmax(pred_logits, dim=-1)[:, :, 0] # Prob of being 'text' class
        
        # Number of foreground predictions (thresholded at 0.5 for example)
        card_pred = (prob_foreground > 0.5).sum(1).float() # Sums foreground predictions per image
        
        target_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=pred_logits.device).float()
        
        # L1 loss between predicted and target cardinality
        card_loss = F.l1_loss(card_pred, target_lengths, reduction='mean')
        
        losses = {'loss_cardinality': card_loss}
        return losses

    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses for the bounding boxes, using L1 and GIoU loss.
        targets dicts must contain the key "boxes" (normalized cxcywh).
        """
        assert 'pred_bboxes' in outputs
        idx = self._get_src_permutation_idx(indices) # (batch_idx, src_idx)

        src_boxes = outputs['pred_bboxes'][idx] # This is on CUDA
        target_boxes = targets[0]['boxes'][indices[0][1]] # This is on CPU

        # Move target_boxes to CUDA
        target_boxes = target_boxes.to(src_boxes.device) # <--- ADD THIS LINE HERE

        if src_boxes.numel() == 0:
            return {'loss_bbox': torch.tensor(0.0, device=outputs['pred_bboxes'].device),
                    'loss_giou': torch.tensor(0.0, device=outputs['pred_bboxes'].device)}

        # L1 loss for bounding boxes
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / src_boxes.shape[0]} # Average over matched boxes

        # GIoU Loss for bounding boxes
        # generalized_box_iou expects xyxy format, so convert
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / src_boxes.shape[0]
        
        return losses

    def loss_polygons(self, outputs, targets, indices):
        """Compute the losses for text polygons (e.g., L1 loss on normalized bezier points).
        targets dicts must contain the key "polygons".
        """
        assert 'pred_polygons' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_polygons = outputs['pred_polygons'][idx] # This is on CUDA
        target_polygons_list = [targets[0]['polygons'][i] for i in indices[0][1]] # These are on CPU

        if src_polygons.numel() == 0 or len(target_polygons_list) == 0:
            return {'loss_polygon': torch.tensor(0.0, device=outputs['pred_polygons'].device)}
        
        padded_target_polygons = []
        for poly_tensor in target_polygons_list:
            # Move poly_tensor to the same device as src_polygons BEFORE padding/stacking
            poly_tensor = poly_tensor.to(src_polygons.device) # <--- ADD THIS LINE HERE
            if poly_tensor.shape[0] > src_polygons.shape[1]:
                poly_tensor = poly_tensor[:src_polygons.shape[1]]
            padded_poly = F.pad(poly_tensor, (0, src_polygons.shape[1] - poly_tensor.shape[0]), value=0.0)
            padded_target_polygons.append(padded_poly)
        
        target_polygons_tensor = torch.stack(padded_target_polygons) # This tensor will now be on CUDA
        
        loss_polygon = F.l1_loss(src_polygons, target_polygons_tensor, reduction='none')
        losses = {'loss_polygon': loss_polygon.sum() / src_polygons.shape[0]}
        return losses

    def loss_recognition(self, outputs, targets, indices):
        """Compute the recognition loss (Cross-Entropy for character sequences).
        targets dicts must contain the key "recognition".
        """
        assert 'pred_chars_logits' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_rec_logits = outputs['pred_chars_logits'][idx] # Matched predicted sequence logits (N_matched, Max_Seq_Len, Vocab_Size)
        target_rec_seq = targets[0]['recognition'][indices[0][1]] # Matched ground truth character sequences (N_matched, Max_Seq_Len)

        if src_rec_logits.numel() == 0 or target_rec_seq.numel() == 0:
            return {'loss_recognition': torch.tensor(0.0, device=outputs['pred_chars_logits'].device)}
        
        # Move target_rec_seq to same device as src_rec_logits if needed
        target_rec_seq = target_rec_seq.to(src_rec_logits.device)
        
        # Flatten for CrossEntropyLoss:
        # Inputs: (N, C) where N is total items, C is vocab_size
        # Targets: (N) where N is total items
        flat_src_rec_logits = src_rec_logits.reshape(-1, self.vocab_size)
        flat_target_rec_seq = target_rec_seq.reshape(-1)
        
        # CrossEntropyLoss, ignoring padding index
        loss_rec = F.cross_entropy(flat_src_rec_logits, flat_target_rec_seq, 
                                   reduction='none', ignore_index=self.padding_idx)
        
        # Average loss over non-padding tokens
        num_non_padding = (flat_target_rec_seq != self.padding_idx).sum().item()
        if num_non_padding > 0:
            loss_rec = loss_rec.sum() / num_non_padding
        else:
            # If all tokens are padding, return zero loss
            loss_rec = torch.tensor(0.0, device=outputs['pred_chars_logits'].device)

        losses = {'loss_recognition': loss_rec}
        return losses

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'polygons': self.loss_polygons,
            'recognition': self.loss_recognition,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors.
             targets: list of dicts, such that targets[i] is the labels for image i.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # (Auxiliary losses are omitted for simplicity in this conceptual implementation)

        return losses
