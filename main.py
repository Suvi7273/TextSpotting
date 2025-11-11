import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Import modules from the separate files
from vimts_backbone_taqi import VimTSModule1
from vimts_decoder_heads import TaskAwareDecoder, TaskAwareDecoderLayer, PredictionHeads
from text_spotting_dataset import collate_fn , AdaptiveResize, AdaptiveResizeTest, build_vocabulary_from_text_files, TextFileDataset
from matcher import HungarianMatcher, box_cxcywh_to_xyxy 
from detr_losses import SetCriterion 


# --- Full VimTS Model ---
class VimTSFullModel(nn.Module):
    def __init__(self, resnet_pretrained=True,
                 rem_in_channels=1024, rem_out_channels=1024,
                 transformer_feature_dim=1024, transformer_num_heads=8, transformer_num_encoder_layers=3,
                 num_queries=100,
                 transformer_num_decoder_layers=6,
                 num_foreground_classes=1, # 1 for 'text'
                 vocab_size=97, # 0-95 for chars, 96 for padding
                 num_polygon_points=16,
                 max_recognition_seq_len=25,
                 use_adapter=True,
                 use_pqgm=False, 
                 task_id=0):
        super().__init__()
        self.module1 = VimTSModule1(
            resnet_pretrained=resnet_pretrained,
            rem_in_channels=rem_in_channels, rem_out_channels=rem_out_channels,
            transformer_feature_dim=transformer_feature_dim,
            transformer_num_heads=transformer_num_heads,
            transformer_num_layers=transformer_num_encoder_layers,
            num_queries=num_queries,
            use_pqgm=use_pqgm,
            task_id=task_id
        )

        decoder_layer = TaskAwareDecoderLayer(
            d_model=transformer_feature_dim,
            nhead=transformer_num_heads
        )
        self.decoder = TaskAwareDecoder(
            decoder_layer=decoder_layer,
            num_layers=transformer_num_decoder_layers,
            d_model=transformer_feature_dim,  
            max_queries=num_queries * 2 
        )

        self.prediction_heads = PredictionHeads(
            d_model=transformer_feature_dim,
            num_foreground_classes=num_foreground_classes,
            num_chars=vocab_size,
            num_polygon_points=num_polygon_points,
            max_recognition_seq_len=max_recognition_seq_len,
            use_adapter=use_adapter
        )
        self.num_queries = num_queries # Store for matching with targets (for training)

    def forward(self, img):
        module1_output = self.module1(img)
        
        encoded_image_features = module1_output["encoded_image_features"]
        detection_queries = module1_output["detection_queries"]
        recognition_queries = module1_output["recognition_queries"]

        # Ensure detection and recognition queries are of same num_queries for now,
        # so matcher can handle. The paper suggests they might be the same set.
        # If not, matcher needs to be adapted for matching text instance detection + text instance recognition.
        # For now, let's assume num_detection_queries == num_recognition_queries == self.num_queries
        
        refined_detection_queries, refined_recognition_queries, _ = self.decoder(
            detection_queries, recognition_queries, encoded_image_features
        )

        final_predictions = self.prediction_heads(
            refined_detection_queries, refined_recognition_queries
        )
        
        final_predictions["coarse_bboxes_and_scores"] = module1_output["coarse_bboxes_and_scores"]

        return final_predictions

def visualize_output(original_image_path, model_output, gt_info, vocab_map=None, padding_idx=None, 
                     show_gt=True, show_preds=True, score_threshold=0.5, max_preds=50):
    """
    Visualize predictions and ground truth on image.
    
    Args:
        original_image_path: Path to original image
        model_output: Dictionary with model predictions
        gt_info: Ground truth target dictionary
        vocab_map: Character ID to character mapping
        padding_idx: Padding index for recognition
        show_gt: Whether to show ground truth
        show_preds: Whether to show predictions
        score_threshold: Minimum confidence score to display
        max_preds: Maximum number of predictions to show
    """
    image = Image.open(original_image_path).convert('RGB')
    img_width, img_height = image.size

    plt.figure(figsize=(16, 12))
    plt.imshow(image)
    ax = plt.gca()

    # --- Draw Ground Truth ---
    if show_gt and gt_info['boxes'].numel() > 0:
        for bbox_coords in gt_info['boxes']:
            cx_norm, cy_norm, w_norm, h_norm = bbox_coords.tolist()
            x_min = (cx_norm - w_norm / 2) * img_width
            y_min = (cy_norm - h_norm / 2) * img_height
            width = w_norm * img_width
            height = h_norm * img_height

            rect = patches.Rectangle((x_min, y_min), width, height, 
                                     linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
            ax.add_patch(rect)

    # --- Draw Predictions ---
    if show_preds:
        pred_logits_raw = model_output['pred_logits'][0]  # (N_queries, num_classes)
        pred_scores = F.softmax(pred_logits_raw, dim=-1)[:, 0]  # Probability of 'text' class
        
        final_pred_bboxes = model_output['pred_bboxes'][0]  # (N_queries, 4)
        final_pred_rec_logits = model_output['pred_chars_logits'][0] if 'pred_chars_logits' in model_output else None

        # Sort predictions by score
        scores_np = pred_scores.cpu().numpy()
        sorted_indices = np.argsort(scores_np)[::-1]  # Descending order
        
        # Count how many predictions above threshold
        num_above_threshold = (scores_np >= score_threshold).sum()
        print(f"\nPredictions above threshold {score_threshold}: {num_above_threshold}")
        print(f"Top 10 scores: {sorted(scores_np, reverse=True)[:10]}")
        
        predictions_drawn = 0
        for idx in sorted_indices:
            if predictions_drawn >= max_preds:
                break
                
            score = scores_np[idx]
            
            if score >= score_threshold:
                bbox_data = final_pred_bboxes[idx]
                cx_norm, cy_norm, w_norm, h_norm = bbox_data.cpu().tolist()
                
                # Convert normalized (cx, cy, w, h) to pixel (x_min, y_min, width, height)
                x_min = (cx_norm - w_norm / 2) * img_width
                y_min = (cy_norm - h_norm / 2) * img_height
                width = w_norm * img_width
                height = h_norm * img_height

                # Draw bounding box
                rect = patches.Rectangle((x_min, y_min), width, height,
                                         linewidth=2, edgecolor='red', facecolor='none', linestyle='-')
                ax.add_patch(rect)
                
                # Decode recognition prediction
                predicted_text = ""
                if vocab_map and final_pred_rec_logits is not None:
                    rec_logits_per_query = final_pred_rec_logits[idx]  # (max_seq_len, vocab_size)
                    predicted_char_ids = rec_logits_per_query.argmax(dim=-1).cpu().tolist()
                    for char_id in predicted_char_ids:
                        if char_id == padding_idx:
                            break
                        predicted_text += vocab_map.get(char_id, '?')
                
                # Create label
                text_label = f'{score:.3f}'
                if predicted_text:
                    text_label += f' "{predicted_text}"'

                # Draw label with background
                plt.text(x_min, y_min - 5, text_label, color='white', fontsize=9, 
                        weight='bold', bbox=dict(facecolor='red', alpha=0.7, pad=2))
                
                predictions_drawn += 1
        
        print(f"Drew {predictions_drawn} predictions")

    # Add legend
    legend_y = 20
    if show_gt:
        plt.text(10, legend_y, "Green dashed: Ground Truth", color='white', fontsize=10, 
                weight='bold', bbox=dict(facecolor='green', alpha=0.7, pad=3))
        legend_y += 25
    if show_preds:
        plt.text(10, legend_y, f"Red solid: Predictions (score ≥ {score_threshold})", color='white', 
                fontsize=10, weight='bold', bbox=dict(facecolor='red', alpha=0.7, pad=3))

    plt.axis('off')
    plt.title(f"Image: {gt_info['file_name']}", fontsize=14, weight='bold')
    plt.tight_layout()
    

import math
if __name__ == "__main__":
    GT_DIR = '/content/drive/MyDrive/dataset_ts/mlt_sample/Train_GT'
    IMAGE_DIR = '/content/drive/MyDrive/dataset_ts/mlt_sample/TrainImages'

    # Build vocabulary (excluding Arabic)
    id_to_char, char_to_id, VOCAB_SIZE, PADDING_IDX = build_vocabulary_from_text_files(
        gt_dir=GT_DIR,
        required_language='Latin',
        save_vocab_path='/content/vocabulary.pkl'
    )

    print(f"Vocabulary size: {VOCAB_SIZE}, Padding index: {PADDING_IDX}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters
    FEATURE_DIM = 1024
    NUM_QUERIES = 100
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 6
    NUM_HEADS = 8
    NUM_FOREGROUND_CLASSES = 1 # 'text' class

    MAX_RECOGNITION_SEQ_LEN = 25 
    NUM_POLYGON_POINTS = 16 

    # Training parameters - IMPROVED
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    WARMUP_EPOCHS = 5  # Add warmup

    # Data augmentation for training
    transform_train = transforms.Compose([
        AdaptiveResize(min_size=640, max_size=896, max_long_side=1600),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Add augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        AdaptiveResizeTest(shorter_size=1000, max_long_side=1824),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # IMPORTANT: Filter out samples with no meaningful text during training
    class FilteredDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, min_text_length=1):
            self.base_dataset = base_dataset
            self.min_text_length = min_text_length
            
            # Filter to keep only samples with meaningful text
            self.valid_indices = []
            for idx in range(len(base_dataset)):
                _, target = base_dataset[idx]
                # Check if any recognition sequence has non-padding, non-placeholder tokens
                rec_seqs = target['recognition']
                has_meaningful_text = False
                for rec_seq in rec_seqs:
                    # Count non-padding and non-placeholder (95) tokens
                    meaningful_tokens = ((rec_seq != 96) & (rec_seq != 95)).sum()
                    if meaningful_tokens >= self.min_text_length:
                        has_meaningful_text = True
                        break
                if has_meaningful_text:
                    self.valid_indices.append(idx)
            
            print(f"\nFiltered dataset: {len(self.valid_indices)}/{len(base_dataset)} samples have meaningful text")
        
        def __len__(self):
            return len(self.valid_indices)
        
        def __getitem__(self, idx):
            return self.base_dataset[self.valid_indices[idx]]

    # Create base dataset
    base_dataset = TextFileDataset(
        gt_dir=GT_DIR,
        img_dir=IMAGE_DIR,
        transform=transform_train,
        max_recognition_seq_len=MAX_RECOGNITION_SEQ_LEN,
        padding_value=PADDING_IDX,
        char_to_id=char_to_id,
        require_language='Latin'           # Only include images with Latin text
    )

    # Apply the FilteredDataset wrapper if needed
    dataset = FilteredDataset(base_dataset, min_text_length=2)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)

    USE_ADAPTER = True
    FREEZE_BACKBONE = False  # For initial training, keep backbone trainable with lower LR
    USE_PQGM = False
    TASK_ID = 0

    # --- Initialize Full Model ---
    model = VimTSFullModel(
        resnet_pretrained=True,  # IMPORTANT: Use pretrained backbone
        rem_in_channels=1024,
        rem_out_channels=FEATURE_DIM,
        transformer_feature_dim=FEATURE_DIM,
        transformer_num_heads=NUM_HEADS,
        transformer_num_encoder_layers=NUM_ENCODER_LAYERS,
        num_queries=NUM_QUERIES,
        transformer_num_decoder_layers=NUM_DECODER_LAYERS,
        num_foreground_classes=NUM_FOREGROUND_CLASSES,
        vocab_size=VOCAB_SIZE,
        num_polygon_points=NUM_POLYGON_POINTS,
        max_recognition_seq_len=MAX_RECOGNITION_SEQ_LEN,
        use_adapter=USE_ADAPTER,
        use_pqgm=USE_PQGM, 
        task_id=TASK_ID
    ).to(device)

    # --- IMPROVED: Use different learning rates for backbone vs new layers ---
    backbone_params = []
    new_params = []

    for name, param in model.named_parameters():
        if 'resnet_backbone' in name:
            backbone_params.append(param)
        else:
            new_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  # 10x lower LR for backbone
        {'params': new_params, 'lr': LEARNING_RATE}
    ]

    # --- Initialize DETR Criterion and Matcher ---
    matcher = HungarianMatcher(
        cost_class=2,  # Increased from 1
        cost_bbox=5,
        cost_giou=2,
        cost_recognition=2,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_RECOGNITION_SEQ_LEN,
        padding_idx=PADDING_IDX
    )

    # IMPROVED: Adjust loss weights based on task importance
    weight_dict = {
        'loss_ce': 2.0,          # Classification
        'loss_bbox': 5.0,        # Bounding box L1 (part of 'boxes' loss)
        'loss_giou': 2.0,        # GIoU (part of 'boxes' loss)
        'loss_recognition': 3.0, # Recognition
        'loss_cardinality': 1.0, # Cardinality
        'loss_polygon': 1.0      # Polygon
    }

    # IMPORTANT FIX: The loss names in losses_to_compute must match the methods in SetCriterion
    # The SetCriterion has these loss methods:
    #   - 'labels' -> computes loss_ce (classification)
    #   - 'boxes' -> computes loss_bbox AND loss_giou
    #   - 'cardinality' -> computes loss_cardinality
    #   - 'polygons' -> computes loss_polygon
    #   - 'recognition' -> computes loss_recognition

    # CORRECT: Use the method names, not the individual loss component names
    losses_to_compute = ['labels', 'boxes', 'cardinality', 'polygons']

    # Add recognition only if you have good quality data
    if len(dataset) > len(base_dataset) * 0.3:  # If >30% have meaningful text
        losses_to_compute.append('recognition')
        print("\nIncluding recognition loss in training")
    else:
        print("\nSkipping recognition loss due to insufficient text data")
        # Remove recognition from weight_dict if not using it
        weight_dict.pop('loss_recognition', None)

    print(f"Losses to compute: {losses_to_compute}")

    eos_coef = 0.1

    criterion = SetCriterion(
        num_foreground_classes=NUM_FOREGROUND_CLASSES,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=losses_to_compute,  # Pass the correct loss method names
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_RECOGNITION_SEQ_LEN,
        padding_idx=PADDING_IDX
    ).to(device)

    print(f"\nCriterion initialized with losses: {losses_to_compute}")
    print(f"Loss weights: {weight_dict}")

    optimizer = optim.AdamW(param_groups, lr=LEARNING_RATE, weight_decay=1e-4)

    # IMPROVED: Learning rate scheduler with warmup
    def get_lr_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing after warmup
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = get_lr_scheduler_with_warmup(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)

    # Gradient accumulation and clipping
    ACCUMULATION_STEPS = 4
    MAX_GRAD_NORM = 0.1

    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}")

        # images shape: (B, 3, H, W)
        img = images[0]  # take first image in batch

        # Convert from tensor → NumPy → HWC for matplotlib
        img = img.permute(1, 2, 0).cpu().numpy()

        # Undo normalization if you applied transforms.Normalize(...)
        # img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        # img = img.clip(0, 1)

        plt.imshow(img)
        plt.title(f"Batch {batch_idx}")
        plt.axis("off")
        plt.show()

        print("Targets:")
        print(targets)
        print()

        if batch_idx >= 2:
            break  # stop after a few batches

    # --- Training Loop ---
    print("\nStarting DETR-style training loop...")
    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        criterion.train()
        total_epoch_loss = 0
        
        optimizer.zero_grad()

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            # targets is a list of dictionaries (even for batch_size=1)
            # Move individual tensors within target dicts to device
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            with autocast():
                predictions = model(images)
                loss_dict = criterion(predictions, targets)
                
                # Check if loss_dict contains any NaN or Inf
                loss_values_valid = all(not (torch.isnan(v).any() or torch.isinf(v).any()) 
                                       for v in loss_dict.values() if isinstance(v, torch.Tensor))
                
                if not loss_values_valid:
                    print(f"Warning: Invalid loss values detected at epoch {epoch+1}, batch {batch_idx+1}")
                    for k, v in loss_dict.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: NaN={torch.isnan(v).any()}, Inf={torch.isinf(v).any()}")
                    continue  # Skip this batch
                
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                loss = loss / ACCUMULATION_STEPS  # Normalize loss
            
            # Check if loss is valid before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid total loss at epoch {epoch+1}, batch {batch_idx+1}. Skipping.")
                continue
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                # Unscale gradients and clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_epoch_loss += loss.item() * ACCUMULATION_STEPS

            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(dataloader)}, Total Loss: {loss.item() * ACCUMULATION_STEPS:.4f}", end=' | ')
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        print(f"{k}: {v.item():.4f}", end=' ')
                print()
        
        # Step any remaining gradients
        if (len(dataloader) % ACCUMULATION_STEPS) != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        scheduler.step()

        print(f"Epoch {epoch+1} finished, Average Total Loss: {total_epoch_loss / len(dataloader):.4f}")

    print("\nDETR-style training finished.")
    
    # Save model checkpoint
    checkpoint_path = '/content/vimts_model_checkpoint.pth'
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': {'id_to_char': id_to_char, 'char_to_id': char_to_id, 
                  'vocab_size': VOCAB_SIZE, 'padding_idx': PADDING_IDX},
        'config': {
            'feature_dim': FEATURE_DIM,
            'num_queries': NUM_QUERIES,
            'num_encoder_layers': NUM_ENCODER_LAYERS,
            'num_decoder_layers': NUM_DECODER_LAYERS,
            'use_adapter': USE_ADAPTER,
            'use_pqgm': USE_PQGM,
            'task_id': TASK_ID
        }
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # --- After training, visualize results ---
    print("\nRunning inference on a sample...")
    model.eval()

    sample_idx = random.randint(0, len(dataset) - 1)
    image_tensor_viz, gt_info_viz = dataset[sample_idx]

    if image_tensor_viz.dim() == 3:
        image_tensor_viz = image_tensor_viz.unsqueeze(0)
    image_tensor_viz = image_tensor_viz.to(device)

    with torch.no_grad():
        output_viz = model(image_tensor_viz)

    print("\n--- Model Output Statistics ---")
    print(f"Pred BBoxes shape: {output_viz['pred_bboxes'].shape}")
    print(f"Pred Logits shape: {output_viz['pred_logits'].shape}")

    pred_logits_scores = F.softmax(output_viz['pred_logits'][0], dim=-1)[:, 0].cpu().numpy()
    print(f"\nScore statistics:")
    print(f"  Max score: {pred_logits_scores.max():.4f}")
    print(f"  Mean score: {pred_logits_scores.mean():.4f}")
    print(f"  Min score: {pred_logits_scores.min():.4f}")
    print(f"  Scores > 0.5: {(pred_logits_scores > 0.5).sum()}")
    print(f"  Scores > 0.1: {(pred_logits_scores > 0.1).sum()}")
    print(f"  Scores > 0.01: {(pred_logits_scores > 0.01).sum()}")

    original_image_full_path = os.path.join(IMAGE_DIR, gt_info_viz['file_name'])

    # Use lower threshold to see all detections
    visualize_output(
        original_image_full_path, 
        output_viz, 
        gt_info_viz, 
        vocab_map=id_to_char, 
        padding_idx=PADDING_IDX,
        show_gt=True, 
        show_preds=True, 
        score_threshold=0.01,  # Very low threshold to see all predictions
        max_preds=100
    )

    output_filename = f"visualization_{gt_info_viz['file_name']}"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"\nSaved visualization to {output_filename}")
    plt.show()