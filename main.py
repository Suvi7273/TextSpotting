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
from text_spotting_dataset import TotalTextDataset, collate_fn ,build_vocabulary_from_json, AdaptiveResize, AdaptiveResizeTest  
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

# --- Visualization Function (updated to use new pred_logits format) ---
def visualize_output(original_image_path, model_output, gt_info, vocab_map=None, padding_idx=None, show_gt=True, show_preds=True, score_threshold=0.5):
    image = Image.open(original_image_path).convert('RGB')
    
    img_width, img_height = image.size

    plt.figure(figsize=(12, 12)) # Make figure a bit larger
    plt.imshow(image)
    ax = plt.gca()

    # --- Draw Ground Truth ---
    if show_gt:
        for bbox_coords in gt_info['boxes']: # Use 'boxes' from new target format
            # Convert normalized cxcywh to pixel xywh for drawing
            cx_norm, cy_norm, w_norm, h_norm = bbox_coords.tolist()
            x_min = (cx_norm - w_norm / 2) * img_width
            y_min = (cy_norm - h_norm / 2) * img_height
            width = w_norm * img_width
            height = h_norm * img_height

            rect = patches.Rectangle((x_min, y_min), width, height, 
                                     linewidth=1, edgecolor='g', facecolor='none', linestyle='--')
            ax.add_patch(rect)
        
        # GT Polygons
        for poly_coords_norm_tensor in gt_info['polygons']: # List of tensors now
            poly_coords_norm_list = poly_coords_norm_tensor.tolist()
            # Convert normalized polygon points back to pixel coordinates
            poly_pixel = []
            for i in range(0, len(poly_coords_norm_list), 2):
                poly_pixel.append(poly_coords_norm_list[i] * img_width)
                poly_pixel.append(poly_coords_norm_list[i+1] * img_height)
            
            poly = [(poly_pixel[i], poly_pixel[i+1]) for i in range(0, len(poly_pixel), 2)]
            if len(poly) > 0:
                polygon_patch = patches.Polygon(poly, closed=True, linewidth=2, edgecolor='b', facecolor='none')
                ax.add_patch(polygon_patch)
        
        plt.text(5, 5, "Green dashed: GT BBox", color='g', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        plt.text(5, 20, "Blue: GT Polygon", color='b', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    # --- Draw Predictions ---
    if show_preds:
        # Get final predicted bboxes and logits
        # pred_logits are raw logits (B, N_queries, num_classes+1)
        # We want the probability of the foreground class (class 0 'text')
        pred_logits_raw = model_output['pred_logits'][0] # (N_queries, num_classes+1)
        pred_scores = F.softmax(pred_logits_raw, dim=-1)[:, 0] # Probability of 'text' class
        
        final_pred_bboxes = model_output['pred_bboxes'][0] # (N_queries, 4) (cx, cy, w, h)
        final_pred_rec_logits = model_output['pred_chars_logits'][0] # (N_queries, max_seq_len, vocab_size)

        for i in range(final_pred_bboxes.shape[0]):
            score = pred_scores[i].item()
            
            if score >= score_threshold:
                bbox_data = final_pred_bboxes[i]
                cx_norm, cy_norm, w_norm, h_norm = bbox_data.tolist()
                
                # Convert normalized (cx, cy, w, h) to pixel (x_min, y_min, width, height)
                x_min = (cx_norm - w_norm / 2) * img_width
                y_min = (cy_norm - h_norm / 2) * img_height
                width = w_norm * img_width
                height = h_norm * img_height

                rect = patches.Rectangle((x_min, y_min), width, height,
                                         linewidth=1, edgecolor='red', facecolor='none', linestyle='-')
                ax.add_patch(rect)
                
                # Decode recognition prediction if vocab_map is provided
                predicted_text = ""
                if vocab_map and final_pred_rec_logits.shape[0] > i:
                    rec_logits_per_query = final_pred_rec_logits[i] # (max_seq_len, vocab_size)
                    predicted_char_ids = rec_logits_per_query.argmax(dim=-1).tolist()
                    for char_id in predicted_char_ids:
                        if char_id == padding_idx: # Use the passed padding_idx
                            break
                        predicted_text += vocab_map.get(char_id, '?') # Map ID to char
                
                text_label = f'{score:.2f}'
                if predicted_text:
                    text_label += f' "{predicted_text}"'

                plt.text(x_min, y_min - 5, text_label, color='red', fontsize=7, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.text(5, 35, f"Red: Predicted BBox (score > {score_threshold})", color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    plt.title(f"Image: {gt_info['file_name']}")
    # plt.show() # Removed as per Colab fix


# --- Example Usage / Training Loop ---
if __name__ == "__main__":
    # --- Configuration ---
    JSON_PATH = '/content/drive/MyDrive/dataset_ts/mlt2017_sample/train.json'
    IMAGE_DIR = '/content/drive/MyDrive/dataset_ts/mlt2017_sample/img'
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build vocabulary from dataset
    id_to_char, char_to_id, VOCAB_SIZE, PADDING_IDX = build_vocabulary_from_json(JSON_PATH)
    print(f"Vocabulary size: {VOCAB_SIZE}, Padding index: {PADDING_IDX}")

    # Model parameters
    FEATURE_DIM = 1024
    NUM_QUERIES = 100
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 6
    NUM_HEADS = 8
    NUM_FOREGROUND_CLASSES = 1 # 'text' class
    
    MAX_RECOGNITION_SEQ_LEN = 25 
    NUM_POLYGON_POINTS = 16 
    
    # Training parameters
    BATCH_SIZE = 1 # Keep at 1 for now due to complexity of collate_fn for DETR targets
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100 # Increase for more "learning" (even with random init)

    transform_train = transforms.Compose([
        AdaptiveResize(min_size=640, max_size=896, max_long_side=1600),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        AdaptiveResizeTest(shorter_size=1000, max_long_side=1824),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TotalTextDataset(json_path=JSON_PATH, img_dir=IMAGE_DIR, transform=transform_train, 
                               max_recognition_seq_len=MAX_RECOGNITION_SEQ_LEN,
                               padding_value=PADDING_IDX,check_recognition_quality=True)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    
    USE_ADAPTER = True  # Set to True for adapter-based fine-tuning
    FREEZE_BACKBONE = False  # Set to True for stage 2 (after pre-training)
    USE_PQGM = False     # Start with False for initial training, True for multi-task
    TASK_ID = 0          # 0=word-level, 1=line-level, 2=video-level

    # --- Initialize Full Model ---
    model = VimTSFullModel(
        resnet_pretrained=True, # Set to True for real training to get better features
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
    
    if FREEZE_BACKBONE:
        print("Freezing backbone and encoder...")
        for name, param in model.named_parameters():
            if 'module1' in name and 'task_aware_query_init' not in name:
                param.requires_grad = False
            if 'decoder' in name and 'adapter' not in name:
                param.requires_grad = False
            if 'prediction_heads' in name and 'adapter' not in name:
                param.requires_grad = False
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params}/{total_params} ({100*trainable_params/total_params:.2f}%)")

    # --- Initialize DETR Criterion and Matcher ---
    matcher = HungarianMatcher(
        cost_class=1, # Weight for classification cost
        cost_bbox=5,  # Weight for L1 bbox cost
        cost_giou=2,  # Weight for GIoU bbox cost
        cost_recognition=2, # Weight for recognition cost
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_RECOGNITION_SEQ_LEN,
        padding_idx=PADDING_IDX
    )
    
    # Weights for the different loss terms during backpropagation
    weight_dict = {
        'loss_ce': 2.0,         # λ_c = 2.0 (paper)
        'loss_bbox': 5.0,       # λ_b = 5.0 (paper)
        'loss_giou': 2.0,       # GIoU weight (paper)
        'loss_recognition': 1.0, # α_r = 1.0 (paper)
        'loss_cardinality': 1.0,
        'loss_polygon': 1.0     # α_p = 1.0 (paper)
    }
    # Losses to compute, as strings
    losses_to_compute = ['labels', 'boxes', 'polygons', 'recognition', 'cardinality']

    # eos_coef is the weight for the "no_object" class in classification loss
    # If num_foreground_classes=1 (text), then actual pred_logits output 2 classes (text, no_object).
    # eos_coef applies to the 'no_object' class.
    # Set to a value like 0.1 for more stability.
    eos_coef = 0.1 

    criterion = SetCriterion(
        num_foreground_classes=NUM_FOREGROUND_CLASSES,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=losses_to_compute,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_RECOGNITION_SEQ_LEN,
        padding_idx=PADDING_IDX
    ).to(device)


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler (as per paper: reduce at 180k and 210k iterations)
    # For shorter training, we'll use epochs instead
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[int(NUM_EPOCHS*0.75), int(NUM_EPOCHS*0.875)],  # 75% and 87.5% of training
        gamma=0.1
    )
    
    # Gradient accumulation steps (to simulate larger batch size)
    ACCUMULATION_STEPS = 4
    
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    # Add gradient clipping
    MAX_GRAD_NORM = 0.1

    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

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

    # --- Inference and Visualization after training ---
    print("\nRunning inference and visualization on a random sample after training...")
    model.eval() # Set model to evaluation mode
    criterion.eval()
    
    # Pick a random image from the dataset for visualization
    sample_idx = random.randint(0, len(dataset) - 1)
    image_tensor_viz, gt_info_viz = dataset[sample_idx]
    
    if image_tensor_viz.dim() == 3:
        image_tensor_viz = image_tensor_viz.unsqueeze(0)
    image_tensor_viz = image_tensor_viz.to(device)

    with torch.no_grad():
        output_viz = model(image_tensor_viz)

    print("\n--- Output of Full VimTS Model (after training) ---")
    print(f"Pred BBoxes shape: {output_viz['pred_bboxes'].shape}")
    print(f"Pred Logits shape: {output_viz['pred_logits'].shape}")
    print(f"Pred Chars Logits shape: {output_viz['pred_chars_logits'].shape}")
    
    pred_logits_scores = F.softmax(output_viz['pred_logits'][0], dim=-1)[:, 0].cpu().numpy() # Probability of 'text' class
    high_score_preds = [s for s in pred_logits_scores if s >= 0.01]
    print(f"Number of refined predicted bboxes (score >= 0.01): {len(high_score_preds)}")

    original_image_full_path = os.path.join(IMAGE_DIR, gt_info_viz['file_name'])
    
    print(f"\nVisualizing final output for image: {gt_info_viz['file_name']}")
    visualize_output(original_image_full_path, output_viz, gt_info_viz, 
                     vocab_map=id_to_char, padding_idx=PADDING_IDX, # Pass vocab map for recognition text
                     show_gt=True, show_preds=True, score_threshold=0.01)

    # --- Save the output image ---
    output_filename = f"final_visualization_{gt_info_viz['file_name']}.png"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1) # Added bbox_inches and pad_inches
    print(f"Saved visualization to {output_filename}")
    
    plt.show() # MUST be the very last line in the cell
