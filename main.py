# main.py

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

# --- TotalText Dataset Loader (same as before) ---
class TotalTextDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.images = self.data['images']
        self.annotations = self.data['annotations']
        
        # Map image_id to annotations
        self.img_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.img_id_to_annotations:
                self.img_id_to_annotations[image_id] = []
            self.img_id_to_annotations[image_id].append(ann)
            
        self.image_infos = {img['id']: img for img in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_path = os.path.join(self.img_dir, img_filename)

        image = Image.open(img_path).convert('RGB')
        
        annotations = self.img_id_to_annotations.get(img_id, [])
        
        gt_bboxes = [] # [x_min, y_min, width, height]
        gt_polygons = [] # list of [x1, y1, x2, y2, ...]
        gt_recs = [] # list of recognition strings (or tokenized IDs)
        
        for ann in annotations:
            gt_bboxes.append(ann['bbox']) 
            if ann['segmentation']:
                gt_polygons.extend(ann['segmentation'])
            
            # Convert numeric 'rec' to a list of integers, representing character IDs
            # You'll need a vocabulary to map these IDs back to characters for actual recognition
            # For now, we'll just store the raw IDs. The paper uses 96 for padding/unknown.
            gt_recs.append(ann['rec']) # e.g., [84, 72, 69, 96, ..., 96]
            
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32) if gt_bboxes else torch.empty((0, 4), dtype=torch.float32)
        
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            
        original_width, original_height = image.size

        return image_tensor, {
            'image_id': img_id,
            'file_name': img_filename,
            'original_size': (original_width, original_height),
            'gt_bboxes': gt_bboxes,
            'gt_polygons': gt_polygons,
            'gt_recs': gt_recs # Added ground truth recognition labels
        }

# --- Full VimTS Model ---
class VimTSFullModel(nn.Module):
    def __init__(self, resnet_pretrained=True,
                 rem_in_channels=1024, rem_out_channels=1024,
                 transformer_feature_dim=1024, transformer_num_heads=8, transformer_num_encoder_layers=3,
                 num_queries=100,
                 transformer_num_decoder_layers=6, # Standard for DETR
                 num_chars=97, # Assuming 96 for padding + 1 for start token, or actual vocab size
                 num_polygon_points=16):
        super().__init__()
        self.module1 = VimTSModule1(
            resnet_pretrained=resnet_pretrained,
            rem_in_channels=rem_in_channels, rem_out_channels=rem_out_channels,
            transformer_feature_dim=transformer_feature_dim,
            transformer_num_heads=transformer_num_heads,
            transformer_num_layers=transformer_num_encoder_layers,
            num_queries=num_queries
        )

        decoder_layer = TaskAwareDecoderLayer(
            d_model=transformer_feature_dim,
            nhead=transformer_num_heads
        )
        self.decoder = TaskAwareDecoder(
            decoder_layer=decoder_layer,
            num_layers=transformer_num_decoder_layers
        )

        self.prediction_heads = PredictionHeads(
            d_model=transformer_feature_dim,
            num_chars=num_chars,
            num_polygon_points=num_polygon_points
        )

    def forward(self, img):
        module1_output = self.module1(img)
        
        encoded_image_features = module1_output["encoded_image_features"]
        detection_queries = module1_output["detection_queries"]
        recognition_queries = module1_output["recognition_queries"]
        # coarse_bboxes_and_scores = module1_output["coarse_bboxes_and_scores"] # Can use this for aux loss

        refined_detection_queries, refined_recognition_queries, _ = self.decoder(
            detection_queries, recognition_queries, encoded_image_features
        )

        final_predictions = self.prediction_heads(
            refined_detection_queries, refined_recognition_queries
        )
        
        # Merge initial coarse bboxes with final predictions for a complete output dict
        final_predictions["coarse_bboxes_and_scores"] = module1_output["coarse_bboxes_and_scores"]

        return final_predictions

# --- Loss Functions (Simplified for demonstration) ---
# In a real system, you'd use Hungarian matching and more sophisticated losses (Focal Loss, GIoU, Cross-Entropy)
# as mentioned in Section III.G of the paper. This is a placeholder for basic functionality.

def dummy_bbox_loss(pred_bboxes, gt_bboxes):
    # Very simple L1 loss, assuming 1:1 match for simplicity.
    # In reality, this requires Hungarian matching to assign predictions to ground truths.
    if gt_bboxes.numel() == 0 or pred_bboxes.numel() == 0:
        return torch.tensor(0.0, device=pred_bboxes.device)
    
    # For a simple demo, let's assume one prediction per GT, or just average over all preds
    # This is NOT how DETR loss is calculated, it needs matching!
    loss = F.l1_loss(pred_bboxes[:, :gt_bboxes.shape[0]], gt_bboxes) # Simplistic: assume N_pred >= N_gt and take first N_gt
    return loss

def dummy_class_loss(pred_logits, has_text_gt):
    # Binary cross-entropy for text/no-text classification
    if has_text_gt.numel() == 0:
        return torch.tensor(0.0, device=pred_logits.device)
    
    # Assuming pred_logits is objectness score [0,1], and has_text_gt is 1 for text, 0 for no text
    # This is also simplistic; proper classification loss is more complex with matching.
    target_scores = torch.ones_like(pred_logits[:, :has_text_gt.shape[0], :]) # Assume all GTs are text
    loss = F.binary_cross_entropy_with_logits(pred_logits[:, :has_text_gt.shape[0], :], target_scores)
    return loss

def dummy_recognition_loss(pred_chars_logits, gt_recs, num_chars, max_seq_len=25):
    # This is a highly simplified recognition loss placeholder.
    # It would typically be a sequence-to-sequence loss (e.g., CTC or Transformer decoder loss).
    if not gt_recs or pred_chars_logits.numel() == 0:
        return torch.tensor(0.0, device=pred_chars_logits.device)

    # Convert gt_recs from list of lists of char_ids to a padded tensor
    # Find max length among GT recs in the batch
    batch_gt_rec_tensors = []
    for rec_list in gt_recs:
        # Assuming rec_list is already character IDs, pad to max_seq_len
        rec_tensor = torch.tensor(rec_list, dtype=torch.long, device=pred_chars_logits.device)
        if len(rec_tensor) > max_seq_len:
            rec_tensor = rec_tensor[:max_seq_len]
        padded_rec_tensor = F.pad(rec_tensor, (0, max_seq_len - len(rec_tensor)), value=96) # 96 for padding
        batch_gt_rec_tensors.append(padded_rec_tensor)
    
    # Concatenate into (Batch_size * Num_rec_queries, Max_seq_len) if Num_rec_queries matches GT_recs
    # Simplistic: Assume one recognition query per GT text
    target_recs = torch.stack(batch_gt_rec_tensors, dim=0).view(-1) # (Batch * Max_seq_len)

    # Flatten pred_chars_logits if it's (B, N_rec, Num_chars) to (B*N_rec, Num_chars)
    # This is also assuming one prediction maps to one GT.
    flat_pred_chars_logits = pred_chars_logits.view(-1, num_chars) # (B*N_rec, Num_chars)

    # Use CrossEntropyLoss (suitable if pred_chars_logits are raw logits)
    # This assumes recognition queries are already matched to GT recs (which is hard without Hungarian)
    loss = F.cross_entropy(flat_pred_chars_logits[:len(target_recs)], target_recs)
    return loss


# --- Visualization Function (same as before) ---
def visualize_output(original_image_path, model_output, gt_info, show_gt=True, show_preds=True, score_threshold=0.5):
    image = Image.open(original_image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    img_width, img_height = image.size

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    # --- Draw Ground Truth ---
    if show_gt:
        for bbox_coords in gt_info['gt_bboxes']:
            x_min, y_min, width, height = bbox_coords.tolist()
            rect = patches.Rectangle((x_min, y_min), width, height, 
                                     linewidth=1, edgecolor='g', facecolor='none', linestyle='--')
            ax.add_patch(rect)
        
        for poly_coords_list in gt_info['gt_polygons']:
            poly = [(poly_coords_list[i], poly_coords_list[i+1]) for i in range(0, len(poly_coords_list), 2)]
            if len(poly) > 0:
                polygon_patch = patches.Polygon(poly, closed=True, linewidth=2, edgecolor='b', facecolor='none')
                ax.add_patch(polygon_patch)
        
        plt.text(5, 5, "Green dashed: GT BBox", color='g', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        plt.text(5, 20, "Blue: GT Polygon", color='b', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    # --- Draw Predictions (Coarse & Refined) ---
    if show_preds:
        # We will visualize FINAL predictions from the decoder, not just coarse ones.
        # Although the coarse ones are also in model_output if you want to inspect them.
        
        # Get final predicted bboxes and logits (assuming detection queries map to these)
        final_pred_bboxes = model_output['pred_bboxes'][0] # (N_det, 4)
        final_pred_logits = model_output['pred_logits'][0] # (N_det, 1)

        for i in range(final_pred_bboxes.shape[0]):
            bbox_data = final_pred_bboxes[i]
            logit_score = final_pred_logits[i].item() # Get scalar score

            cx_norm, cy_norm, w_norm, h_norm = bbox_data.tolist()
            
            # The pred_logits are sigmoid-activated, so they are already [0,1] scores.
            if logit_score >= score_threshold:
                # Convert normalized (cx, cy, w, h) to pixel (x_min, y_min, width, height)
                x_min = (cx_norm - w_norm / 2) * img_width
                y_min = (cy_norm - h_norm / 2) * img_height
                width = w_norm * img_width
                height = h_norm * img_height

                rect = patches.Rectangle((x_min, y_min), width, height,
                                         linewidth=1, edgecolor='red', facecolor='none', linestyle='-')
                ax.add_patch(rect)
                plt.text(x_min, y_min - 5, f'{logit_score:.2f}', color='red', fontsize=7, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.text(5, 35, f"Red: Refined Predicted BBox (score > {score_threshold})", color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    plt.title(f"Image: {gt_info['file_name']}")
    # plt.show() # Removed as per Colab fix

# --- Example Usage / Training Loop ---
if __name__ == "__main__":
    # --- Configuration ---
    JSON_PATH = '/content/drive/MyDrive/sample/train.json'
    IMAGE_DIR = '/content/drive/MyDrive/sample/img'
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters
    FEATURE_DIM = 1024
    NUM_QUERIES = 100
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 6
    NUM_HEADS = 8
    VOCAB_SIZE = 97 # Example: 96 for padding/unknown, 1 for start/end, others for chars
    NUM_POLYGON_POINTS = 16 # For Bezier curves
    
    # Training parameters (minimal for demonstration)
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5 # Just a few for a demo

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TotalTextDataset(json_path=JSON_PATH, img_dir=IMAGE_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # --- Initialize Full Model ---
    model = VimTSFullModel(
        resnet_pretrained=False, # Set to True for real training to get better features
        rem_in_channels=1024,
        rem_out_channels=FEATURE_DIM,
        transformer_feature_dim=FEATURE_DIM,
        transformer_num_heads=NUM_HEADS,
        transformer_num_encoder_layers=NUM_ENCODER_LAYERS,
        num_queries=NUM_QUERIES,
        transformer_num_decoder_layers=NUM_DECODER_LAYERS,
        num_chars=VOCAB_SIZE,
        num_polygon_points=NUM_POLYGON_POINTS
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Training Loop (Simplified) ---
    print("\nStarting (simplified) training loop...")
    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            # For simplicity, we're not properly handling batching of gt_bboxes, gt_polygons, gt_recs
            # in the dummy loss functions, as they have variable sizes per image.
            # A full DETR implementation uses specific data collation and Hungarian matching.
            # Here, we'll process one image at a time from the batch for loss calculation for this demo.
            
            # This is a VERY simplistic way to get GT for a single image in batch
            # For actual batch training, a custom collate_fn and loss computation with matching is needed.
            # We'll just take the first item's GT for now.
            
            if BATCH_SIZE > 1:
                print("Warning: Dummy loss functions are not correctly handling batch size > 1 for GT. Using first item's GT.")
            
            gt_bboxes_batch = targets['gt_bboxes'][0].to(device) # Shape N_gt, 4
            # gt_polygons_batch = targets['gt_polygons'][0] # Polygon loss not implemented in dummy
            gt_recs_batch = targets['gt_recs'][0] # List of lists of char IDs

            optimizer.zero_grad()
            predictions = model(images)

            # --- Calculate Losses (DUMMY LOSSES - DO NOT USE FOR REAL TRAINING) ---
            # These dummy losses assume 1:1 correspondence which is incorrect for DETR.
            # Proper DETR loss involves Hungarian matching between predictions and GT.
            loss_bbox = dummy_bbox_loss(predictions['pred_bboxes'], gt_bboxes_batch)
            loss_cls = dummy_class_loss(predictions['pred_logits'], gt_bboxes_batch) # Use existence of bbox for class
            loss_rec = dummy_recognition_loss(predictions['pred_chars_logits'], gt_recs_batch, VOCAB_SIZE)
            
            # Combine losses
            loss = loss_bbox + loss_cls + loss_rec
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} finished, Average Loss: {total_loss / len(dataloader):.4f}")

    print("\n(Simplified) training finished.")

    # --- Inference and Visualization after (dummy) training ---
    print("\nRunning inference and visualization on a random sample after training...")
    model.eval() # Set model to evaluation mode
    
    # Pick a random image from the dataset for visualization
    sample_idx = random.randint(0, len(dataset) - 1)
    image_tensor_viz, gt_info_viz = dataset[sample_idx]
    
    # Ensure image_tensor is a batch for the model
    if image_tensor_viz.dim() == 3:
        image_tensor_viz = image_tensor_viz.unsqueeze(0)
    image_tensor_viz = image_tensor_viz.to(device)

    with torch.no_grad():
        output_viz = model(image_tensor_viz)

    print("\n--- Output of Full VimTS Model (after dummy training) ---")
    print(f"Pred BBoxes shape: {output_viz['pred_bboxes'].shape}")
    print(f"Pred Logits shape: {output_viz['pred_logits'].shape}")
    print(f"Pred Chars Logits shape: {output_viz['pred_chars_logits'].shape}")
    
    pred_logits_scores = output_viz['pred_logits'][0].sigmoid().cpu().numpy() # For visualization threshold
    high_score_preds = [s for s in pred_logits_scores if s >= 0.01] # Check for low threshold
    print(f"Number of refined predicted bboxes (score >= 0.01): {len(high_score_preds)}")

    original_image_full_path = os.path.join(IMAGE_DIR, gt_info_viz['file_name'])
    
    print(f"\nVisualizing final output for image: {gt_info_viz['file_name']}")
    visualize_output(original_image_full_path, output_viz, gt_info_viz, 
                     show_gt=True, show_preds=True, score_threshold=0.01) # Low threshold to see everything

    plt.show() # MUST be the very last line in the cell

    # print("\n--- IMPORTANT NOTES ON DUMMY TRAINING ---")
    # print("1. The loss functions (`dummy_bbox_loss`, `dummy_class_loss`, `dummy_recognition_loss`) ")
    # print("   are HIGHLY SIMPLIFIED and DO NOT implement proper DETR-style Hungarian matching.")
    # print("   They are for demonstration purposes ONLY to make the code run.")
    # print("   Real DETR training requires robust matching between predictions and ground truths.")
    # print("2. `resnet_pretrained` is set to `False`. For actual performance, set it to `True`.")
    # print("3. Positional embeddings for Transformer Encoder/Decoder are omitted for simplicity but are CRUCIAL.")
    # print("4. `gt_recs` would need proper tokenization and vocabulary mapping for meaningful recognition loss.")
    # print("5. This training loop is illustrative. For robust results, use the official VimTS/ESTextSpotter implementation.")
