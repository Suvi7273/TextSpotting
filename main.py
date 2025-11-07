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
from text_spotting_dataset import TotalTextDataset, collate_fn # Use the new collate_fn
from matcher import HungarianMatcher, box_cxcywh_to_xyxy # Also bring box utility for visualization
from detr_losses import SetCriterion # Import SetCriterion


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
                 max_recognition_seq_len=25):
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
            num_foreground_classes=num_foreground_classes,
            num_chars=vocab_size,
            num_polygon_points=num_polygon_points,
            max_recognition_seq_len=max_recognition_seq_len
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
    JSON_PATH = '/content/sample/train.json'
    IMAGE_DIR = '/content/sample/img'
    
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
    
    # Vocab size and padding for recognition
    # Based on your JSON 'rec' field having values like 84, 72, 69, 96:
    # It seems 96 is padding. Other numbers are character IDs.
    # Let's create a dummy mapping for common ASCII characters and padding.
    # We need to ensure the highest character ID + 1 is VOCAB_SIZE.
    # Max value in your sample is 96, so VOCAB_SIZE = 97 is correct.
    
    id_to_char = {i: chr(i + 32) for i in range(95)} # Example: map 0 to ' ', 1 to '!', etc. up to 94 to '~'
    id_to_char[95] = '<unk>' # Unknown character
    id_to_char[96] = '<pad>' # Padding character
    char_to_id = {v: k for k, v in id_to_char.items()}
    
    VOCAB_SIZE = len(id_to_char) # 97
    PADDING_IDX = char_to_id['<pad>'] # 96

    MAX_RECOGNITION_SEQ_LEN = 25 
    NUM_POLYGON_POINTS = 16 
    
    # Training parameters
    BATCH_SIZE = 1 # Keep at 1 for now due to complexity of collate_fn for DETR targets
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50 # Increase for more "learning" (even with random init)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TotalTextDataset(json_path=JSON_PATH, img_dir=IMAGE_DIR, transform=transform, 
                               max_recognition_seq_len=MAX_RECOGNITION_SEQ_LEN,
                               padding_value=PADDING_IDX)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)

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
        max_recognition_seq_len=MAX_RECOGNITION_SEQ_LEN
    ).to(device)

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
        'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2, 'loss_recognition': 2,
        'loss_cardinality': 1, 'loss_polygon': 1 # Added polygon loss weight
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

    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Training Loop ---
    print("\nStarting DETR-style training loop...")
    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        criterion.train()
        total_epoch_loss = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            # targets is a list of dictionaries (even for batch_size=1)
            # Move individual tensors within target dicts to device
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            predictions = model(images)

            # --- Calculate Losses using SetCriterion ---
            loss_dict = criterion(predictions, targets)
            
            # Weighted sum of all losses
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(dataloader)}, Total Loss: {loss.item():.4f}", end=' | ')
                for k, v in loss_dict.items():
                    print(f"{k}: {v.item():.4f}", end=' ')
                print()

        print(f"Epoch {epoch+1} finished, Average Total Loss: {total_epoch_loss / len(dataloader):.4f}")

    print("\nDETR-style training finished.")

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

    # print("\n--- IMPORTANT NOTES ON DETR-STYLE TRAINING ---")
    # print("1. This is a conceptual implementation of DETR loss components. ")
    # print("   The matching process (Hungarian algorithm) and loss calculations are more robust.")
    # print("2. `resnet_pretrained` is set to `False`. For actual performance, set it to `True`.")
    # print("3. Positional embeddings for Transformer Encoder/Decoder are omitted for simplicity but are CRUCIAL.")
    # print("4. This code uses a simplified `id_to_char` mapping. A proper vocabulary building is needed.")
    # print("5. DETR's `collate_fn` for batching images (NestedTensor) and targets (list of dicts) ")
    # print("   is more complex than implemented here, especially for variable-sized elements.")
    # print("   The current `collate_fn` only handles `batch_size=1` effectively.")
    # print("6. Polygon loss is an L1 on bezier points, which is a simplification. ")
    # print("   Advanced methods might use specialized polygon IoU or curve matching metrics.")
    # print("7. For robust results, always refer to the official DETR/VimTS/ESTextSpotter implementations.")
