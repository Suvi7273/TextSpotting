import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# --- 1. Feature Extraction Components (from previous response, slightly refined) ---

class ResNet50Backbone(nn.Module):
    """
    ResNet50 backbone for feature extraction.
    Uses layers up to layer3 (conv4_x stage) as described in many detection papers
    to get features before significant downsampling.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        # Features up to layer3 (output stride 16, channels 1024)
        # Can adjust to layer4 (output stride 32, channels 2048) if needed
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3, # Output of layer3 usually has 1024 channels
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

    def forward(self, x):
        # x is assumed to be (Batch, Channel, Height, Width)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1) # Reshape to (B, H*W, C) for Transformer
        
        # In a real implementation, you'd add positional embeddings here
        # E.g., x = x + self.pos_embed(h, w)
        
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
    def __init__(self, feature_dim, num_queries, num_detection_tokens=None, num_recognition_tokens=None):
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable embeddings for initial queries
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        
        # Simplified heads for initial coarse bbox prediction (cx, cy, w, h, objectness)
        # In a real DETR-like model, these would be prediction heads on the output of
        # a Transformer Decoder, or from learned object queries.
        self.bbox_coord_head = nn.Linear(feature_dim, 4) # (cx, cy, w, h)
        self.bbox_score_head = nn.Linear(feature_dim, 1) # objectness score

        # In ESTextSpotter, detection and recognition queries are often derived
        # from the same set of learnable queries, potentially with different linear
        # transformations or attention mechanisms.
        # For simplicity, we can use a portion of the main queries or separate embeddings.
        # The paper suggests "extracted within the coarse bounding box coordinates" for recognition
        # and "transform the bounding box coordinates" for detection.
        # This implies a more complex interaction than simple embeddings.

        # For this demo, we will generate dummy queries with the correct shape.
        # In a true impl, num_detection_tokens and num_recognition_tokens might
        # refer to the length of recognition sequence or dimensionality of query.
        # We assume they refer to the feature_dim here for simplicity.
        self.detection_query_project = nn.Linear(feature_dim, feature_dim)
        self.recognition_query_project = nn.Linear(feature_dim, feature_dim)

    def forward(self, encoded_features):
        # encoded_features: (B, H*W, Feature_dim) from TransformerEncoder

        # The paper states "a liner layer to output the coarse bounding box coordinates
        # and probabilities. Then, the top N coarse bounding box coordinates are selected".
        # This can be implemented by applying a prediction head to a set of learned queries
        # that interact with the encoded_features.
        
        # For this conceptual example, let's just use our learned `query_embed`
        # as the initial queries that get "predicted" on.
        
        # Expand queries to batch size
        initial_queries = self.query_embed.weight.unsqueeze(0).repeat(encoded_features.shape[0], 1, 1) # (B, num_queries, feature_dim)

        # In a real model, `initial_queries` would interact with `encoded_features`
        # (e.g., through cross-attention in a Decoder) to produce refined query embeddings.
        # For Module 1, we'll treat these as our "query features" from which bboxes are predicted.
        
        # Predict coarse bounding box coordinates and scores from initial queries
        # Apply sigmoid to normalize coordinates to [0,1] and scores to [0,1]
        coarse_bboxes_coords = self.bbox_coord_head(initial_queries).sigmoid() # (B, num_queries, 4)
        coarse_bboxes_scores = self.bbox_score_head(initial_queries).sigmoid() # (B, num_queries, 1)
        
        coarse_bboxes_and_scores = torch.cat([coarse_bboxes_coords, coarse_bboxes_scores], dim=-1) # (B, num_queries, 5)

        # "Then, the top N coarse bounding box coordinates are selected based on the probabilities."
        # For this demo, we'll just use all `num_queries` as our 'N'.
        # In practice, you'd sort by scores and pick the top ones.

        # Generate initial detection and recognition queries from the initial queries (or refined ones)
        # The paper's description suggests a more intricate derivation based on the coarse bboxes.
        # Here, we project the `initial_queries` to get task-specific query embeddings.
        detection_queries = self.detection_query_project(initial_queries) # (B, num_queries, feature_dim)
        recognition_queries = self.recognition_query_project(initial_queries) # (B, num_queries, feature_dim)

        return encoded_features, detection_queries, recognition_queries, coarse_bboxes_and_scores

# --- Overall Module 1 Wrapper ---
class VimTSModule1(nn.Module):
    def __init__(self, resnet_pretrained=True,
                 rem_in_channels=1024, rem_out_channels=1024,
                 transformer_feature_dim=1024, transformer_num_heads=8, transformer_num_layers=3,
                 num_queries=100):
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

    def forward(self, img):
        # 1. Feature Extraction
        resnet_features = self.resnet_backbone(img)
        rem_features = self.receptive_enhancement_module(resnet_features)
        
        # The paper says "The output of the REM and ResNet are sent into a Transformer encoder"
        # This implies either concatenation or using features from different scales.
        # For simplicity, we'll use REM features as primary input to the Transformer Encoder here.
        # In a real model, often multi-scale features are used.
        encoded_image_features = self.transformer_encoder(rem_features)

        # 2. Task-aware Query Initialization
        encoded_features_for_decoder, detection_queries, recognition_queries, coarse_bboxes_and_scores = \
            self.task_aware_query_init(encoded_image_features)

        return {
            "encoded_image_features": encoded_features_for_decoder,
            "detection_queries": detection_queries,
            "recognition_queries": recognition_queries,
            "coarse_bboxes_and_scores": coarse_bboxes_and_scores
        }

# --- TotalText Dataset Loader ---
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
            
        # Filter out images without annotations if necessary, or just use images directly
        self.image_infos = {img['id']: img for img in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_path = os.path.join(self.img_dir, img_filename)

        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        annotations = self.img_id_to_annotations.get(img_id, [])
        
        # Process annotations: extract bbox and polygon
        # For simplicity, we'll convert segmentation polygons to a list of lists of floats
        # and bboxes to [x_min, y_min, width, height]
        gt_bboxes = [] # [x_min, y_min, width, height]
        gt_polygons = [] # list of [x1, y1, x2, y2, ...]
        
        for ann in annotations:
            # Bbox format in COCO is [x_min, y_min, width, height]
            gt_bboxes.append(ann['bbox']) 
            # Segmentation is a list of polygons, each polygon is [x1, y1, x2, y2, ...]
            # TotalText often uses a single segmentation polygon per instance
            if ann['segmentation']:
                gt_polygons.extend(ann['segmentation']) # extend to flatten list if multiple polys per instance
            
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32) if gt_bboxes else torch.empty((0, 4), dtype=torch.float32)
        
        # If transform is applied, it will handle resizing/normalization
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image) # Default to Tensor conversion
            
        # We need original dimensions for visualization later
        original_width, original_height = image.size

        return image_tensor, {
            'image_id': img_id,
            'file_name': img_filename,
            'original_size': (original_width, original_height),
            'gt_bboxes': gt_bboxes,
            'gt_polygons': gt_polygons # Keep as list of lists for drawing
        }

# --- Visualization Function ---
def visualize_output(original_image_path, model_output, gt_info, show_gt=True, show_preds=True, score_threshold=0.5):
    """
    Visualizes the original image with ground truth and predicted bounding boxes/polygons.
    
    Args:
        original_image_path (str): Path to the original image file.
        model_output (dict): Output from VimTSModule1.
        gt_info (dict): Ground truth information from the dataset.
        show_gt (bool): Whether to show ground truth annotations.
        show_preds (bool): Whether to show predicted coarse bounding boxes.
        score_threshold (float): Minimum score for predicted bounding boxes to be shown.
    """
    image = Image.open(original_image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    img_width, img_height = image.size

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    # --- Draw Ground Truth ---
    if show_gt:
        # GT BBoxes
        for bbox_coords in gt_info['gt_bboxes']:
            # bbox_coords are [x_min, y_min, width, height]
            x_min, y_min, width, height = bbox_coords.tolist()
            rect = patches.Rectangle((x_min, y_min), width, height, 
                                     linewidth=1, edgecolor='g', facecolor='none', linestyle='--')
            ax.add_patch(rect)
        
        # GT Polygons (more accurate for arbitrarily-shaped text)
        for poly_coords_list in gt_info['gt_polygons']:
            # poly_coords_list is [x1, y1, x2, y2, ..., xN, yN]
            poly = [(poly_coords_list[i], poly_coords_list[i+1]) for i in range(0, len(poly_coords_list), 2)]
            if len(poly) > 0:
                # Pillow's draw.polygon is good for filled, but we want outline
                # Matplotlib's Polygon patch is better for outlines
                polygon_patch = patches.Polygon(poly, closed=True, linewidth=2, edgecolor='b', facecolor='none')
                ax.add_patch(polygon_patch)
        
        plt.text(5, 5, "Green dashed: GT BBox", color='g', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        plt.text(5, 20, "Blue: GT Polygon", color='b', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    # --- Draw Predictions ---
    if show_preds:
        # Predicted coarse_bboxes_and_scores are (B, num_queries, 5)
        # where last dim is (cx, cy, w, h, obj_score) normalized [0,1]
        
        # Take the first image in the batch for visualization
        pred_bboxes = model_output['coarse_bboxes_and_scores'][0]
        
        for bbox_data in pred_bboxes:
            cx_norm, cy_norm, w_norm, h_norm, score = bbox_data.tolist()
            
            if score >= score_threshold:
                # Convert normalized (cx, cy, w, h) to pixel (x_min, y_min, width, height)
                x_min = (cx_norm - w_norm / 2) * img_width
                y_min = (cy_norm - h_norm / 2) * img_height
                width = w_norm * img_width
                height = h_norm * img_height

                rect = patches.Rectangle((x_min, y_min), width, height,
                                         linewidth=1, edgecolor='r', facecolor='none', linestyle='-')
                ax.add_patch(rect)
                plt.text(x_min, y_min - 5, f'{score:.2f}', color='r', fontsize=7, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.text(5, 35, f"Red: Predicted Coarse BBox (score > {score_threshold})", color='r', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    plt.title(f"Image: {gt_info['file_name']}")
    #plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    # Adjust these paths to your setup
    JSON_PATH = '/content/drive/MyDrive/sample/train.json'
    IMAGE_DIR = '/content/drive/MyDrive/sample/img' # Directory where your 0000000.jpg, 0000001.jpg etc. are located

    # Model parameters (example values, tune based on actual implementation/paper)
    FEATURE_DIM = 1024 # Output channels of ResNet50 layer3 and REM
    NUM_QUERIES = 100  # Number of initial queries the model uses
    
    # Preprocessing transform for input images
    # Resize to a fixed size for the model input, then normalize
    # Keep aspect ratio is often done, but for simplicity here we just resize.
    transform = transforms.Compose([
        transforms.Resize((512, 512)), # Example fixed size for model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TotalTextDataset(json_path=JSON_PATH, img_dir=IMAGE_DIR, transform=transform)
    
    # Pick a random image from the dataset to visualize
    sample_idx = random.randint(0, len(dataset) - 1)
    # sample_idx = 0 # Or pick a specific image_id if you want

    image_tensor, gt_info = dataset[sample_idx]
    
    print(f"GT BBoxes for image {gt_info['file_name']}: {gt_info['gt_bboxes'].shape}")
    print(f"GT Polygons for image {gt_info['file_name']}: {len(gt_info['gt_polygons'])} instances")

    # --- Initialize Module 1 ---
    # Set pretrained=True for real usage to leverage ImageNet weights
    model_m1 = VimTSModule1(
        resnet_pretrained=False, # Set to True for real usage
        rem_in_channels=1024,
        rem_out_channels=FEATURE_DIM,
        transformer_feature_dim=FEATURE_DIM,
        transformer_num_heads=8,
        transformer_num_layers=3,
        num_queries=NUM_QUERIES
    )
    # Set model to evaluation mode (important for BatchNorm/Dropout)
    model_m1.eval()

    # --- Forward Pass ---
    with torch.no_grad(): # No gradient calculation needed for inference/visualization
        output_m1 = model_m1(image_tensor)

    print("--- Output of Conceptual Module 1 ---")
    print(f"Encoded Image Features shape: {output_m1['encoded_image_features'].shape}")
    print(f"Detection Queries shape: {output_m1['detection_queries'].shape}")
    print(f"Recognition Queries shape: {output_m1['recognition_queries'].shape}")
    print(f"Coarse BBoxes and Scores shape (cx,cy,w,h,obj_score): {output_m1['coarse_bboxes_and_scores'].shape}")
    
    # Add this to check predicted bboxes and scores
    pred_bboxes_raw = output_m1['coarse_bboxes_and_scores'][0]
    print(f"Number of predicted bboxes before score filtering: {pred_bboxes_raw.shape[0]}")
    high_score_preds = [b for b in pred_bboxes_raw if b[4] >= 0.5] # b[4] is the score
    print(f"Number of predicted bboxes with score >= 0.5: {len(high_score_preds)}")

    original_image_filename = gt_info['file_name']
    original_image_full_path = os.path.join(IMAGE_DIR, original_image_filename)
    
    print(f"\nVisualizing output for image: {original_image_filename}")
    visualize_output(original_image_full_path, output_m1, gt_info, 
                     show_gt=True, show_preds=True, score_threshold=0.5)
    
    print("done!!")
    
    # print("\n--- Additional Notes ---")
    # print("1. 'Encoded Image Features' and 'Detection/Recognition Queries' are high-dimensional tensors.")
    # print("   Their direct visual interpretation is limited without further processing (e.g., dimensionality reduction for features, or using queries in a decoder).")
    # print("2. 'Coarse BBoxes and Scores' are the most directly interpretable output of Module 1 for visualization.")
    # print("3. For actual training, you would pass 'encoded_image_features', 'detection_queries', 'recognition_queries' ")
    # print("   to the subsequent Transformer Decoder and other heads, along with your 'gt_bboxes' and 'gt_polygons' ")
    # print("   for loss calculation (e.g., using Hungarian matching as mentioned in the paper).")
    # print("4. The `rec` field in your JSON would be tokenized and used as ground truth for a text recognition head.")
    # print("5. To get accurate performance, you must use the full VimTS architecture, proper training procedures,")
    # print("   and potentially a pre-trained ResNet50 (`resnet_pretrained=True`).")
    plt.show()
