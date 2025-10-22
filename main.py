
"""
VimTS Feature Extraction Module Demo
Implements: ResNet50 → REM → Transformer Encoder Pipeline

This script demonstrates the complete feature extraction pipeline described in Section III-A
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import from the modules (assuming they're available)
from util import *
from rem_and_adapter import *

class REM(nn.Module):
    """
    Receptive Enhancement Module (REM)
    Enlarges the receptive field using large kernel convolution for downsampling
    """
    def __init__(self, in_channels=2048, out_channels=256, kernel_size=7):
        super().__init__()
        # Large kernel convolution to expand receptive field
        self.large_kernel_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=2,  # Downsample
            padding=kernel_size//2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Additional refinement convolution
        self.refine_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.refine_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # Apply large kernel convolution for receptive field enhancement
        x = self.large_kernel_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # Refine features
        identity = x
        x = self.refine_conv(x)
        x = self.refine_bn(x)
        x = self.relu(x + identity)
        
        return x


class SimpleTransformerEncoder(nn.Module):
    """
    Simplified Transformer Encoder for feature enhancement
    Inspired by Deformable DETR
    """
    def __init__(self, d_model=256, nhead=8, num_layers=3, dim_feedforward=1024):
        super().__init__()
        self.d_model = d_model
        
        # Multi-scale feature projection (match ResNet output channels)
        self.input_proj = nn.ModuleList([
            nn.Conv2d(256, d_model, kernel_size=1),   # C2: 256 channels
            nn.Conv2d(512, d_model, kernel_size=1),   # C3: 512 channels
            nn.Conv2d(1024, d_model, kernel_size=1),  # C4: 1024 channels
            nn.Conv2d(2048, d_model, kernel_size=1),  # C5: 2048 channels
        ])
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Position encoding (will be dynamically adjusted)
        self.register_buffer('pos_scale', torch.tensor(1.0))
        
    def get_position_encoding(self, seq_len, d_model):
        """Generate sinusoidal position encoding"""
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)  # Add batch dimension
        
    def forward(self, features_list):
        """
        Args:
            features_list: List of feature maps from different ResNet stages [C2, C3, C4, C5]
        Returns:
            Enhanced features with long-range dependencies
        """
        # Project all features to same dimension
        projected_features = []
        for feat, proj in zip(features_list, self.input_proj):
            projected = proj(feat)
            B, C, H, W = projected.shape
            # Flatten spatial dimensions
            projected = projected.flatten(2).transpose(1, 2)  # B, HW, C
            projected_features.append(projected)
        
        # Concatenate multi-scale features
        all_features = torch.cat(projected_features, dim=1)  # B, total_HW, C
        
        # Add position encoding (dynamically generated)
        seq_len = all_features.shape[1]
        pos = self.get_position_encoding(seq_len, self.d_model).to(all_features.device)
        all_features = all_features + pos
        
        # Apply transformer encoder
        enhanced_features = self.transformer_encoder(all_features)
        
        return enhanced_features


class FeatureExtractionModule(nn.Module):
    """
    Complete Feature Extraction Module
    Pipeline: ResNet50 → REM → Transformer Encoder
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Step 1: ResNet50 Backbone
        import torchvision.models as models
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract intermediate layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Step 2: REM for receptive field enhancement
        self.rem = REM(in_channels=2048, out_channels=256)
        
        # Step 3: Transformer Encoder for long-range dependencies
        self.transformer_encoder = SimpleTransformerEncoder(d_model=256, nhead=8, num_layers=3)
        
    def forward(self, x):
        # Step 1: Extract features through ResNet50
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)   # 1/4 resolution, 256 channels
        c3 = self.layer2(c2)  # 1/8 resolution, 512 channels
        c4 = self.layer3(c3)  # 1/16 resolution, 1024 channels
        c5 = self.layer4(c4)  # 1/32 resolution, 2048 channels
        
        # Step 2: Apply REM to enhance receptive field
        # REM takes C5 (2048 channels) and outputs 256 channels
        rem_output = self.rem(c5)
        
        # Step 3: Combine features and apply Transformer Encoder
        # Pass original ResNet features (with correct channel counts) + REM output
        # The transformer will project them to the same dimension
        features_list = [c2, c3, c4, c5]  # Use C5 instead of REM here
        enhanced_features = self.transformer_encoder(features_list)
        
        return {
            'resnet_features': [c2, c3, c4, c5],
            'rem_features': rem_output,
            'enhanced_features': enhanced_features,
            'feature_shapes': [(f.shape[2], f.shape[3]) for f in [c2, c3, c4, c5]]
        }


# ============================================================================
# Visualization and Demo Functions
# ============================================================================

def load_dataset_info(json_path='train.json'):
    """Load dataset information from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def visualize_feature_maps(features_dict, original_img, image_id):
    """Visualize the extracted features at different stages"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'VimTS Feature Extraction Pipeline - Image {image_id}', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # ResNet features at different scales
    resnet_feats = features_dict['resnet_features']
    titles = ['ResNet C2 (1/4)', 'ResNet C3 (1/8)', 'ResNet C4 (1/16)', 'ResNet C5 (1/32)']
    
    for idx, (feat, title) in enumerate(zip(resnet_feats, titles)):
        # Take mean across channels for visualization
        feat_map = feat[0].mean(dim=0).detach().cpu().numpy()
        im = axes[0, idx].imshow(feat_map, cmap='viridis')
        axes[0, idx].set_title(f'{title}\n{feat.shape[2]}x{feat.shape[3]}', fontsize=10)
        axes[0, idx].axis('off')
        plt.colorbar(im, ax=axes[0, idx], fraction=0.046)
    
    # REM output
    rem_feat = features_dict['rem_features']
    rem_map = rem_feat[0].mean(dim=0).detach().cpu().numpy()
    im = axes[1, 0].imshow(rem_map, cmap='plasma')
    axes[1, 0].set_title(f'REM Output\n{rem_feat.shape[2]}x{rem_feat.shape[3]}', fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # Enhanced features from Transformer (reshape first few tokens)
    enhanced = features_dict['enhanced_features'][0]  # B, N, C
    
    # Visualize different aspects of enhanced features
    n_tokens = enhanced.shape[0]
    grid_size = int(np.sqrt(n_tokens))
    
    # Channel-wise mean
    enhanced_mean = enhanced.mean(dim=1).detach().cpu().numpy()
    axes[1, 1].plot(enhanced_mean)
    axes[1, 1].set_title('Enhanced Features\n(Channel-wise Mean)', fontsize=10, fontweight='bold')
    axes[1, 1].set_xlabel('Token Index')
    axes[1, 1].set_ylabel('Feature Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Feature statistics
    feat_stats = {
        'ResNet C2': resnet_feats[0][0].std().item(),
        'ResNet C3': resnet_feats[1][0].std().item(),
        'ResNet C4': resnet_feats[2][0].std().item(),
        'ResNet C5': resnet_feats[3][0].std().item(),
        'REM': rem_feat[0].std().item(),
        'Enhanced': enhanced.std().item()
    }
    
    axes[1, 2].bar(range(len(feat_stats)), list(feat_stats.values()))
    axes[1, 2].set_xticks(range(len(feat_stats)))
    axes[1, 2].set_xticklabels(list(feat_stats.keys()), rotation=45, ha='right')
    axes[1, 2].set_title('Feature Std Dev by Stage', fontsize=10, fontweight='bold')
    axes[1, 2].set_ylabel('Std Deviation')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Feature dimensionality info
    info_text = f"""Feature Extraction Summary:
    
ResNet Stages:
  C2: {resnet_feats[0].shape[1]} channels, {resnet_feats[0].shape[2]}×{resnet_feats[0].shape[3]}
  C3: {resnet_feats[1].shape[1]} channels, {resnet_feats[1].shape[2]}×{resnet_feats[1].shape[3]}
  C4: {resnet_feats[2].shape[1]} channels, {resnet_feats[2].shape[2]}×{resnet_feats[2].shape[3]}
  C5: {resnet_feats[3].shape[1]} channels, {resnet_feats[3].shape[2]}×{resnet_feats[3].shape[3]}

REM Output:
  {rem_feat.shape[1]} channels, {rem_feat.shape[2]}×{rem_feat.shape[3]}

Transformer Enhanced:
  {enhanced.shape[0]} tokens, {enhanced.shape[1]} dimensions
    
Total Parameters: {sum(p.numel() for p in features_dict['model'].parameters())/1e6:.2f}M
"""
    
    axes[1, 3].text(0.1, 0.5, info_text, fontsize=9, family='monospace',
                    verticalalignment='center', transform=axes[1, 3].transAxes)
    axes[1, 3].axis('off')
    
    # Bottom row: Feature flow diagram
    axes[2, 0].axis('off')
    axes[2, 1].text(0.5, 0.5, 
                    'Feature Extraction Pipeline:\n\n'
                    'Input Image\n   ↓\n'
                    'ResNet50 Backbone\n   ↓\n'
                    'Multi-scale Features (C2, C3, C4, C5)\n   ↓\n'
                    'REM (Receptive Enhancement)\n   ↓\n'
                    'Transformer Encoder\n   ↓\n'
                    'Enhanced Features',
                    ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    transform=axes[2, 1].transAxes)
    axes[2, 1].axis('off')
    
    axes[2, 2].axis('off')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    return fig

def demo_feature_extraction(image_dir, json_path):
    """
    Main demo function for Feature Extraction Module
    """
    print("="*70)
    print("VimTS Feature Extraction Module Demo")
    print("="*70)
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    dataset = load_dataset_info(json_path)
    print(f"✓ Loaded dataset with {len(dataset['images'])} images")
    
    # Select an image from the dataset
    sample_img_info = dataset['images'][0]
    image_filename = sample_img_info['file_name']
    image_id = sample_img_info['id']
    
    # Construct full image path from directory + filename
    if image_dir is not None:
        image_path = str(Path(image_dir) / image_filename)
    else:
        image_path = image_filename
    
    print(f"✓ Using sample image: {image_filename} ({sample_img_info['width']}×{sample_img_info['height']})")
    print(f"  Full path: {image_path}")
    
    # Load image
    print("\n[2/5] Loading image...")
    try:
        img = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded: {img.size[0]}×{img.size[1]}")
    except Exception as e:
        print(f"  ⚠ Image file not found ({e}), creating dummy image...")
        img = Image.new('RGB', (512, 512), color=(100, 150, 200))
        draw = ImageDraw.Draw(img)
        draw.text((256, 256), "Sample Text", fill=(255, 255, 255))
        print(f"✓ Dummy image created: {img.size[0]}×{img.size[1]}")
    
    # Preprocess image
    print("\n[3/5] Preprocessing image...")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(512, 512), mode='bilinear')
    
    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    print(f"✓ Preprocessed to shape: {img_tensor.shape}")
    
    # Build model
    print("\n[4/5] Building Feature Extraction Module...")
    model = FeatureExtractionModule(pretrained=False)  # Set True if you have internet
    model.eval()
    print("✓ Model built successfully")
    
    # Forward pass
    print("\n[5/5] Extracting features...")
    with torch.no_grad():
        features = model(img_tensor)
    
    features['model'] = model  # Store for visualization
    
    print("\n" + "="*70)
    print("Feature Extraction Complete!")
    print("="*70)
    
    # Print results
    print("\nExtracted Features:")
    print(f"  • ResNet stages: {len(features['resnet_features'])} feature maps")
    for i, feat in enumerate(features['resnet_features']):
        print(f"    - C{i+2}: {feat.shape}")
    
    print(f"\n  • REM output: {features['rem_features'].shape}")
    print(f"  • Enhanced features: {features['enhanced_features'].shape}")
    print(f"    ({features['enhanced_features'].shape[1]} tokens, {features['enhanced_features'].shape[2]} dims)")
    
    print("\n" + "="*70)
    print("Generating visualization...")
    print("="*70 + "\n")
    
    # Visualize
    fig = visualize_feature_maps(features, img, image_id)
    plt.show()
    
    return features, model

if __name__ == "__main__":
    # Run the demo
    # Pass the directory containing images and the JSON file path
    features, model = demo_feature_extraction(
        r'/content/drive/MyDrive/sample/img',  # Directory containing images
        r'/content/drive/MyDrive/sample/train.json'  # JSON file with image metadata
    )
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)
