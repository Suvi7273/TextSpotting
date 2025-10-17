"""
VimTS Module 1 Data Flow Demonstration
This script shows how data transforms through the backbone (Module 1)
Usage: python main.py
"""

import torch
import json
import os
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import our modules
from util import (
    NestedTensor, 
    nested_tensor_from_tensor_list,
    build_position_encoding,
    test_module1_components
)
from rem_and_adapter import (
    build_backbone,
    test_module2_components
)

# ============================================================================
# 1. Data Loading and Preprocessing
# ============================================================================

class SimpleTextSpottingDataset:
    """Simple dataset loader for text spotting demonstration"""
    
    def __init__(self, json_path, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Load annotations
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {json_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load image
        img_path = self.img_dir / sample['image']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'image_id': sample.get('image_id', idx),
            'filename': sample['image'],
            'original_size': image.shape[-2:] if isinstance(image, torch.Tensor) else image.size[::-1]
        }

def get_transform(train=False):
    """Get image transform pipeline"""
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return T.Compose(transforms)

# ============================================================================
# 2. Visualization and Analysis Functions
# ============================================================================

def print_tensor_stats(name, tensor):
    """Print statistics about a tensor"""
    if isinstance(tensor, torch.Tensor):
        print(f"\n  {name}:")
        print(f"    Shape: {tensor.shape}")
        print(f"    Dtype: {tensor.dtype}")
        print(f"    Device: {tensor.device}")
        print(f"    Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        print(f"    Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")
    else:
        print(f"\n  {name}: {type(tensor)}")

def visualize_nested_tensor(nested_tensor, level_name=""):
    """Visualize a NestedTensor structure"""
    print(f"\n{'='*60}")
    print(f"NestedTensor Analysis - {level_name}")
    print(f"{'='*60}")
    
    tensors, mask = nested_tensor.decompose()
    
    print(f"\nTensor Component:")
    print(f"  Shape: {tensors.shape}")
    print(f"  Channels: {tensors.shape[1]}")
    print(f"  Spatial dims: {tensors.shape[2]} x {tensors.shape[3]}")
    
    print(f"\nMask Component:")
    print(f"  Shape: {mask.shape}")
    print(f"  True (padded) pixels: {mask.sum().item()}")
    print(f"  False (valid) pixels: {(~mask).sum().item()}")
    
    # Analyze per image in batch
    batch_size = tensors.shape[0]
    print(f"\nPer-Image Analysis (Batch size: {batch_size}):")
    for i in range(batch_size):
        valid_pixels = (~mask[i]).sum().item()
        total_pixels = mask[i].numel()
        print(f"  Image {i}: {valid_pixels}/{total_pixels} valid pixels ({100*valid_pixels/total_pixels:.1f}%)")

def analyze_backbone_features(features_dict, pos_encodings):
    """Analyze multi-scale features from backbone"""
    print(f"\n{'='*60}")
    print("Multi-Scale Feature Analysis")
    print(f"{'='*60}")
    
    print(f"\nNumber of feature levels: {len(features_dict)}")
    
    for level_idx, (level_key, nested_feat) in enumerate(features_dict.items()):
        print(f"\n--- Level {level_idx} (Key: {level_key}) ---")
        
        # Feature tensor
        feat_tensor, feat_mask = nested_feat.decompose()
        print(f"Features:")
        print(f"  Shape: {feat_tensor.shape}")
        print(f"  Channels: {feat_tensor.shape[1]}")
        print(f"  Spatial: {feat_tensor.shape[2]}x{feat_tensor.shape[3]}")
        print(f"  Memory: {feat_tensor.element_size() * feat_tensor.nelement() / 1024**2:.2f} MB")
        
        # Position encoding
        pos = pos_encodings[level_idx]
        print(f"\nPosition Encoding:")
        print(f"  Shape: {pos.shape}")
        print(f"  Channels: {pos.shape[1]}")
        
        # Receptive field info (approximate)
        stride = 2 ** (level_idx + 2)  # Assuming standard ResNet strides
        print(f"\nReceptive Field Info:")
        print(f"  Approx. stride: {stride}x")
        print(f"  Each feature covers ~{stride}x{stride} pixels in original image")

def visualize_feature_maps(features_dict, pos_encodings, samples, output_dir='visualizations'):
    """
    Visualize feature maps as heatmaps
    Shows original image, feature channels, and position encodings
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Visualizing Feature Maps as Heatmaps")
    print(f"{'='*60}")
    
    for level_idx, (level_key, nested_feat) in enumerate(features_dict.items()):
        feat_tensor, feat_mask = nested_feat.decompose()
        pos = pos_encodings[level_idx]
        
        batch_size = feat_tensor.shape[0]
        num_channels = feat_tensor.shape[1]
        
        print(f"\nLevel {level_idx}: Visualizing {num_channels} channels...")
        
        # Visualize for each image in batch
        for b in range(batch_size):
            print(f"  Creating visualization for image {b}...")
            
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(4, 5, hspace=0.3, wspace=0.3)
            
            # Title
            fig.suptitle(f'Feature Maps - Level {level_idx} - Image {b}\n'
                        f'Shape: {feat_tensor.shape[1]}x{feat_tensor.shape[2]}x{feat_tensor.shape[3]}',
                        fontsize=16, fontweight='bold')
            
            # Show original image
            ax_orig = fig.add_subplot(gs[0, :2])
            try:
                # Load and display original image
                img_path = Path(samples[b]['filename'])
                if not img_path.is_absolute():
                    img_path = Path('images') / img_path
                orig_img = Image.open(img_path).convert('RGB')
                ax_orig.imshow(orig_img)
                ax_orig.set_title(f'Original Image\n{samples[b]["filename"]}', fontweight='bold')
                ax_orig.axis('off')
            except:
                ax_orig.text(0.5, 0.5, 'Original image\nnot available', 
                           ha='center', va='center')
                ax_orig.axis('off')
            
            # Show mask
            ax_mask = fig.add_subplot(gs[0, 2])
            mask_vis = feat_mask[b].cpu().numpy()
            ax_mask.imshow(mask_vis, cmap='RdYlGn_r')
            ax_mask.set_title(f'Mask\n{(~feat_mask[b]).sum().item()} valid pixels', fontweight='bold')
            ax_mask.axis('off')
            
            # Show average feature activation across all channels
            ax_avg = fig.add_subplot(gs[0, 3])
            avg_feat = feat_tensor[b].mean(dim=0).cpu().numpy()
            im_avg = ax_avg.imshow(avg_feat, cmap='viridis')
            ax_avg.set_title('Average Feature\nActivation', fontweight='bold')
            ax_avg.axis('off')
            plt.colorbar(im_avg, ax=ax_avg, fraction=0.046)
            
            # Show position encoding (first 2 channels combined)
            ax_pos = fig.add_subplot(gs[0, 4])
            pos_vis = (pos[b, 0] + pos[b, 1]).cpu().numpy()
            im_pos = ax_pos.imshow(pos_vis, cmap='twilight')
            ax_pos.set_title('Position Encoding\n(combined)', fontweight='bold')
            ax_pos.axis('off')
            plt.colorbar(im_pos, ax=ax_pos, fraction=0.046)
            
            # Show first 15 feature channels
            channels_to_show = min(15, num_channels)
            for i in range(channels_to_show):
                row = 1 + i // 5
                col = i % 5
                ax = fig.add_subplot(gs[row, col])
                
                # Get feature map for this channel
                feat_map = feat_tensor[b, i].cpu().numpy()
                
                # Apply mask to show only valid regions
                masked_feat = feat_map.copy()
                masked_feat[mask_vis] = np.nan
                
                # Plot
                im = ax.imshow(masked_feat, cmap='viridis', interpolation='nearest')
                ax.set_title(f'Ch {i}\n'
                           f'μ={feat_map.mean():.2f}, σ={feat_map.std():.2f}',
                           fontsize=9)
                ax.axis('off')
                
                # Add tiny colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Add statistics text
            stats_text = f"""
Level {level_idx} Statistics:
• Total channels: {num_channels}
• Spatial size: {feat_tensor.shape[2]}x{feat_tensor.shape[3]}
• Receptive field: ~{2**(level_idx+2)}x stride
• Active channels (>0.1): {(feat_tensor[b].mean(dim=[1,2]) > 0.1).sum().item()}
• Min activation: {feat_tensor[b].min():.3f}
• Max activation: {feat_tensor[b].max():.3f}
• Mean activation: {feat_tensor[b].mean():.3f}
            """
            
            fig.text(0.02, 0.02, stats_text, fontsize=10, 
                    verticalalignment='bottom', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Save figure
            output_path = os.path.join(output_dir, 
                                      f'features_level{level_idx}_img{b}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"    ✓ Saved: {output_path}")
            plt.close()
    
    print(f"\n✓ All visualizations saved to '{output_dir}/' directory")
    print(f"  Check the PNG files to see feature map heatmaps!")

def visualize_tensor_samples(features_dict, output_dir='visualizations'):
    """Print small samples of actual tensor values"""
    print(f"\n{'='*60}")
    print("Tensor Value Samples (First 5x5 of First Channel)")
    print(f"{'='*60}")
    
    for level_idx, (level_key, nested_feat) in enumerate(features_dict.items()):
        feat_tensor, _ = nested_feat.decompose()
        
        print(f"\nLevel {level_idx}:")
        print(f"Full shape: {feat_tensor.shape}")
        print(f"\nFirst image, first channel, top-left 5x5 corner:")
        print("-" * 50)
        
        # Get 5x5 sample
        sample = feat_tensor[0, 0, :5, :5].cpu().numpy()
        
        # Print nicely formatted
        for i in range(sample.shape[0]):
            row_str = "  ".join([f"{val:7.3f}" for val in sample[i]])
            print(f"  [{row_str}]")
        
        print(f"\nThese are the actual numbers the model uses!")
        print(f"Each number represents learned feature activations")

# ============================================================================
# 3. Main Demo Function
# ============================================================================

def demo_module1_data_flow(json_path='train.json', img_dir='images', num_samples=2):
    """
    Demonstrate data flow through Module 1
    
    Args:
        json_path: Path to train.json annotation file
        img_dir: Directory containing images
        num_samples: Number of samples to process
    """
    
    print("\n" + "="*70)
    print("VimTS MODULE 1 DATA FLOW DEMONSTRATION")
    print("="*70)
    
    # Check files exist
    if not os.path.exists(json_path):
        print(f"\nError: {json_path} not found!")
        print("Please create a train.json file with format:")
        print('[{"image": "img1.jpg", "image_id": 1}, ...]')
        return
    
    if not os.path.exists(img_dir):
        print(f"\nError: {img_dir} directory not found!")
        return
    
    # ========================================
    # STEP 1: Initialize Components
    # ========================================
    print("\n" + "-"*70)
    print("STEP 1: Initializing VimTS Components")
    print("-"*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build backbone
    print("\nBuilding backbone...")
    backbone = build_backbone()
    backbone = backbone.to(device)
    backbone.eval()  # Set to eval mode
    print("✓ Backbone initialized")
    
    # Load dataset
    print("\nLoading dataset...")
    transform = get_transform(train=False)
    dataset = SimpleTextSpottingDataset(json_path, img_dir, transform)
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # ========================================
    # STEP 2: Load and Preprocess Images
    # ========================================
    print("\n" + "-"*70)
    print("STEP 2: Loading and Preprocessing Images")
    print("-"*70)
    
    samples = []
    image_tensors = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        samples.append(sample)
        image_tensors.append(sample['image'])
        
        print(f"\nSample {i}:")
        print(f"  Filename: {sample['filename']}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Original size: {sample['original_size']}")
    
    # ========================================
    # STEP 3: Create NestedTensor (Batching)
    # ========================================
    print("\n" + "-"*70)
    print("STEP 3: Creating NestedTensor (Batch with Padding)")
    print("-"*70)
    
    print("\nImages have different sizes, creating padded batch...")
    nested_input = nested_tensor_from_tensor_list(image_tensors)
    nested_input = nested_input.to(device)
    
    visualize_nested_tensor(nested_input, "Input Batch")
    
    # ========================================
    # STEP 4: Backbone Forward Pass
    # ========================================
    print("\n" + "-"*70)
    print("STEP 4: Backbone Forward Pass (Feature Extraction)")
    print("-"*70)
    
    print("\nExtracting multi-scale features...")
    with torch.no_grad():
        features_list, pos_encodings = backbone(nested_input)
    
    print(f"✓ Extracted {len(features_list)} feature levels")
    
    # Convert list to dict for easier access
    features_dict = {str(i): feat for i, feat in enumerate(features_list)}
    
    # ========================================
    # STEP 5: Analyze Features
    # ========================================
    print("\n" + "-"*70)
    print("STEP 5: Analyzing Extracted Features")
    print("-"*70)
    
    analyze_backbone_features(features_dict, pos_encodings)
    
    # ========================================
    # STEP 6: Detailed Feature Inspection
    # ========================================
    print("\n" + "-"*70)
    print("STEP 6: Detailed Feature Inspection")
    print("-"*70)
    
    # Inspect highest resolution features (usually level 0)
    print("\nInspecting highest resolution features (Level 0):")
    feat0 = features_list[0]
    pos0 = pos_encodings[0]
    
    feat_tensor, feat_mask = feat0.decompose()
    
    print(f"\nFeature Tensor Statistics:")
    print_tensor_stats("Features", feat_tensor)
    print_tensor_stats("Position Encoding", pos0)
    
    # Show feature activation patterns
    print(f"\nFeature Activation Analysis:")
    batch_size = feat_tensor.shape[0]
    for b in range(batch_size):
        feat_img = feat_tensor[b]  # C x H x W
        channel_means = feat_img.mean(dim=[1, 2])  # Average over spatial dims
        
        print(f"\n  Image {b}:")
        print(f"    Active channels (>0.1): {(channel_means > 0.1).sum().item()}/{len(channel_means)}")
        print(f"    Top 5 channel activations: {channel_means.topk(5).values.cpu().numpy()}")
    
    # ========================================
    # STEP 6.5: Visualize Feature Maps
    # ========================================
    print("\n" + "-"*70)
    print("STEP 6.5: Creating Feature Map Visualizations")
    print("-"*70)
    
    visualize_feature_maps(features_dict, pos_encodings, samples)
    visualize_tensor_samples(features_dict)
    
    # ========================================
    # STEP 7: Memory Analysis
    # ========================================
    print("\n" + "-"*70)
    print("STEP 7: Memory Analysis")
    print("-"*70)
    
    total_memory = 0
    print("\nMemory usage per component:")
    
    # Input
    input_mem = nested_input.tensors.element_size() * nested_input.tensors.nelement() / 1024**2
    print(f"  Input batch: {input_mem:.2f} MB")
    total_memory += input_mem
    
    # Features
    for i, feat in enumerate(features_list):
        feat_tensor, _ = feat.decompose()
        feat_mem = feat_tensor.element_size() * feat_tensor.nelement() / 1024**2
        print(f"  Feature level {i}: {feat_mem:.2f} MB")
        total_memory += feat_mem
    
    # Position encodings
    for i, pos in enumerate(pos_encodings):
        pos_mem = pos.element_size() * pos.nelement() / 1024**2
        print(f"  Position encoding {i}: {pos_mem:.2f} MB")
        total_memory += pos_mem
    
    print(f"\nTotal feature memory: {total_memory:.2f} MB")
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("MODULE 1 DATA FLOW SUMMARY")
    print("="*70)
    
    print(f"""
Input:
  - Batch size: {nested_input.tensors.shape[0]}
  - Image size: {nested_input.tensors.shape[2]}x{nested_input.tensors.shape[3]} (padded)
  - Channels: {nested_input.tensors.shape[1]}

Output (Multi-scale Features):
  - Number of levels: {len(features_list)}
  - Channel counts: {[f.tensors.shape[1] for f in features_list]}
  - Spatial dimensions: {[f'{f.tensors.shape[2]}x{f.tensors.shape[3]}' for f in features_list]}

Key Transformations:
  1. Images → Padded Batch (NestedTensor)
  2. Batch → ResNet Backbone → Multi-scale Features
  3. Features → Position Encodings added
  4. Ready for next module (Transformer Encoder)

These multi-scale features will be fed to:
  - Module 2: Transformer Encoder (not implemented in this demo)
  - Module 3: Deformable Attention
  - Module 4: Text Detection/Recognition Heads
    """)
    
    print("="*70)
    print("Demo completed successfully!")
    print("="*70)
    
    return features_dict, pos_encodings, samples

# ============================================================================
# 4. Create Sample Dataset Helper
# ============================================================================

def create_sample_dataset(output_dir='sample_data'):
    """Create a minimal sample dataset for testing"""
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    # Create dummy images
    images = []
    for i in range(3):
        # Random sizes to test padding
        h, w = np.random.randint(200, 400), np.random.randint(300, 500)
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img)
        
        filename = f'sample_{i}.jpg'
        img_pil.save(os.path.join(img_dir, filename))
        images.append({'image': filename, 'image_id': i})
    
    # Create train.json
    json_path = os.path.join(output_dir, 'train.json')
    with open(json_path, 'w') as f:
        json.dump(images, f, indent=2)
    
    print(f"Sample dataset created in {output_dir}/")
    print(f"  - {len(images)} images in {img_dir}/")
    print(f"  - Annotations in {json_path}")
    
    return json_path, img_dir

# ============================================================================
# 5. Main Entry Point
# ============================================================================

if __name__ == '__main__':
    print("VimTS Module 1 Demo - Data Flow Visualization\n")
    
    # Check if dataset exists
    if not os.path.exists('train.json'):
        print("No train.json found. Creating sample dataset...")
        json_path, img_dir = create_sample_dataset()
    else:
        json_path = 'train.json'
        img_dir = 'images'
    
    # Run the demo
    try:
        features, positions, samples = demo_module1_data_flow(
            json_path=json_path,
            img_dir=img_dir,
            num_samples=2
        )
        
        print("\n✓ All transformations completed successfully!")
        print("\nNext steps:")
        print("- Features are ready for transformer encoder")
        print("- Position encodings enable spatial awareness")
        print("- Multi-scale features capture different levels of detail")
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
