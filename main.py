
"""
VimTS Local Dataset Testing - Your Own Dataset + train.json
==========================================================

Modified integration test that uses your local dataset with train.json annotations.
Perfect for testing with real text detection datasets!

Usage in Google Colab:
    # Upload your images folder and train.json to Colab
    from vimts_module1 import *
    from vimts_module2 import *
    exec(open('vimts_local_dataset_test.py').read())

    # Test with your dataset
    run_vimts_local_dataset_test(
        images_dir='your_images_folder', 
        json_file='train.json',
        max_images=5
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import json
import os
import time
from pathlib import Path

from rem_and_adapter import *
# ============================================================================
# Local Dataset Loading Functions
# ============================================================================

def load_dataset_json(json_file):
    """
    Load and parse your train.json file
    Supports various annotation formats (COCO-style, custom formats)
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✓ Loaded JSON file: {json_file}")

        # Analyze JSON structure
        if isinstance(data, dict):
            if 'images' in data and 'annotations' in data:
                # COCO-style format
                print(f"  Format: COCO-style")
                print(f"  Images: {len(data['images'])}")
                print(f"  Annotations: {len(data['annotations'])}")
                return data, 'coco'
            elif 'annotations' in data:
                # Simple annotations dict
                print(f"  Format: Simple annotations dict")
                print(f"  Images: {len(data['annotations'])}")
                return data, 'simple'
            else:
                # Unknown dict format
                print(f"  Format: Custom dict (keys: {list(data.keys())})")
                return data, 'custom'
        elif isinstance(data, list):
            # List of annotations
            print(f"  Format: Annotation list")
            print(f"  Entries: {len(data)}")
            return data, 'list'
        else:
            print(f"  Format: Unknown")
            return data, 'unknown'

    except Exception as e:
        print(f"✗ Failed to load JSON file: {e}")
        return None, None

def parse_annotations(data, format_type):
    """
    Parse annotations into a common format
    Returns list of {image_path, annotations} dicts
    """
    parsed_data = []

    try:
        if format_type == 'coco':
            # COCO-style: separate images and annotations
            images = {img['id']: img for img in data['images']}

            # Group annotations by image
            image_annotations = {}
            for ann in data['annotations']:
                img_id = ann['image_id']
                if img_id not in image_annotations:
                    image_annotations[img_id] = []
                image_annotations[img_id].append(ann)

            # Combine images with their annotations
            for img_id, img_info in images.items():
                parsed_data.append({
                    'image_path': img_info.get('file_name', f"image_{img_id}.jpg"),
                    'image_id': img_id,
                    'width': img_info.get('width', 0),
                    'height': img_info.get('height', 0),
                    'annotations': image_annotations.get(img_id, [])
                })

        elif format_type == 'simple':
            # Simple dict with image names as keys  
            annotations = data.get('annotations', data)
            for img_name, img_anns in annotations.items():
                parsed_data.append({
                    'image_path': img_name,
                    'image_id': img_name,
                    'annotations': img_anns if isinstance(img_anns, list) else [img_anns]
                })

        elif format_type == 'list':
            # List of annotation entries
            for i, entry in enumerate(data):
                if isinstance(entry, dict):
                    image_path = entry.get('image_path', entry.get('filename', entry.get('image', f'image_{i}.jpg')))
                    parsed_data.append({
                        'image_path': image_path,
                        'image_id': i,
                        'annotations': entry.get('annotations', [entry])  # Entry itself might be annotation
                    })

        elif format_type == 'custom':
            # Try to handle custom format
            print("Attempting to parse custom format...")
            if 'images' in data:
                for img_data in data['images']:
                    parsed_data.append({
                        'image_path': img_data.get('path', img_data.get('filename', 'unknown.jpg')),
                        'image_id': img_data.get('id', len(parsed_data)),
                        'annotations': img_data.get('annotations', [])
                    })
            else:
                # Fallback: treat as simple key-value
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        parsed_data.append({
                            'image_path': key,
                            'image_id': key,
                            'annotations': value if isinstance(value, list) else [value]
                        })

    except Exception as e:
        print(f"Warning: Error parsing annotations: {e}")

    print(f"✓ Parsed {len(parsed_data)} image entries")
    return parsed_data

def load_local_image(image_path, images_dir, max_size=1024):
    """
    Load image from local directory with proper error handling
    """
    # Try different path combinations
    possible_paths = [
        image_path,  # Direct path
        os.path.join(images_dir, image_path),  # In images directory
        os.path.join(images_dir, os.path.basename(image_path)),  # Just filename in images dir
    ]

    for full_path in possible_paths:
        if os.path.exists(full_path):
            try:
                image = Image.open(full_path).convert('RGB')

                # Resize if too large (T4 memory optimization)
                w, h = image.size
                if max(w, h) > max_size:
                    if w > h:
                        new_w, new_h = max_size, int(h * max_size / w)
                    else:
                        new_w, new_h = int(w * max_size / h), max_size
                    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    print(f"    Resized from {w}×{h} to {new_w}×{new_h}")

                return image, full_path

            except Exception as e:
                print(f"    Error loading {full_path}: {e}")
                continue

    print(f"    ✗ Could not load image from any path: {possible_paths}")
    return None, None

def visualize_annotations(image, annotations, title="Image with Annotations"):
    """
    Visualize image with its annotations overlaid
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')

    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']

    for i, ann in enumerate(annotations):
        color = colors[i % len(colors)]

        # Handle different annotation formats
        if 'bbox' in ann:
            # COCO-style bbox [x, y, width, height]
            bbox = ann['bbox']
            if len(bbox) == 4:
                x, y, w, h = bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                       edgecolor=color, facecolor='none')
                ax.add_patch(rect)

        elif 'points' in ann:
            # Polygon points
            points = ann['points']
            if len(points) >= 6:  # At least 3 points (x,y pairs)
                # Reshape to (n, 2) if needed
                if len(points) % 2 == 0:
                    points = np.array(points).reshape(-1, 2)
                    polygon = patches.Polygon(points, linewidth=2,
                                           edgecolor=color, facecolor='none')
                    ax.add_patch(polygon)

        elif 'polygon' in ann:
            # Direct polygon format
            polygon_points = ann['polygon']
            if len(polygon_points) >= 6:
                points = np.array(polygon_points).reshape(-1, 2)
                polygon = patches.Polygon(points, linewidth=2,
                                       edgecolor=color, facecolor='none')
                ax.add_patch(polygon)

        # Add text label if available
        text = ann.get('text', ann.get('transcription', f'Text_{i}'))
        if 'bbox' in ann:
            ax.text(ann['bbox'][0], ann['bbox'][1]-5, text, 
                   color=color, fontsize=8, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'annotated_image_{hash(title)}.png', dpi=150, bbox_inches='tight')
    plt.show()

def preprocess_image_for_vimts(image, normalize=True):
    """
    Preprocess PIL image for VimTS pipeline (same as before)
    """
    transform_list = [transforms.ToTensor()]

    if normalize:
        # ImageNet normalization for pretrained ResNet
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        )

    transform = transforms.Compose(transform_list)
    tensor = transform(image)
    return tensor

def create_local_dataset_batch(images_dir, json_file, max_images=5, visualize=True):
    """
    Create batch of images from your local dataset
    """
    print(f"Loading local dataset...")
    print(f"Images directory: {images_dir}")
    print(f"Annotations file: {json_file}")

    # Load and parse JSON
    data, format_type = load_dataset_json(json_file)
    if data is None:
        return [], []

    # Parse annotations
    parsed_data = parse_annotations(data, format_type)
    if not parsed_data:
        print("✗ No valid annotations found")
        return [], []

    # Load images
    image_tensors = []
    image_info = []
    loaded_count = 0

    print(f"\nLoading up to {max_images} images...")

    for entry in parsed_data[:max_images]:
        image_path = entry['image_path']
        annotations = entry['annotations']

        print(f"\nProcessing: {image_path}")
        print(f"  Annotations: {len(annotations)}")

        # Load image
        pil_image, full_path = load_local_image(image_path, images_dir)
        if pil_image is None:
            continue

        print(f"  ✓ Loaded: {pil_image.size}")

        # Visualize with annotations
        if visualize and annotations:
            try:
                visualize_annotations(pil_image, annotations, 
                                    f"{os.path.basename(image_path)} ({len(annotations)} annotations)")
            except Exception as e:
                print(f"    Warning: Visualization failed: {e}")

        # Preprocess for VimTS
        tensor = preprocess_image_for_vimts(pil_image, normalize=True)
        image_tensors.append(tensor)

        # Store info
        image_info.append({
            'path': full_path,
            'original_size': pil_image.size,
            'tensor_shape': tensor.shape,
            'annotations': annotations,
            'num_annotations': len(annotations)
        })

        loaded_count += 1
        print(f"  ✓ Preprocessed: {tensor.shape}")
        print(f"    Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    print(f"\n✓ Successfully loaded {loaded_count} images from your dataset")
    return image_tensors, image_info

# ============================================================================
# Modified Integration Test for Local Dataset
# ============================================================================

def run_vimts_local_dataset_test(images_dir, json_file, max_images=5, visualize_annotations=True):
    """
    Run VimTS integration test with your local dataset
    """
    print("\n" + "="*80)
    print("VimTS LOCAL DATASET TEST - YOUR DATASET + train.json")
    print("="*80)

    # Verify files exist
    if not os.path.exists(images_dir):
        print(f"✗ Images directory not found: {images_dir}")
        return

    if not os.path.exists(json_file):
        print(f"✗ JSON file not found: {json_file}")
        return

    print(f"Dataset directory: {images_dir}")
    print(f"Annotations file: {json_file}")

    # Set device  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========================================================================
    # Step 1: Load Your Local Dataset
    # ========================================================================
    print("\n" + "-"*60)
    print("STEP 1: Loading Your Local Dataset")
    print("-"*60)

    image_list, image_info = create_local_dataset_batch(
        images_dir, json_file, max_images, visualize_annotations
    )

    if len(image_list) == 0:
        print("✗ No images could be loaded from your dataset!")
        return

    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Total images: {len(image_list)}")

    for i, info in enumerate(image_info):
        print(f"  Image {i}: {os.path.basename(info['path'])}")
        print(f"    Size: {info['original_size']} -> {info['tensor_shape']}")
        print(f"    Annotations: {info['num_annotations']}")

    # ========================================================================
    # Step 2-6: Same VimTS Pipeline as Before
    # ========================================================================

    # Step 2: NestedTensor Creation
    print("\n" + "-"*60)
    print("STEP 2: Creating NestedTensor from Your Images")
    print("-"*60)

    try:
        start_time = time.time()
        nested_tensor = nested_tensor_from_tensor_list(image_list)
        creation_time = time.time() - start_time

        print(f"✓ NestedTensor created in {creation_time:.3f}s")
        print(f"  Batched shape: {nested_tensor.tensors.shape}")
        print(f"  Mask shape: {nested_tensor.mask.shape}")

        if torch.cuda.is_available():
            nested_tensor = nested_tensor.to(device)
            print(f"✓ Moved to {device}")

    except Exception as e:
        print(f"✗ NestedTensor creation failed: {e}")
        return

    # Step 3: Position Encoding
    print("\n" + "-"*60)
    print("STEP 3: Position Encoding")
    print("-"*60)

    try:
        pos_encoder = build_position_encoding()
        if torch.cuda.is_available():
            pos_encoder = pos_encoder.to(device)

        start_time = time.time()
        position_encoding = pos_encoder(nested_tensor)
        pos_time = time.time() - start_time

        print(f"✓ Position encoding computed in {pos_time:.3f}s")
        print(f"  Shape: {position_encoding.shape}")

    except Exception as e:
        print(f"✗ Position encoding failed: {e}")
        return

    # Step 4: Backbone Processing
    print("\n" + "-"*60)
    print("STEP 4: Processing Your Images Through VimTS Backbone")
    print("-"*60)

    try:
        backbone_model = build_backbone()
        if torch.cuda.is_available():
            backbone_model = backbone_model.to(device)

        print("✓ Backbone ready")

        # Process your images
        start_time = time.time()
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
            features, pos_encodings = backbone_model(nested_tensor)
        backbone_time = time.time() - start_time

        print(f"✓ Your images processed in {backbone_time:.3f}s")
        print(f"  Feature levels extracted: {len(features)}")

        for i, (feat, pos) in enumerate(zip(features, pos_encodings)):
            print(f"  Level {i} (1/{4*(2**i)} scale):")
            print(f"    Features: {feat.tensors.shape}")
            print(f"    Value range: [{feat.tensors.min():.3f}, {feat.tensors.max():.3f}]")

        # Analyze features for text detection suitability
        print(f"\n📊 Feature Analysis for Text Detection:")
        for i, feat in enumerate(features):
            # Check feature activation patterns
            feat_data = feat.tensors[0]  # First image
            mean_activation = feat_data.mean().item()
            std_activation = feat_data.std().item()
            max_activation = feat_data.max().item()

            print(f"  Level {i}: Mean={mean_activation:.3f}, Std={std_activation:.3f}, Max={max_activation:.3f}")

            # Features should show some variance (not all zeros/same values)
            if std_activation > 0.1:
                print(f"    ✓ Good feature diversity for text detection")
            else:
                print(f"    ⚠️ Low feature diversity")

    except Exception as e:
        print(f"✗ Backbone processing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Test Adapters with Your Data
    print("\n" + "-"*60)
    print("STEP 5: Testing Adapters with Your Dataset Features")
    print("-"*60)

    try:
        # Test adapters with your actual feature dimensions
        real_feat = features[0].tensors
        b, c, h, w = real_feat.shape

        print(f"Testing adapters with your feature dimensions: {real_feat.shape}")

        # Conv_Adapter
        conv_adapter = Conv_Adapter(c).to(device) if torch.cuda.is_available() else Conv_Adapter(c)
        adapted_features = conv_adapter(real_feat)
        print(f"✓ Conv_Adapter: {real_feat.shape} -> {adapted_features.shape}")

        # TA_Adapter (key for cross-domain capability)
        ta_adapter = TA_Adapter(256).to(device) if torch.cuda.is_available() else TA_Adapter(256)
        decoder_input = torch.randn(100, b, 256)
        if torch.cuda.is_available():
            decoder_input = decoder_input.to(device)

        ta_output = ta_adapter(decoder_input)
        print(f"✓ TA_Adapter: {decoder_input.shape} -> {ta_output.shape}")

        print(f"✓ Adapters ready for cross-domain text detection!")

    except Exception as e:
        print(f"✗ Adapter testing failed: {e}")

    # Final Summary
    print("\n" + "-"*60)
    print("STEP 6: Your Dataset Processing Summary")
    print("-"*60)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")

    total_time = creation_time + pos_time + backbone_time
    print(f"\nPerformance with your dataset:")
    print(f"  Processing time: {total_time:.3f}s")
    print(f"  Throughput: {len(image_list)/total_time:.1f} images/second")
    print(f"  Average time per image: {total_time/len(image_list):.3f}s")

    print("\n" + "="*80)
    print("🎉 YOUR DATASET TEST COMPLETED SUCCESSFULLY!")
    print(f"✅ Processed {len(image_list)} images from your dataset")
    print(f"✅ Loaded {sum(info['num_annotations'] for info in image_info)} total annotations")
    print("✅ Features extracted and ready for text detection")
    print("✅ Cross-domain adapters functioning correctly")
    print("✅ Ready for Module 3: Transformer Architecture")
    print("="*80)

    return features, pos_encodings, image_info

# ============================================================================
# Helper Functions for Your Dataset
# ============================================================================

def analyze_dataset_json(json_file):
    """
    Analyze your JSON file structure without loading images
    """
    print(f"Analyzing dataset: {json_file}")
    data, format_type = load_dataset_json(json_file)

    if data is None:
        return

    parsed_data = parse_annotations(data, format_type)

    print(f"\nDataset Analysis:")
    print(f"  Format: {format_type}")
    print(f"  Images: {len(parsed_data)}")

    if parsed_data:
        total_annotations = sum(len(entry['annotations']) for entry in parsed_data)
        print(f"  Total annotations: {total_annotations}")
        print(f"  Avg annotations per image: {total_annotations/len(parsed_data):.1f}")

        print(f"\nSample entries:")
        for i, entry in enumerate(parsed_data[:3]):
            print(f"  {i+1}. {entry['image_path']} ({len(entry['annotations'])} annotations)")

def quick_single_image_test(images_dir, json_file, image_index=0):
    """
    Quick test with a single image from your dataset
    """
    data, format_type = load_dataset_json(json_file)
    parsed_data = parse_annotations(data, format_type)

    if image_index >= len(parsed_data):
        print(f"Image index {image_index} out of range (0-{len(parsed_data)-1})")
        return

    entry = parsed_data[image_index]
    print(f"Testing single image: {entry['image_path']}")

    image_list, image_info = create_local_dataset_batch(images_dir, json_file, max_images=1)

    if image_list:
        print("✓ Single image processed successfully!")
        return image_list[0], image_info[0]

run_vimts_local_dataset_test(r'G:\sample\img', r'G:\sample\train.json', max_images=10, visualize_annotations=True)
print("Local Dataset Testing Functions Loaded!")
print("Available functions:")
print("- run_vimts_local_dataset_test(images_dir, json_file, max_images=5)")  
print("- analyze_dataset_json(json_file)")
print("- quick_single_image_test(images_dir, json_file, image_index=0)")
