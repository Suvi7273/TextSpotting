# DATA QUALITY DIAGNOSTIC SCRIPT
# Run this BEFORE training to understand your data

import json
import numpy as np
from collections import Counter

def diagnose_dataset(json_path):
    """
    Comprehensive dataset diagnosis
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("DATASET QUALITY DIAGNOSIS")
    print("="*80)
    
    # Basic stats
    print(f"\n1. BASIC STATISTICS")
    print(f"   Total images: {len(data['images'])}")
    print(f"   Total annotations: {len(data['annotations'])}")
    
    # Annotations per image
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    anns_per_img = [len(anns) for anns in img_to_anns.values()]
    print(f"\n2. ANNOTATIONS PER IMAGE")
    print(f"   Min: {min(anns_per_img)}")
    print(f"   Max: {max(anns_per_img)}")
    print(f"   Mean: {np.mean(anns_per_img):.2f}")
    print(f"   Median: {np.median(anns_per_img):.0f}")
    
    # Recognition quality analysis
    print(f"\n3. RECOGNITION DATA QUALITY")
    total_anns = len(data['annotations'])
    
    empty_rec = 0
    placeholder_only = 0
    has_meaningful = 0
    rec_lengths = []
    
    all_char_ids = Counter()
    meaningful_char_ids = Counter()
    
    for ann in data['annotations']:
        if 'rec' not in ann or not ann['rec']:
            empty_rec += 1
        else:
            rec = ann['rec']
            all_char_ids.update(rec)
            
            # Check for meaningful content
            meaningful_tokens = [x for x in rec if x not in [95, 96]]
            
            if len(meaningful_tokens) == 0:
                placeholder_only += 1
            else:
                has_meaningful += 1
                rec_lengths.append(len(meaningful_tokens))
                meaningful_char_ids.update(meaningful_tokens)
    
    print(f"   Total annotations: {total_anns}")
    print(f"   Empty recognition: {empty_rec} ({100*empty_rec/total_anns:.1f}%)")
    print(f"   Placeholder only (95/96): {placeholder_only} ({100*placeholder_only/total_anns:.1f}%)")
    print(f"   Has meaningful text: {has_meaningful} ({100*has_meaningful/total_anns:.1f}%)")
    
    if rec_lengths:
        print(f"\n   Text length statistics (meaningful tokens):")
        print(f"     Min: {min(rec_lengths)}")
        print(f"     Max: {max(rec_lengths)}")
        print(f"     Mean: {np.mean(rec_lengths):.2f}")
        print(f"     Median: {np.median(rec_lengths):.0f}")
    
    # Character distribution
    print(f"\n4. CHARACTER DISTRIBUTION")
    print(f"   Total unique character IDs: {len(all_char_ids)}")
    print(f"   Meaningful character IDs (excluding 95, 96): {len(meaningful_char_ids)}")
    
    # Most common characters
    if meaningful_char_ids:
        print(f"\n   Top 10 most common character IDs:")
        for char_id, count in meaningful_char_ids.most_common(10):
            char = chr(char_id + 32) if 0 <= char_id <= 94 else f"<{char_id}>"
            print(f"     ID {char_id} ('{char}'): {count} occurrences")
    
    # Box size analysis
    print(f"\n5. BOUNDING BOX STATISTICS")
    box_widths = []
    box_heights = []
    box_areas = []
    
    for ann in data['annotations']:
        bbox = ann['bbox']  # [x, y, w, h]
        box_widths.append(bbox[2])
        box_heights.append(bbox[3])
        box_areas.append(bbox[2] * bbox[3])
    
    print(f"   Width - Min: {min(box_widths):.1f}, Max: {max(box_widths):.1f}, Mean: {np.mean(box_widths):.1f}")
    print(f"   Height - Min: {min(box_heights):.1f}, Max: {max(box_heights):.1f}, Mean: {np.mean(box_heights):.1f}")
    print(f"   Area - Min: {min(box_areas):.1f}, Max: {max(box_areas):.1f}, Mean: {np.mean(box_areas):.1f}")
    
    # Image size analysis
    print(f"\n6. IMAGE SIZE STATISTICS")
    img_widths = [img['width'] for img in data['images']]
    img_heights = [img['height'] for img in data['images']]
    
    print(f"   Width - Min: {min(img_widths)}, Max: {max(img_widths)}, Mean: {np.mean(img_widths):.0f}")
    print(f"   Height - Min: {min(img_heights)}, Max: {max(img_heights)}, Mean: {np.mean(img_heights):.0f}")
    
    # Recommendations
    print(f"\n7. RECOMMENDATIONS")
    print(f"   ✓ = Good, ⚠ = Warning, ✗ = Critical Issue")
    
    if has_meaningful >= total_anns * 0.5:
        print(f"   ✓ Recognition data quality: {has_meaningful} ({100*has_meaningful/total_anns:.1f}%) have text")
    elif has_meaningful >= total_anns * 0.2:
        print(f"   ⚠ Recognition data quality: Only {has_meaningful} ({100*has_meaningful/total_anns:.1f}%) have text")
        print(f"      → Consider training detection first, then add recognition")
    else:
        print(f"   ✗ Recognition data quality: Only {has_meaningful} ({100*has_meaningful/total_anns:.1f}%) have text")
        print(f"      → Focus on detection only: losses = ['labels', 'boxes', 'giou', 'cardinality']")
    
    if len(data['images']) < 50:
        print(f"   ⚠ Dataset size: Only {len(data['images'])} images")
        print(f"      → Expect slow convergence, use strong augmentation")
    else:
        print(f"   ✓ Dataset size: {len(data['images'])} images")
    
    if np.mean(anns_per_img) < 3:
        print(f"   ⚠ Sparsity: Average {np.mean(anns_per_img):.1f} annotations per image")
        print(f"      → Model may struggle to learn, increase NUM_QUERIES")
    else:
        print(f"   ✓ Density: Average {np.mean(anns_per_img):.1f} annotations per image")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'has_meaningful_text': has_meaningful >= total_anns * 0.3,
        'sufficient_data': len(data['images']) >= 50,
        'good_density': np.mean(anns_per_img) >= 3
    }

JSON_PATH = r"G:\mlt2017_sample\train.json"
# Run diagnosis
diagnosis_results = diagnose_dataset(JSON_PATH)

# Adjust training based on diagnosis
print("RECOMMENDED TRAINING CONFIGURATION:")
print("="*80)

if not diagnosis_results['has_meaningful_text']:
    print("⚠ LOW TEXT QUALITY DETECTED")
    print("\nRecommended changes:")
    print("1. Remove 'recognition' from losses_to_compute")
    print("   losses_to_compute = ['labels', 'boxes', 'giou', 'cardinality']")
    print("\n2. Reduce recognition loss weight")
    print("   weight_dict['loss_recognition'] = 0.5  # or remove it")
    print("\n3. Focus on detection performance first")

if not diagnosis_results['sufficient_data']:
    print("\n⚠ SMALL DATASET DETECTED")
    print("\nRecommended changes:")
    print("1. Increase NUM_EPOCHS to 100-200")
    print("2. Use strong data augmentation")
    print("3. Consider using a smaller model (fewer decoder layers)")
    print("4. Use lower learning rate: 5e-5")

if not diagnosis_results['good_density']:
    print("\n⚠ LOW ANNOTATION DENSITY")
    print("\nRecommended changes:")
    print("1. Increase NUM_QUERIES to 150-200")
    print("2. Adjust matcher costs to favor recall over precision")

print("\n" + "="*80)