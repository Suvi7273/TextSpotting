import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import os

# Import your modules (using the T4-optimized versions)
from vimts_backbone_taqi import VimTSModule1
from vimts_decoder_heads import TaskAwareDecoder, TaskAwareDecoderLayer, PredictionHeads
from text_spotting_dataset import collate_fn, AdaptiveResize, build_vocabulary_from_text_files, TextFileDataset
from matcher import HungarianMatcher
from detr_losses import SetCriterion

# ==============================================================================
# T4-OPTIMIZED CONFIGURATION
# ==============================================================================

# Data paths
GT_DIR = '/content/drive/MyDrive/dataset_ts/mlt_sample/Train_GT'
IMAGE_DIR = '/content/drive/MyDrive/dataset_ts/mlt_sample/TrainImages'

# Build vocabulary
id_to_char, char_to_id, VOCAB_SIZE, PADDING_IDX = build_vocabulary_from_text_files(
    gt_dir=GT_DIR,
    required_language='Latin',
    save_vocab_path='/content/vocabulary.pkl'
)

print(f"Vocabulary size: {VOCAB_SIZE}, Padding index: {PADDING_IDX}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# T4-OPTIMIZED HYPERPARAMETERS
FEATURE_DIM = 128  # Reduced from 256
NUM_QUERIES = 50   # Reduced from 100
NUM_ENCODER_LAYERS = 2  # Reduced from 3
NUM_DECODER_LAYERS = 2  # Reduced from 3
NUM_HEADS = 4  # Reduced from 8
MAX_RECOGNITION_SEQ_LEN = 25
NUM_FOREGROUND_CLASSES = 1
NUM_POLYGON_POINTS = 16

# TRAINING CONFIG
BATCH_SIZE = 1
LEARNING_RATE = 1e-4  # Conservative LR
NUM_EPOCHS = 50
GRADIENT_ACCUMULATION_STEPS = 4  # Simulate larger batch
MAX_GRAD_NORM = 0.5

# Image size limits (CRITICAL for T4)
MAX_IMAGE_SIZE = 800  # Reduced from 1600
MIN_IMAGE_SIZE = 400  # Reduced from 640

print(f"\nT4 Configuration:")
print(f"  Feature Dim: {FEATURE_DIM}")
print(f"  Num Queries: {NUM_QUERIES}")
print(f"  Encoder Layers: {NUM_ENCODER_LAYERS}")
print(f"  Decoder Layers: {NUM_DECODER_LAYERS}")
print(f"  Max Image Size: {MAX_IMAGE_SIZE}")


# ==============================================================================
# FULL MODEL WITH T4 OPTIMIZATIONS
# ==============================================================================

class VimTSFullModel(nn.Module):
    def __init__(self, resnet_pretrained=True,
                 feature_dim=128,
                 transformer_num_heads=4,
                 transformer_num_encoder_layers=2,
                 num_queries=50,
                 transformer_num_decoder_layers=2,
                 num_foreground_classes=1,
                 vocab_size=97,
                 num_polygon_points=16,
                 max_recognition_seq_len=25):
        super().__init__()
        
        self.module1 = VimTSModule1(
            resnet_pretrained=resnet_pretrained,
            feature_dim=feature_dim,
            transformer_num_heads=transformer_num_heads,
            transformer_num_layers=transformer_num_encoder_layers,
            num_queries=num_queries
        )

        decoder_layer = TaskAwareDecoderLayer(
            d_model=feature_dim,
            nhead=transformer_num_heads,
            dim_feedforward=feature_dim * 2  # Reduced FFN
        )
        
        self.decoder = TaskAwareDecoder(
            decoder_layer=decoder_layer,
            num_layers=transformer_num_decoder_layers,
            d_model=feature_dim,
            max_queries=num_queries * 2
        )

        self.prediction_heads = PredictionHeads(
            d_model=feature_dim,
            num_foreground_classes=num_foreground_classes,
            num_chars=vocab_size,
            num_polygon_points=num_polygon_points,
            max_recognition_seq_len=max_recognition_seq_len,
            use_adapter=False  # Disable adapter to save memory
        )
        
        self.num_queries = num_queries

    def forward(self, img):
        module1_output = self.module1(img)
        
        encoded_image_features = module1_output["encoded_image_features"]
        detection_queries = module1_output["detection_queries"]
        recognition_queries = module1_output["recognition_queries"]

        refined_detection_queries, refined_recognition_queries, _ = self.decoder(
            detection_queries, recognition_queries, encoded_image_features
        )

        final_predictions = self.prediction_heads(
            refined_detection_queries, refined_recognition_queries
        )
        
        final_predictions["coarse_bboxes_and_scores"] = module1_output["coarse_bboxes_and_scores"]

        return final_predictions


# ==============================================================================
# DATASET WITH MEMORY-EFFICIENT TRANSFORMS
# ==============================================================================

class MemoryEfficientResize:
    """Aggressive resizing to fit in T4 memory"""
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, img, target):
        w, h = img.size
        scale = self.max_size / max(w, h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Round to multiples of 32 for efficiency
        new_w = (new_w // 32) * 32
        new_h = (new_h // 32) * 32
        
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        if target is not None:
            target["size"] = torch.tensor([new_h, new_w])
            target["scale"] = scale
            return img, target
        return img


from torchvision import transforms
from PIL import Image

transform_train = transforms.Compose([
    MemoryEfficientResize(max_size=MAX_IMAGE_SIZE),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Reduced augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = TextFileDataset(
    gt_dir=GT_DIR,
    img_dir=IMAGE_DIR,
    transform=None,  # We'll handle transforms manually
    max_recognition_seq_len=MAX_RECOGNITION_SEQ_LEN,
    padding_value=PADDING_IDX,
    char_to_id=char_to_id,
    require_language='Latin'
)

# Custom collate with transform
def collate_with_transform(batch):
    images = []
    targets = []
    
    for img, target in batch:
        # Apply transform
        img_tensor = transform_train(img)
        images.append(img_tensor)
        targets.append(target)
    
    # Stack images
    images = torch.stack(images)
    return images, targets

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # Set to 0 to save memory
    collate_fn=collate_fn,
    pin_memory=True
)

# ==============================================================================
# INITIALIZE MODEL, CRITERION, OPTIMIZER
# ==============================================================================

model = VimTSFullModel(
    resnet_pretrained=True,
    feature_dim=FEATURE_DIM,
    transformer_num_heads=NUM_HEADS,
    transformer_num_encoder_layers=NUM_ENCODER_LAYERS,
    num_queries=NUM_QUERIES,
    transformer_num_decoder_layers=NUM_DECODER_LAYERS,
    num_foreground_classes=NUM_FOREGROUND_CLASSES,
    vocab_size=VOCAB_SIZE,
    num_polygon_points=NUM_POLYGON_POINTS,
    max_recognition_seq_len=MAX_RECOGNITION_SEQ_LEN
).to(device)

# Enable gradient checkpointing to save memory
# torch.utils.checkpoint can be used in transformer layers

print(f"\nModel Parameters:")
print(f"  Total: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Initialize matcher and criterion
matcher = HungarianMatcher(
    cost_class=2.0,
    cost_bbox=5.0,
    cost_giou=2.0,
    cost_recognition=0.0,  # Start without recognition
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_RECOGNITION_SEQ_LEN,
    padding_idx=PADDING_IDX
)

weight_dict = {
    'loss_ce': 2.0,
    'loss_bbox': 5.0,
    'loss_giou': 2.0,
    'loss_cardinality': 1.0,
}

losses_to_compute = ['labels', 'boxes', 'cardinality']

criterion = SetCriterion(
    num_foreground_classes=NUM_FOREGROUND_CLASSES,
    matcher=matcher,
    weight_dict=weight_dict,
    eos_coef=0.1,
    losses=losses_to_compute,
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_RECOGNITION_SEQ_LEN,
    padding_idx=PADDING_IDX
).to(device)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# Mixed precision scaler
scaler = GradScaler()

# ==============================================================================
# TRAINING LOOP WITH MEMORY OPTIMIZATIONS
# ==============================================================================

print("\n" + "="*80)
print("Starting T4-Optimized Training")
print("="*80)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_epoch_loss = 0
    optimizer.zero_grad()
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        try:
            # Move to device
            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Mixed precision forward pass
            with autocast():
                predictions = model(images)
                loss_dict = criterion(predictions, targets)
                
                # Compute total loss
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() 
                          if k in weight_dict)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            
            # Update weights every N steps
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # Print progress
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
            
            # Clear cache periodically
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nOOM Error at batch {batch_idx+1}! Skipping batch...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            else:
                raise e
    
    scheduler.step()
    avg_loss = total_epoch_loss / len(dataloader)
    print(f"\n✅ Epoch {epoch+1} finished, Avg Loss: {avg_loss:.4f}")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'/content/checkpoint_epoch_{epoch+1}.pth')
        print(f"Saved checkpoint at epoch {epoch+1}")

print("\n✅ Training finished!")