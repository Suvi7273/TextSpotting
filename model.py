
"""
VimTS Implementation - Module 1: Core Utilities and Dependencies
Optimized for Google Colab T4 GPU with limited memory

This module contains the fundamental components that other VimTS modules depend on.
Save this as 'vimts_module1.py' and upload to your Colab environment.

Usage:
    # In Google Colab:
    from vimts_module1 import *

    # Test the components
    test_module1_components()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
import copy

# Check device and memory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"VimTS Module 1 - Using device: {device}")

# ============================================================================
# 1. NestedTensor Class - Essential for handling batched images with different sizes
# ============================================================================
class NestedTensor(object):
    """
    Tensor container for batched images with different sizes.
    Essential for DETR-based architectures like VimTS.
    """
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    """
    Create a NestedTensor from a list of tensors with potentially different sizes.
    Memory optimized for T4 GPU.
    """
    if tensor_list[0].ndim == 3:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = (len(tensor_list),) + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('Only 3D tensors (C, H, W) are supported')
    return NestedTensor(tensor, mask)

# ============================================================================
# 2. Frozen BatchNorm - Essential for the backbone (from original VimTS)
# ============================================================================
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    This is crucial for maintaining pretrained backbone performance.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # Optimized for memory efficiency
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

# ============================================================================
# 3. Position Encoding - Essential for transformer (from original VimTS)
# ============================================================================
class PositionEmbeddingSine(nn.Module):
    """
    Sine-based position embedding for transformer architectures.
    Adapted from DETR and optimized for VimTS.
    """
    def __init__(self, num_pos_feats=64, temperatureH=10000, temperatureW=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def build_position_encoding(args=None):
    """
    Build position encoding with VimTS configuration.
    T4 GPU optimized parameters.
    """
    if args is None:
        # Default VimTS configuration for T4
        class VimTSConfig:
            position_embedding = 'sine'
            hidden_dim = 256  # Optimized for T4 memory
            pe_temperatureH = 20  # From VimTS config
            pe_temperatureW = 20  # From VimTS config
        args = VimTSConfig()

    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(
            N_steps, 
            temperatureH=args.pe_temperatureH,
            temperatureW=args.pe_temperatureW,
            normalize=True
        )
    else:
        raise ValueError(f"Position embedding {args.position_embedding} not supported")

    return position_embedding

# ============================================================================
# 4. Utility Functions (from original VimTS)
# ============================================================================
def inverse_sigmoid(x, eps=1e-5):
    """Inverse sigmoid function - crucial for DETR-based architectures"""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """Wrapper around F.interpolate for consistency"""
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def clean_state_dict(state_dict):
    """Clean state dict for loading pretrained weights"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

def is_main_process():
    """Check if this is the main process (for distributed training)"""
    return True  # Simplified for single GPU setup

# ============================================================================
# 5. Test Functions
# ============================================================================
def test_module1_components():
    """Test all components in Module 1"""
    print("\n" + "="*60)
    print("Testing VimTS Module 1 Components:")
    print("="*60)

    # Test NestedTensor
    print("1. Testing NestedTensor...")
    try:
        dummy_tensor1 = torch.randn(3, 224, 224)
        dummy_tensor2 = torch.randn(3, 256, 256)
        nested = nested_tensor_from_tensor_list([dummy_tensor1, dummy_tensor2])
        print(f"   ✓ NestedTensor shape: {nested.tensors.shape}")
        print(f"   ✓ Mask shape: {nested.mask.shape}")
    except Exception as e:
        print(f"   ✗ NestedTensor test failed: {e}")

    # Test FrozenBatchNorm2d  
    print("\n2. Testing FrozenBatchNorm2d...")
    try:
        frozen_bn = FrozenBatchNorm2d(256)
        test_input = torch.randn(1, 256, 56, 56)
        frozen_output = frozen_bn(test_input)
        print(f"   ✓ Input shape: {test_input.shape}")
        print(f"   ✓ Output shape: {frozen_output.shape}")
        print(f"   ✓ Parameters frozen: {not any(p.requires_grad for p in frozen_bn.parameters())}")
    except Exception as e:
        print(f"   ✗ FrozenBatchNorm2d test failed: {e}")

    # Test Position Encoding
    print("\n3. Testing PositionEmbeddingSine...")
    try:
        pos_enc = build_position_encoding()
        test_nested = NestedTensor(
            torch.randn(2, 3, 224, 224), 
            torch.zeros(2, 224, 224, dtype=torch.bool)
        )
        pos_encoding = pos_enc(test_nested)
        print(f"   ✓ Input tensor shape: {test_nested.tensors.shape}")
        print(f"   ✓ Position encoding shape: {pos_encoding.shape}")
    except Exception as e:
        print(f"   ✗ Position encoding test failed: {e}")

    # Test utility functions
    print("\n4. Testing utility functions...")
    try:
        test_sigmoid = torch.tensor([0.1, 0.5, 0.9])
        inv_sig = inverse_sigmoid(test_sigmoid)
        recovered = torch.sigmoid(inv_sig)
        print(f"   ✓ Sigmoid: {test_sigmoid}")
        print(f"   ✓ Inverse sigmoid: {inv_sig}")
        print(f"   ✓ Recovery error: {torch.abs(test_sigmoid - recovered).max().item():.6f}")
    except Exception as e:
        print(f"   ✗ Utility functions test failed: {e}")

    print("\n" + "="*60)
    print("Module 1 Testing Complete!")
    print("="*60)

    # Memory usage info
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")

# ============================================================================
# Module Summary
# ============================================================================
print("VimTS Module 1 loaded successfully!")
print("Components available:")
print("- NestedTensor & nested_tensor_from_tensor_list")
print("- FrozenBatchNorm2d")
print("- PositionEmbeddingSine & build_position_encoding")
print("- Utility functions: inverse_sigmoid, interpolate, accuracy, clean_state_dict")
print("- test_module1_components() for testing")
