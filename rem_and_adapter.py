
"""
VimTS Implementation - Module 2: Backbone Architecture and Adapters
Optimized for Google Colab T4 GPU

This module implements the ResNet backbone with FPN and the critical adapter components
that enable VimTS's cross-domain generalization capabilities.

Usage:
    # In Google Colab (after importing Module 1):
    from vimts_module1 import *
    from vimts_module2 import *

    # Test the components
    test_module2_components()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from typing import List, Dict, Optional
from collections import OrderedDict
from torchvision.models._utils import IntermediateLayerGetter

# Assume Module 1 components are available
from util import NestedTensor, FrozenBatchNorm2d, build_position_encoding, nested_tensor_from_tensor_list

print(f"VimTS Module 2 - Backbone and Adapters")

# ============================================================================
# 1. Adapter Components (Critical for VimTS Cross-Domain Generalization)
# ============================================================================

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    This is crucial for stable adapter training.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class Conv_BN_ReLU(nn.Module):
    """Basic convolution block with BatchNorm and ReLU"""
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Adapter(nn.Module):
    """
    Basic adapter module for feature adaptation.
    Key component for cross-domain generalization in VimTS.
    """
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)   

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class T_Adapter(nn.Module):
    """
    Temporal adapter with cross-attention mechanism.
    Critical for video-image adaptation in VimTS.
    """
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = zero_module(nn.Linear(D_features, D_hidden_features))
        self.D_fc3 = zero_module(nn.Linear(D_features, D_hidden_features))
        self.D_fc2 = zero_module(nn.Linear(D_hidden_features, D_features))   

        # Cross-attention components
        self.s1 = zero_module(nn.MultiheadAttention(D_hidden_features, 8, dropout=0.1))
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = zero_module(nn.LayerNorm(D_hidden_features))
        self.s2 = zero_module(nn.MultiheadAttention(D_hidden_features, 8, dropout=0.1))
        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = zero_module(nn.LayerNorm(D_hidden_features))
        self.ad = zero_module(Adapter(D_hidden_features, mlp_ratio=2))

    def forward(self, x, t):
        # x is source features, t is target features
        xs = self.D_fc1(x)
        ts = self.D_fc3(t)

        # Cross-attention between source and target
        tgt = self.s1(ts.flatten(1,2).transpose(0,1),   
                     xs.flatten(1,2).transpose(0,1),   
                     xs.flatten(1,2).transpose(0,1))[0].transpose(0, 1).reshape(ts.shape)
        tgt = ts + self.dropout1(tgt)
        tgt = self.norm1(tgt)

        # Self-attention within target
        tgt_inter = torch.swapdims(tgt, 0, 2)
        tgt_inter2 = self.s2(tgt_inter.flatten(0, 1).transpose(0, 1),   
                            tgt_inter.flatten(0, 1).transpose(0, 1),   
                            tgt_inter.flatten(0, 1).transpose(0, 1))[0].transpose(0, 1).reshape(tgt_inter.shape)
        tgt_inter = tgt_inter + self.dropout2(tgt_inter2)
        tgt = torch.swapdims(self.norm2(tgt_inter), 0, 2)

        # Final adaptation
        tgt = self.ad(tgt)
        tgt = self.act(tgt)
        tgt = self.D_fc2(tgt)

        if self.skip_connect:
            x = t + tgt
        else:
            x = tgt
        return x

class TA_Adapter(nn.Module):
    """
    Temporal-Attention Adapter - combines multiple T_Adapters.
    This is the main adapter used in VimTS decoder layers.
    """
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.ad2 = T_Adapter(D_features, mlp_ratio=0.25)
        self.ad3 = T_Adapter(D_features, mlp_ratio=0.25)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt):   
        # tgt: nq, bs, d_model
        x1 = self.ad2(tgt[:,:,0:1], tgt)
        x2 = self.ad3(tgt[:,:,1:], x1)
        return x2

class Conv_Adapter(nn.Module):
    """Convolutional adapter for feature map adaptation"""
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Conv2d(D_features, D_hidden_features, 1, 1)
        self.D_fc2 = nn.Conv2d(D_hidden_features, D_features, 1, 1)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            return x + xs
        return xs

class FPEM_v2(nn.Module):
    """
    Feature Pyramid Enhancement Module v2.
    Processes multi-scale features with adapters.
    """
    def __init__(self, in_channels, out_channels):
        super(FPEM_v2, self).__init__()
        planes = out_channels
        self.linear = nn.ModuleList([zero_module(nn.Conv2d(planes, planes, kernel_size=1)) for i in range(4)])
        self.adapter = nn.ModuleList([Conv_Adapter(planes, skip_connect=False) for i in range(4)])

    def forward(self, f1, f2, f3, f4):
        f1 = self.adapter[0](f1)
        f2 = self.adapter[1](f2)
        f3 = self.adapter[2](f3)
        f4 = self.adapter[3](f4)
        return [self.linear[0](f1), self.linear[1](f2), self.linear[2](f3), self.linear[3](f4)]

# ============================================================================
# 2. Backbone Architecture (ResNet with FPN)
# ============================================================================

class BackboneBase(nn.Module):
    """Base class for backbone networks"""
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_indices: list):
        super().__init__()

        # Freeze backbone parameters except specified layers
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        # Setup return layers based on indices
        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update({"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)})

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        out = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm - optimized for T4"""
    def __init__(self, name: str, train_backbone: bool, dilation: bool, 
                 return_interm_indices: list, batch_norm=None):

        if batch_norm is None:
            batch_norm = FrozenBatchNorm2d

        if name in ['resnet50', 'resnet101']:
            # Load pretrained ResNet but be memory conscious
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=True,  # Always use pretrained for better performance
                norm_layer=batch_norm
            )
        else:
            raise NotImplementedError(f"Backbone {name} not supported. Use resnet50 or resnet101.")

        # Only support ResNet50/101 (ResNet18/34 have different channel counts)
        assert name not in ('resnet18', 'resnet34'), "Only resnet50 and resnet101 are supported."
        assert return_interm_indices in [[0,1,2,3], [1,2,3], [3]], "Invalid return_interm_indices"

        # Channel counts for ResNet50/101: [256, 512, 1024, 2048]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4-len(return_interm_indices):]

        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)

class Joiner(nn.Sequential):
    """Joins backbone with position encoding"""
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        xs = self[0](tensor_list)  # backbone
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos

def build_backbone(args=None):
    """
    Build backbone for VimTS - T4 optimized version
    """
    if args is None:
        # Default VimTS configuration for T4
        class VimTSBackboneConfig:
            backbone = 'resnet50'  # Use ResNet50 for T4 (less memory than ResNet101)
            lr_backbone = 1e-5     # From VimTS config
            dilation = False
            return_interm_indices = [1, 2, 3]  # Skip layer0 to save memory
            backbone_freeze_keywords = None
            use_checkpoint = False
            position_embedding = 'sine'
            hidden_dim = 256
            pe_temperatureH = 20
            pe_temperatureW = 20
        args = VimTSBackboneConfig()

    # Build position embedding
    position_embedding = build_position_encoding(args)

    # Check if we should train backbone
    train_backbone = args.lr_backbone > 0
    if not train_backbone:
        print("Warning: lr_backbone <= 0, backbone will be frozen")

    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0,1,2,3], [1,2,3], [3]], "Invalid return_interm_indices"

    # Build backbone
    if args.backbone in ['resnet50', 'resnet101']:
        backbone = Backbone(
            args.backbone, 
            train_backbone, 
            args.dilation,   
            return_interm_indices,   
            batch_norm=FrozenBatchNorm2d
        )
        bb_num_channels = backbone.num_channels
    else:
        raise NotImplementedError(f"Unknown backbone {args.backbone}")

    # Verify channel counts match indices
    assert len(bb_num_channels) == len(return_interm_indices),         f"Channel count mismatch: {len(bb_num_channels)} != {len(return_interm_indices)}"

    # Create final model
    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels   

    return model

# ============================================================================
# 3. Test Functions
# ============================================================================

def test_adapter_components():
    """Test all adapter components"""
    print("Testing Adapter Components:")
    print("-" * 40)

    # Test basic Adapter
    try:
        adapter = Adapter(256)
        test_input = torch.randn(2, 100, 256)  # (batch, seq_len, features)
        output = adapter(test_input)
        print(f"✓ Adapter: {test_input.shape} -> {output.shape}")
    except Exception as e:
        print(f"✗ Adapter failed: {e}")

    # Test T_Adapter
    try:
        t_adapter = T_Adapter(256)
        x = torch.randn(2, 100, 256)
        t = torch.randn(2, 100, 256)
        output = t_adapter(x, t)
        print(f"✓ T_Adapter: {x.shape}, {t.shape} -> {output.shape}")
    except Exception as e:
        print(f"✗ T_Adapter failed: {e}")

    # Test TA_Adapter
    try:
        ta_adapter = TA_Adapter(256)
        test_input = torch.randn(100, 2, 256)  # (nq, bs, d_model)
        output = ta_adapter(test_input)
        print(f"✓ TA_Adapter: {test_input.shape} -> {output.shape}")
    except Exception as e:
        print(f"✗ TA_Adapter failed: {e}")

    # Test Conv_Adapter
    try:
        conv_adapter = Conv_Adapter(256)
        test_input = torch.randn(2, 256, 32, 32)  # (batch, channels, H, W)
        output = conv_adapter(test_input)
        print(f"✓ Conv_Adapter: {test_input.shape} -> {output.shape}")
    except Exception as e:
        print(f"✗ Conv_Adapter failed: {e}")

def test_backbone_components():
    """Test backbone components"""
    print("\nTesting Backbone Components:")
    print("-" * 40)

    # Test Backbone building
    try:
        backbone_model = build_backbone()
        print(f"✓ Backbone built successfully")
        print(f"  Backbone type: {type(backbone_model).__name__}")
        print(f"  Channel counts: {backbone_model.num_channels}")
    except Exception as e:
        print(f"✗ Backbone building failed: {e}")
        return None

    # Test backbone forward pass with dummy data
    try:
        # Create dummy input
        dummy_img1 = torch.randn(3, 224, 224)
        dummy_img2 = torch.randn(3, 256, 256)  
        nested_input = nested_tensor_from_tensor_list([dummy_img1, dummy_img2])

        # Forward pass
        features, pos_encodings = backbone_model(nested_input)

        print(f"✓ Backbone forward pass successful")
        print(f"  Number of feature levels: {len(features)}")
        for i, (feat, pos) in enumerate(zip(features, pos_encodings)):
            print(f"  Level {i}: Feature {feat.tensors.shape}, Pos {pos.shape}")

    except Exception as e:
        print(f"✗ Backbone forward pass failed: {e}")

    return backbone_model

def test_module2_components():
    """Test all components in Module 2"""
    print("\n" + "="*60)
    print("Testing VimTS Module 2 Components:")
    print("="*60)

    # Test adapters
    test_adapter_components()

    # Test backbone
    backbone_model = test_backbone_components()

    print("\n" + "="*60)
    print("Module 2 Testing Complete!")

    # Memory usage info
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")

    print("="*60)

    return backbone_model

# ============================================================================
# Module Summary
# ============================================================================
print("VimTS Module 2 loaded successfully!")
print("Components available:")
print("- Adapter classes: Adapter, T_Adapter, TA_Adapter, Conv_Adapter")
print("- FPEM_v2 for multi-scale feature processing")
print("- Backbone classes: BackboneBase, Backbone, Joiner")
print("- build_backbone() function")
print("- test_module2_components() for comprehensive testing")
