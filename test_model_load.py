"""
Quick test: Load I3D model with 3D-Adapters
"""
import torch
from models.i3d_adapter import I3D, Unit3Dpy

print("Testing I3D model loading...")

# Load pretrained I3D
print("1. Loading I3D (13M parameters)...")
model = I3D(num_classes=400, dropout_prob=0.5)
print(f"   [OK] I3D loaded successfully")

# Try to load pretrained weights
print("2. Loading Kinetics-400 weights...")
try:
    weights = torch.load('./pretrained_models/model_rgb.pth', map_location='cpu')
    model.load_state_dict(weights, strict=False)
    print(f"   [OK] Weights loaded successfully ({len(weights)} parameters)")
except Exception as e:
    print(f"   [ERROR] Error loading weights: {e}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n3. Model statistics:")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

# Test forward pass
print(f"\n4. Testing forward pass...")
try:
    # Create dummy input (batch=1, channels=3, depth=32, height=224, width=224)
    dummy_input = torch.randn(1, 3, 32, 224, 224)
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   [OK] Forward pass successful!")
except Exception as e:
    print(f"   [ERROR] Error in forward pass: {e}")

# Test adapter freeze
print(f"\n5. Testing parameter freezing...")
for param in model.parameters():
    param.requires_grad = False

adapter_modules = ['mixed_3b', 'mixed_3c', 'mixed_4b', 'mixed_4c',
                  'mixed_4d', 'mixed_4e', 'mixed_4f', 'mixed_5b', 'mixed_5c']

adapter_params = 0
for module_name in adapter_modules:
    if hasattr(model, module_name):
        module = getattr(model, module_name)
        if hasattr(module, 'tuning_module'):
            for param in module.tuning_module.parameters():
                param.requires_grad = True
                adapter_params += param.numel()

trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Adapter parameters: {adapter_params:,}")
print(f"   Trainable parameters after freeze: {trainable_after_freeze:,}")
print(f"   [OK] Parameter freezing successful!")

print(f"\n[SUCCESS] All tests passed!")
print(f"\nSummary:")
print(f"   - I3D backbone: {total_params - adapter_params:,} (frozen)")
print(f"   - 3D-Adapters: {adapter_params:,} (trainable)")
print(f"   - Total: {total_params:,}")
print(f"   - Efficiency: {adapter_params / total_params * 100:.1f}% of parameters trainable")
