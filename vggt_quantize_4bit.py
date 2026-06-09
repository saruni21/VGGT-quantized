#!/usr/bin/env python3
"""
VGGT 4-bit Quantization Script for Mac (Apple Silicon / MPS)
============================================================

This script quantizes the VGGT-1B model to 4-bit precision so it can run
on consumer Mac hardware with limited unified memory (8-32GB RAM).

Key approach:
- Since bitsandbytes is CUDA-only and doesn't support Mac/MPS, we use
  PyTorch's built-in quantization (dynamic quantization + custom weight packing)
- We quantize Linear layers to 4-bit using a custom quantization wrapper
- We keep LayerNorm and other sensitive layers in higher precision
- The script supports both CPU and MPS (Apple Silicon) backends

Requirements:
    pip install torch torchvision numpy Pillow huggingface_hub

Usage:
    python vggt_quantize_4bit.py --images /path/to/images/ --output /path/to/output/

"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
import numpy as np
from PIL import Image

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 4-bit Quantization Utilities (Mac-compatible, no bitsandbytes dependency)
# =============================================================================

class QuantizedLinear4Bit(nn.Module):
    """
    Custom 4-bit quantized linear layer for Mac/CPU inference.

    Stores weights in 4-bit packed format (2 values per uint8).
    Dequantizes on-the-fly during forward pass.

    Memory reduction: ~8x compared to fp32, ~4x compared to fp16/bf16
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 64,
        compute_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.compute_dtype = compute_dtype

        # Calculate number of blocks
        num_blocks_in = (in_features + block_size - 1) // block_size
        num_blocks_out = (out_features + block_size - 1) // block_size

        # 4-bit packed weights: 2 4-bit values per uint8
        # Shape: [out_features, in_features // 2] (packed)
        packed_in_features = (in_features + 1) // 2
        self.register_buffer(
            "weight_packed",
            torch.zeros((out_features, packed_in_features), dtype=torch.uint8)
        )

        # Scaling factors per block: [num_blocks_out, num_blocks_in]
        self.register_buffer(
            "scales",
            torch.ones((num_blocks_out, num_blocks_in), dtype=compute_dtype)
        )

        # Zero points per block (optional, for asymmetric quantization)
        self.register_buffer(
            "zeros",
            torch.zeros((num_blocks_out, num_blocks_in), dtype=compute_dtype)
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def pack_weights(self, weight_fp: torch.Tensor):
        """
        Pack fp32/fp16 weights into 4-bit format.

        Args:
            weight_fp: [out_features, in_features] floating point weights
        """
        weight_fp = weight_fp.to(torch.float32)
        out_features, in_features = weight_fp.shape

        # Pad to block size multiples
        pad_in = (self.block_size - in_features % self.block_size) % self.block_size
        pad_out = (self.block_size - out_features % self.block_size) % self.block_size

        if pad_in > 0 or pad_out > 0:
            weight_fp = F.pad(weight_fp, (0, pad_in, 0, pad_out))

        padded_out, padded_in = weight_fp.shape
        num_blocks_in = padded_in // self.block_size
        num_blocks_out = padded_out // self.block_size

        # Reshape for block-wise quantization
        # [num_blocks_out, block_size, num_blocks_in, block_size]
        weight_blocks = weight_fp.reshape(
            num_blocks_out, self.block_size, num_blocks_in, self.block_size
        )
        weight_blocks = weight_blocks.permute(0, 2, 1, 3)  # [out_blocks, in_blocks, block_size, block_size]

        # Compute per-block min/max for asymmetric quantization
        w_min = weight_blocks.amin(dim=(2, 3), keepdim=True)  # [out_blocks, in_blocks, 1, 1]
        w_max = weight_blocks.amax(dim=(2, 3), keepdim=True)

        # 4-bit range: 0-15
        scales = (w_max - w_min) / 15.0
        scales = scales.squeeze(-1).squeeze(-1)  # [out_blocks, in_blocks]

        # Avoid division by zero
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)

        zeros = w_min.squeeze(-1).squeeze(-1)  # [out_blocks, in_blocks]

        # Quantize to 4-bit integers (0-15)
        weight_int = torch.round((weight_blocks - w_min) / scales.unsqueeze(-1).unsqueeze(-1)).to(torch.int32)
        weight_int = torch.clamp(weight_int, 0, 15)

        # Pack two 4-bit values into one uint8
        # Reshape to [out_features, in_features // 2, 2]
        weight_int = weight_int.permute(0, 2, 1, 3).reshape(padded_out, padded_in)

        # Pack: even columns in lower 4 bits, odd columns in upper 4 bits
        if padded_in % 2 == 1:
            weight_int = F.pad(weight_int, (0, 1))
            padded_in += 1

        weight_even = weight_int[:, 0::2]  # Lower 4 bits
        weight_odd = weight_int[:, 1::2]   # Upper 4 bits
        weight_packed = (weight_odd << 4) | weight_even
        weight_packed = weight_packed.to(torch.uint8)

        # Store
        self.weight_packed[:out_features, :(in_features + 1) // 2] = weight_packed[:out_features, :(in_features + 1) // 2]
        self.scales[:num_blocks_out, :num_blocks_in] = scales[:num_blocks_out, :num_blocks_in]
        self.zeros[:num_blocks_out, :num_blocks_in] = zeros[:num_blocks_out, :num_blocks_in]

    def unpack_weights(self) -> torch.Tensor:
        """
        Unpack 4-bit weights back to floating point for computation.

        Returns:
            weight_fp: [out_features, in_features] dequantized weights
        """
        out_features, packed_in = self.weight_packed.shape
        in_features = packed_in * 2

        # Unpack uint8 to two int32 values
        weight_even = self.weight_packed & 0x0F  # Lower 4 bits
        weight_odd = (self.weight_packed >> 4) & 0x0F  # Upper 4 bits

        # Interleave
        weight_int = torch.zeros((out_features, in_features), dtype=torch.int32, device=self.weight_packed.device)
        weight_int[:, 0::2] = weight_even.to(torch.int32)
        weight_int[:, 1::2] = weight_odd.to(torch.int32)

        # Calculate number of blocks
        num_blocks_in = in_features // self.block_size
        num_blocks_out = out_features // self.block_size

        # Reshape for block-wise dequantization
        weight_blocks = weight_int.reshape(
            num_blocks_out, self.block_size, num_blocks_in, self.block_size
        )
        weight_blocks = weight_blocks.permute(0, 2, 1, 3)  # [out_blocks, in_blocks, block_size, block_size]

        # Dequantize
        scales = self.scales[:num_blocks_out, :num_blocks_in].unsqueeze(-1).unsqueeze(-1)
        zeros = self.zeros[:num_blocks_out, :num_blocks_in].unsqueeze(-1).unsqueeze(-1)

        weight_fp = weight_blocks.to(self.compute_dtype) * scales + zeros

        # Reshape back
        weight_fp = weight_fp.permute(0, 2, 1, 3).reshape(out_features, in_features)

        return weight_fp.to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        x = x.to(self.compute_dtype)
        weight = self.unpack_weights()

        # Trim to actual dimensions (remove padding)
        weight = weight[:self.out_features, :self.in_features]

        output = F.linear(x, weight, self.bias)
        return output


class QuantizedEmbedding4Bit(nn.Module):
    """4-bit quantized embedding layer."""

    def __init__(self, num_embeddings: int, embedding_dim: int, block_size: int = 64):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.block_size = block_size

        packed_dim = (embedding_dim + 1) // 2
        self.register_buffer("weight_packed", torch.zeros((num_embeddings, packed_dim), dtype=torch.uint8))

        num_blocks = (embedding_dim + block_size - 1) // block_size
        self.register_buffer("scales", torch.ones((num_embeddings, num_blocks), dtype=torch.float32))
        self.register_buffer("zeros", torch.zeros((num_embeddings, num_blocks), dtype=torch.float32))

    def pack_weights(self, weight_fp: torch.Tensor):
        """Pack fp weights into 4-bit."""
        weight_fp = weight_fp.to(torch.float32)
        num_embeddings, embedding_dim = weight_fp.shape

        pad_dim = (self.block_size - embedding_dim % self.block_size) % self.block_size
        if pad_dim > 0:
            weight_fp = F.pad(weight_fp, (0, pad_dim))

        padded_dim = weight_fp.shape[1]
        num_blocks = padded_dim // self.block_size

        weight_blocks = weight_fp.reshape(num_embeddings, num_blocks, self.block_size)
        w_min = weight_blocks.amin(dim=2, keepdim=True)
        w_max = weight_blocks.amax(dim=2, keepdim=True)

        scales = ((w_max - w_min) / 15.0).squeeze(-1)
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        zeros = w_min.squeeze(-1)

        weight_int = torch.round((weight_blocks - w_min) / scales.unsqueeze(-1)).to(torch.int32)
        weight_int = torch.clamp(weight_int, 0, 15).reshape(num_embeddings, padded_dim)

        if padded_dim % 2 == 1:
            weight_int = F.pad(weight_int, (0, 1))
            padded_dim += 1

        weight_even = weight_int[:, 0::2]
        weight_odd = weight_int[:, 1::2]
        weight_packed = (weight_odd << 4) | weight_even

        self.weight_packed[:num_embeddings, :(embedding_dim + 1) // 2] = weight_packed[:num_embeddings, :(embedding_dim + 1) // 2]
        self.scales[:num_embeddings, :num_blocks] = scales[:num_embeddings, :num_blocks]
        self.zeros[:num_embeddings, :num_blocks] = zeros[:num_embeddings, :num_blocks]

    def unpack_weights(self) -> torch.Tensor:
        """Unpack 4-bit weights."""
        num_embeddings, packed_dim = self.weight_packed.shape
        embedding_dim = packed_dim * 2

        weight_even = self.weight_packed & 0x0F
        weight_odd = (self.weight_packed >> 4) & 0x0F

        weight_int = torch.zeros((num_embeddings, embedding_dim), dtype=torch.int32, device=self.weight_packed.device)
        weight_int[:, 0::2] = weight_even.to(torch.int32)
        weight_int[:, 1::2] = weight_odd.to(torch.int32)

        num_blocks = embedding_dim // self.block_size
        weight_blocks = weight_int.reshape(num_embeddings, num_blocks, self.block_size)

        scales = self.scales[:num_embeddings, :num_blocks].unsqueeze(-1)
        zeros = self.zeros[:num_embeddings, :num_blocks].unsqueeze(-1)

        weight_fp = weight_blocks.to(torch.float32) * scales + zeros
        weight_fp = weight_fp.reshape(num_embeddings, embedding_dim)

        return weight_fp[:self.num_embeddings, :self.embedding_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.unpack_weights()
        return F.embedding(x, weight)


# =============================================================================
# Model Quantization Functions
# =============================================================================

def quantize_model_4bit(model: nn.Module, compute_dtype: torch.dtype = torch.float32) -> nn.Module:
    """
    Recursively replace Linear and Embedding layers with 4-bit quantized versions.

    Keeps LayerNorm, activation functions, and other sensitive layers in higher precision.

    Args:
        model: PyTorch model to quantize
        compute_dtype: dtype for computation (float32 for CPU, float16 for MPS if supported)

    Returns:
        Quantized model (modified in-place)
    """

    def replace_module(parent_module: nn.Module, child_name: str, child_module: nn.Module):
        """Replace a child module with its quantized version."""

        if isinstance(child_module, nn.Linear):
            # Skip very small layers (not worth quantization overhead)
            if child_module.in_features < 32 or child_module.out_features < 32:
                return

            quantized = QuantizedLinear4Bit(
                in_features=child_module.in_features,
                out_features=child_module.out_features,
                bias=child_module.bias is not None,
                block_size=64,
                compute_dtype=compute_dtype,
            )

            # Copy and quantize weights
            with torch.no_grad():
                quantized.pack_weights(child_module.weight.data)
                if child_module.bias is not None:
                    quantized.bias.data = child_module.bias.data.to(compute_dtype)

            setattr(parent_module, child_name, quantized)
            print(f"  Quantized Linear: {child_name} [{child_module.in_features}x{child_module.out_features}]")

        elif isinstance(child_module, nn.Embedding):
            if child_module.num_embeddings < 32 or child_module.embedding_dim < 32:
                return

            quantized = QuantizedEmbedding4Bit(
                num_embeddings=child_module.num_embeddings,
                embedding_dim=child_module.embedding_dim,
                block_size=64,
            )

            with torch.no_grad():
                quantized.pack_weights(child_module.weight.data)

            setattr(parent_module, child_name, quantized)
            print(f"  Quantized Embedding: {child_name} [{child_module.num_embeddings}x{child_module.embedding_dim}]")

    def recurse_quantize(module: nn.Module, prefix: str = ""):
        """Recursively traverse and quantize modules."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Skip certain sensitive modules
            if any(skip in full_name.lower() for skip in [
                "norm", "ln", "layernorm", "head", "camera_head", 
                "depth_head", "point_head", "track_head"
            ]):
                print(f"  Skipped (sensitive): {full_name}")
                continue

            # Recurse into container modules
            if len(list(child.children())) > 0:
                recurse_quantize(child, full_name)

            # Quantize leaf modules
            replace_module(module, name, child)

    print("\n" + "="*60)
    print("Starting 4-bit quantization...")
    print("="*60)

    recurse_quantize(model)

    print("="*60)
    print("Quantization complete!")
    print("="*60 + "\n")

    return model


def estimate_memory_footprint(model: nn.Module) -> Dict[str, float]:
    """
    Estimate memory footprint of model in MB.

    Returns:
        Dict with total, parameters, buffers sizes
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        if param is not None:
            param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        if buffer is not None:
            buffer_size += buffer.numel() * buffer.element_size()

    total_size = param_size + buffer_size

    return {
        "total_mb": total_size / (1024 ** 2),
        "parameters_mb": param_size / (1024 ** 2),
        "buffers_mb": buffer_size / (1024 ** 2),
    }


# =============================================================================
# VGGT-specific Utilities
# =============================================================================

def load_vggt_model(device: str = "cpu", quantized: bool = False, compute_dtype: torch.dtype = torch.float32):
    """
    Load VGGT model with optional 4-bit quantization.

    Args:
        device: 'cpu', 'mps', or 'cuda'
        quantized: whether to apply 4-bit quantization
        compute_dtype: computation dtype (float32 for CPU, float16 for MPS)

    Returns:
        Loaded model
    """
    try:
        from vggt.models.vggt import VGGT
    except ImportError:
        print("ERROR: VGGT not installed. Please run:")
        print("  git clone https://github.com/facebookresearch/vggt.git")
        print("  cd vggt && pip install -e .")
        sys.exit(1)

    print(f"Loading VGGT-1B model (device={device}, quantized={quantized})...")

    # Load model
    model = VGGT.from_pretrained("facebook/VGGT-1B")

    # Estimate memory before quantization
    mem_before = estimate_memory_footprint(model)
    print(f"\nMemory before quantization:")
    print(f"  Total: {mem_before['total_mb']:.1f} MB")
    print(f"  Parameters: {mem_before['parameters_mb']:.1f} MB")

    if quantized:
        model = quantize_model_4bit(model, compute_dtype=compute_dtype)

        mem_after = estimate_memory_footprint(model)
        print(f"\nMemory after quantization:")
        print(f"  Total: {mem_after['total_mb']:.1f} MB")
        print(f"  Parameters: {mem_after['parameters_mb']:.1f} MB")
        reduction = (1 - mem_after['total_mb'] / mem_before['total_mb']) * 100
        print(f"  Memory reduction: {reduction:.1f}%")

    model = model.to(device)
    model.eval()

    return model


def load_and_preprocess_images(image_paths: List[str], size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Load and preprocess images for VGGT inference.

    Args:
        image_paths: List of image file paths
        size: Optional (H, W) to resize to

    Returns:
        Tensor of shape [N, 3, H, W]
    """
    try:
        from vggt.utils.load_fn import load_and_preprocess_images as vggt_load
        return vggt_load(image_paths)
    except ImportError:
        # Fallback implementation
        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            if size:
                img = img.resize(size[::-1])  # PIL uses (W, H)
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]
            images.append(img_tensor)

        return torch.stack(images)  # [N, 3, H, W]


# =============================================================================
# Inference with Quantized Model
# =============================================================================

def run_inference(
    model: nn.Module,
    images: torch.Tensor,
    device: str,
    compute_dtype: torch.dtype = torch.float32,
    predict_cameras: bool = True,
    predict_depth: bool = True,
    predict_points: bool = True,
    predict_tracks: bool = False,
) -> Dict:
    """
    Run VGGT inference with quantized model.

    Args:
        model: Quantized VGGT model
        images: Preprocessed images [N, 3, H, W]
        device: Device string
        compute_dtype: Computation dtype
        predict_cameras: Whether to predict camera parameters
        predict_depth: Whether to predict depth maps
        predict_points: Whether to predict point maps
        predict_tracks: Whether to predict point tracks

    Returns:
        Dictionary of predictions
    """
    images = images.to(device)

    with torch.no_grad():
        # For MPS, we can't use autocast the same way as CUDA
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=compute_dtype):
                predictions = model(images)
        else:
            # CPU/MPS path
            if compute_dtype == torch.float16 and device == "mps":
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    predictions = model(images)
            else:
                predictions = model(images)

    return predictions


def save_quantized_model(model: nn.Module, save_path: str):
    """
    Save quantized model state dict.

    Args:
        model: Quantized model
        save_path: Path to save .pt file
    """
    print(f"\nSaving quantized model to {save_path}...")
    torch.save(model.state_dict(), save_path)

    # Save metadata
    metadata = {
        "quantized": True,
        "quantization_type": "4-bit_custom",
        "block_size": 64,
        "format_version": "1.0",
    }

    meta_path = save_path.replace(".pt", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved: {save_path}")
    print(f"Metadata saved: {meta_path}")

    # Report file size
    size_mb = os.path.getsize(save_path) / (1024 ** 2)
    print(f"File size: {size_mb:.1f} MB")


def load_quantized_model(model: nn.Module, checkpoint_path: str):
    """
    Load quantized model weights from checkpoint.

    Args:
        model: Model architecture (already quantized)
        checkpoint_path: Path to .pt file

    Returns:
        Model with loaded weights
    """
    print(f"Loading quantized model from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
    return model


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VGGT 4-bit Quantization for Mac/Consumer Hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize and save model
  python vggt_quantize_4bit.py --quantize --save-model ./vggt_4bit.pt

  # Run inference with quantized model
  python vggt_quantize_4bit.py --load-model ./vggt_4bit.pt --images ./my_images/ --output ./results/

  # Run on Mac with MPS (Apple Silicon)
  python vggt_quantize_4bit.py --device mps --images ./my_images/

  # Run on CPU with 4-bit quantization
  python vggt_quantize_4bit.py --device cpu --quantize --images ./my_images/
        """
    )

    parser.add_argument("--images", type=str, help="Path to image folder or video file")
    parser.add_argument("--output", type=str, default="./vggt_output", help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto", 
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Device to use (auto detects best available)")
    parser.add_argument("--quantize", action="store_true", 
                        help="Apply 4-bit quantization to model weights")
    parser.add_argument("--save-model", type=str, help="Save quantized model to path")
    parser.add_argument("--load-model", type=str, help="Load quantized model from path")
    parser.add_argument("--compute-dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Computation dtype (float32 recommended for CPU, float16 for MPS)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to process (for memory limits)")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Resize images to this size (default 512, smaller = less memory)")
    parser.add_argument("--no-cameras", action="store_true", help="Skip camera prediction")
    parser.add_argument("--no-depth", action="store_true", help="Skip depth prediction")
    parser.add_argument("--no-points", action="store_true", help="Skip point map prediction")
    parser.add_argument("--tracks", action="store_true", help="Enable point tracking")

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
            print("Auto-detected MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            args.device = "cuda"
            print("Auto-detected CUDA")
        else:
            args.device = "cpu"
            print("Auto-detected CPU")

    # Set compute dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32,
    }
    compute_dtype = dtype_map[args.compute_dtype]

    # Warn about MPS + float32 memory usage
    if args.device == "mps" and compute_dtype == torch.float32:
        print("\nWARNING: MPS with float32 uses more memory. Consider --compute-dtype float16")

    # Load or quantize model
    if args.load_model:
        # Load pre-quantized model
        try:
            from vggt.models.vggt import VGGT
        except ImportError:
            print("ERROR: VGGT not installed. Please install it first.")
            sys.exit(1)

        model = VGGT.from_pretrained("facebook/VGGT-1B")
        model = quantize_model_4bit(model, compute_dtype=compute_dtype)
        model = load_quantized_model(model, args.load_model)
        model = model.to(args.device)
        model.eval()
    else:
        model = load_vggt_model(
            device=args.device,
            quantized=args.quantize,
            compute_dtype=compute_dtype,
        )

    # Save quantized model if requested
    if args.quantize and args.save_model:
        save_quantized_model(model, args.save_model)

    # Run inference if images provided
    if args.images:
        image_path = Path(args.images)

        if image_path.is_dir():
            image_paths = sorted([
                str(p) for p in image_path.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ])
        elif image_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            # Extract frames from video
            print(f"Extracting frames from video: {image_path}")
            import cv2
            cap = cv2.VideoCapture(str(image_path))
            image_paths = []
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % 5 == 0:  # Extract every 5th frame
                    frame_path = f"/tmp/vggt_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(frame_path, frame)
                    image_paths.append(frame_path)
                frame_count += 1
            cap.release()
            print(f"Extracted {len(image_paths)} frames")
        else:
            image_paths = [str(image_path)]

        if args.max_images:
            image_paths = image_paths[:args.max_images]

        if not image_paths:
            print("No images found!")
            sys.exit(1)

        print(f"\nProcessing {len(image_paths)} images...")

        # Load images
        images = load_and_preprocess_images(image_paths)

        # Resize if needed
        if args.image_size != 512:
            images = F.interpolate(
                images, 
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=False,
            )

        print(f"Image tensor shape: {images.shape}")
        print(f"Image tensor dtype: {images.dtype}")
        print(f"Device: {args.device}")

        # Run inference
        print("\nRunning inference...")
        try:
            predictions = run_inference(
                model=model,
                images=images,
                device=args.device,
                compute_dtype=compute_dtype,
                predict_cameras=not args.no_cameras,
                predict_depth=not args.no_depth,
                predict_points=not args.no_points,
                predict_tracks=args.tracks,
            )

            print("\nInference complete!")
            print(f"Predictions: {list(predictions.keys())}")

            # Save results
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save ALL predictions for viser compatibility
            # Save images tensor for coloring
            torch.save(images.cpu(), output_dir / "images.pt")
            print(f"Saved images tensor: {images.shape}")

            # Save camera parameters (pose_enc -> extrinsic/intrinsic)
            if "pose_enc" in predictions:
                pose_enc = predictions["pose_enc"]
                print(f"Pose encoding shape: {pose_enc.shape if hasattr(pose_enc, 'shape') else 'N/A'}")
                torch.save(pose_enc, output_dir / "pose_enc.pt")

                # Also decode and save extrinsic/intrinsic for viser
                try:
                    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                    torch.save(extrinsic.cpu(), output_dir / "extrinsic.pt")
                    torch.save(intrinsic.cpu(), output_dir / "intrinsic.pt")
                    print(f"Saved extrinsic: {extrinsic.shape}, intrinsic: {intrinsic.shape}")
                except Exception as e:
                    print(f"Could not decode camera matrices: {e}")

            # Save depth maps
            if "depth" in predictions and not args.no_depth:
                depth = predictions["depth"]
                print(f"Depth maps shape: {depth.shape if hasattr(depth, 'shape') else 'N/A'}")
                torch.save(depth.cpu(), output_dir / "depth.pt")

                # Save depth confidence if available
                if "depth_conf" in predictions:
                    torch.save(predictions["depth_conf"].cpu(), output_dir / "depth_conf.pt")

                # Save as images for visualization
                depth_np = depth.cpu().numpy()
                while depth_np.ndim > 3:
                    if depth_np.shape[0] == 1:
                        depth_np = depth_np[0]
                    else:
                        break
                if depth_np.ndim == 4 and depth_np.shape[-1] == 1:
                    depth_np = depth_np.squeeze(-1)
                if depth_np.ndim == 3 and depth_np.shape[0] == 1:
                    depth_np = depth_np.squeeze(0)
                if depth_np.ndim == 4 and depth_np.shape[1] == 1:
                    depth_np = depth_np.squeeze(1)
                if depth_np.ndim == 2:
                    depth_np = depth_np[None, ...]

                num_depth = depth_np.shape[0]
                for i in range(num_depth):
                    d = depth_np[i]
                    d_min, d_max = d.min(), d.max()
                    if d_max > d_min:
                        d_norm = ((d - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
                    else:
                        d_norm = np.zeros_like(d, dtype=np.uint8)
                    Image.fromarray(d_norm).save(output_dir / f"depth_{i:04d}.png")

            # Save point maps (world_points)
            if "world_points" in predictions and not args.no_points:
                world_points = predictions["world_points"]
                print(f"World points shape: {world_points.shape if hasattr(world_points, 'shape') else 'N/A'}")
                torch.save(world_points.cpu(), output_dir / "world_points.pt")

                # Save confidence if available
                if "world_points_conf" in predictions:
                    torch.save(predictions["world_points_conf"].cpu(), output_dir / "world_points_conf.pt")
            elif "points" in predictions and not args.no_points:
                # Fallback for older naming
                points = predictions["points"]
                print(f"Point maps shape: {points.shape if hasattr(points, 'shape') else 'N/A'}")
                torch.save(points.cpu(), output_dir / "world_points.pt")

            print(f"\nResults saved to: {output_dir}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nERROR: Out of memory! Try:")
                print(f"  1. Reduce --image-size (e.g., 256 instead of {args.image_size})")
                print(f"  2. Reduce --max-images (e.g., 5 instead of {len(image_paths)})")
                print(f"  3. Use --device cpu with --compute-dtype float32")
                print(f"  4. Process images one at a time")
            raise

    print("\nDone!")


if __name__ == "__main__":
    main()