#!/usr/bin/env python3
"""
VGGT Quantized Viser Demo for Mac
=================================

Modified version of demo_viser.py that works with the 4-bit quantized model
so it can run on Mac M2 with limited memory.

Two modes:
  1. Run inference + visualize in one go (uses quantized model)
  2. Load pre-computed predictions and visualize (no model needed)

Usage:
    # Mode 1: Full pipeline (quantized inference + viser)
    python demo_viser_quantized.py --image_folder ./images/ --quantize

    # Mode 2: Visualize pre-computed results
    python demo_viser_quantized.py --load_predictions ./results/ --image_folder ./images/

    # With options
    python demo_viser_quantized.py --image_folder ./images/ --quantize --port 8080 --conf_threshold 25
"""

import os
import glob
import time
import threading
import argparse
from typing import List, Optional
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
from PIL import Image

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

# VGGT imports
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    HAS_VGGT = True
except ImportError:
    HAS_VGGT = False
    print("WARNING: VGGT not installed. Only --load_predictions mode will work.")


# =============================================================================
# 4-bit Quantization Utilities (copied from vggt_quantize_4bit_mac.py)
# =============================================================================

class QuantizedLinear4Bit(torch.nn.Module):
    """4-bit quantized linear layer for Mac/CPU inference."""

    def __init__(self, in_features, out_features, bias=True, block_size=64, compute_dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.compute_dtype = compute_dtype
        num_blocks_in = (in_features + block_size - 1) // block_size
        num_blocks_out = (out_features + block_size - 1) // block_size
        packed_in = (in_features + 1) // 2
        self.register_buffer("weight_packed", torch.zeros((out_features, packed_in), dtype=torch.uint8))
        self.register_buffer("scales", torch.ones((num_blocks_out, num_blocks_in), dtype=compute_dtype))
        self.register_buffer("zeros", torch.zeros((num_blocks_out, num_blocks_in), dtype=compute_dtype))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def pack_weights(self, weight_fp):
        weight_fp = weight_fp.to(torch.float32)
        out_features, in_features = weight_fp.shape
        pad_in = (self.block_size - in_features % self.block_size) % self.block_size
        pad_out = (self.block_size - out_features % self.block_size) % self.block_size
        if pad_in > 0 or pad_out > 0:
            weight_fp = torch.nn.functional.pad(weight_fp, (0, pad_in, 0, pad_out))
        padded_out, padded_in = weight_fp.shape
        num_blocks_in = padded_in // self.block_size
        num_blocks_out = padded_out // self.block_size
        weight_blocks = weight_fp.reshape(num_blocks_out, self.block_size, num_blocks_in, self.block_size)
        weight_blocks = weight_blocks.permute(0, 2, 1, 3)
        w_min = weight_blocks.amin(dim=(2, 3), keepdim=True)
        w_max = weight_blocks.amax(dim=(2, 3), keepdim=True)
        scales = (w_max - w_min) / 15.0
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        zeros = w_min.squeeze(-1).squeeze(-1)
        weight_int = torch.round((weight_blocks - w_min) / scales.unsqueeze(-1).unsqueeze(-1)).to(torch.int32)
        weight_int = torch.clamp(weight_int, 0, 15)
        weight_int = weight_int.permute(0, 2, 1, 3).reshape(padded_out, padded_in)
        if padded_in % 2 == 1:
            weight_int = torch.nn.functional.pad(weight_int, (0, 1))
            padded_in += 1
        weight_even = weight_int[:, 0::2]
        weight_odd = weight_int[:, 1::2]
        weight_packed = (weight_odd << 4) | weight_even
        weight_packed = weight_packed.to(torch.uint8)
        self.weight_packed[:out_features, :(in_features + 1) // 2] = weight_packed[:out_features, :(in_features + 1) // 2]
        self.scales[:num_blocks_out, :num_blocks_in] = scales[:num_blocks_out, :num_blocks_in]
        self.zeros[:num_blocks_out, :num_blocks_in] = zeros[:num_blocks_out, :num_blocks_in]

    def unpack_weights(self):
        out_features, packed_in = self.weight_packed.shape
        in_features = packed_in * 2
        weight_even = self.weight_packed & 0x0F
        weight_odd = (self.weight_packed >> 4) & 0x0F
        weight_int = torch.zeros((out_features, in_features), dtype=torch.int32, device=self.weight_packed.device)
        weight_int[:, 0::2] = weight_even.to(torch.int32)
        weight_int[:, 1::2] = weight_odd.to(torch.int32)
        num_blocks_in = in_features // self.block_size
        num_blocks_out = out_features // self.block_size
        weight_blocks = weight_int.reshape(num_blocks_out, self.block_size, num_blocks_in, self.block_size)
        weight_blocks = weight_blocks.permute(0, 2, 1, 3)
        scales = self.scales[:num_blocks_out, :num_blocks_in].unsqueeze(-1).unsqueeze(-1)
        zeros = self.zeros[:num_blocks_out, :num_blocks_in].unsqueeze(-1).unsqueeze(-1)
        weight_fp = weight_blocks.to(self.compute_dtype) * scales + zeros
        weight_fp = weight_fp.permute(0, 2, 1, 3).reshape(out_features, in_features)
        return weight_fp.to(self.compute_dtype)

    def forward(self, x):
        x = x.to(self.compute_dtype)
        weight = self.unpack_weights()
        weight = weight[:self.out_features, :self.in_features]
        return torch.nn.functional.linear(x, weight, self.bias)


class QuantizedEmbedding4Bit(torch.nn.Module):
    """4-bit quantized embedding layer."""

    def __init__(self, num_embeddings, embedding_dim, block_size=64):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        packed_dim = (embedding_dim + 1) // 2
        self.register_buffer("weight_packed", torch.zeros((num_embeddings, packed_dim), dtype=torch.uint8))
        num_blocks = (embedding_dim + block_size - 1) // block_size
        self.register_buffer("scales", torch.ones((num_embeddings, num_blocks), dtype=torch.float32))
        self.register_buffer("zeros", torch.zeros((num_embeddings, num_blocks), dtype=torch.float32))

    def pack_weights(self, weight_fp):
        weight_fp = weight_fp.to(torch.float32)
        num_embeddings, embedding_dim = weight_fp.shape
        pad_dim = (self.block_size - embedding_dim % self.block_size) % self.block_size
        if pad_dim > 0:
            weight_fp = torch.nn.functional.pad(weight_fp, (0, pad_dim))
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
            weight_int = torch.nn.functional.pad(weight_int, (0, 1))
            padded_dim += 1
        weight_even = weight_int[:, 0::2]
        weight_odd = weight_int[:, 1::2]
        weight_packed = (weight_odd << 4) | weight_even
        self.weight_packed[:num_embeddings, :(embedding_dim + 1) // 2] = weight_packed[:num_embeddings, :(embedding_dim + 1) // 2]
        self.scales[:num_embeddings, :num_blocks] = scales[:num_embeddings, :num_blocks]
        self.zeros[:num_embeddings, :num_blocks] = zeros[:num_embeddings, :num_blocks]

    def unpack_weights(self):
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

    def forward(self, x):
        weight = self.unpack_weights()
        return torch.nn.functional.embedding(x, weight)


def quantize_model_4bit(model, compute_dtype=torch.float32):
    """Recursively replace Linear and Embedding layers with 4-bit versions."""

    def replace_module(parent_module, child_name, child_module):
        if isinstance(child_module, torch.nn.Linear):
            if child_module.in_features < 32 or child_module.out_features < 32:
                return
            quantized = QuantizedLinear4Bit(
                in_features=child_module.in_features,
                out_features=child_module.out_features,
                bias=child_module.bias is not None,
                block_size=64,
                compute_dtype=compute_dtype,
            )
            with torch.no_grad():
                quantized.pack_weights(child_module.weight.data)
                if child_module.bias is not None:
                    quantized.bias.data = child_module.bias.data.to(compute_dtype)
            setattr(parent_module, child_name, quantized)

        elif isinstance(child_module, torch.nn.Embedding):
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

    def recurse_quantize(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if any(skip in full_name.lower() for skip in [
                "norm", "ln", "layernorm", "head", "camera_head", 
                "depth_head", "point_head", "track_head"
            ]):
                continue
            if len(list(child.children())) > 0:
                recurse_quantize(child, full_name)
            replace_module(module, name, child)

    print("Starting 4-bit quantization...")
    recurse_quantize(model)
    print("Quantization complete!")
    return model


# =============================================================================
# Viser Visualization (from original demo_viser.py)
# =============================================================================

def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict: Dictionary with keys:
            "images": (S, 3, H, W) - Input images
            "world_points": (S, H, W, 3)
            "world_points_conf": (S, H, W)
            "depth": (S, H, W, 1) or (S, H, W)
            "depth_conf": (S, H, W)
            "extrinsic": (S, 3, 4)
            "intrinsic": (S, 3, 3)
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Extract data with fallbacks for missing keys
    images = pred_dict.get("images")
    world_points_map = pred_dict.get("world_points")
    conf_map = pred_dict.get("world_points_conf")

    depth_map = pred_dict.get("depth")
    depth_conf = pred_dict.get("depth_conf")

    extrinsics_cam = pred_dict.get("extrinsic")
    intrinsics_cam = pred_dict.get("intrinsic")

    # Validate required data
    if images is None:
        raise ValueError("Missing 'images' in predictions. Cannot visualize without input images.")

    if world_points_map is None and depth_map is None:
        raise ValueError("Missing both 'world_points' and 'depth'. Cannot visualize without 3D points.")

    # Normalize shapes: remove batch dimension and extra channel dims
    def normalize_shape(arr, target_ndim):
        """Remove leading singleton dimensions and squeeze extra dims."""
        if arr is None:
            return None
        # Remove leading singleton batch dims
        while arr.ndim > target_ndim and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        # Remove trailing singleton channel dims
        while arr.ndim > target_ndim and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        return arr

    # Expected shapes:
    # images: (S, 3, H, W)
    # world_points: (S, H, W, 3)
    # conf: (S, H, W)
    # depth: (S, H, W) - remove channel dim if present
    # extrinsic: (S, 3, 4)
    # intrinsic: (S, 3, 3)

    images = normalize_shape(images, 4)           # (S, 3, H, W)
    world_points_map = normalize_shape(world_points_map, 4)  # (S, H, W, 3)
    conf_map = normalize_shape(conf_map, 3)         # (S, H, W)
    depth_map = normalize_shape(depth_map, 3)       # (S, H, W)
    depth_conf = normalize_shape(depth_conf, 3)     # (S, H, W)
    extrinsics_cam = normalize_shape(extrinsics_cam, 3)  # (S, 3, 4)
    intrinsics_cam = normalize_shape(intrinsics_cam, 3)  # (S, 3, 3)

    print(f"Normalized shapes:")
    print(f"  images: {images.shape}")
    if world_points_map is not None:
        print(f"  world_points: {world_points_map.shape}")
    if depth_map is not None:
        print(f"  depth: {depth_map.shape}")
    if extrinsics_cam is not None:
        print(f"  extrinsic: {extrinsics_cam.shape}")

    # Set defaults for missing optional data
    if conf_map is None and world_points_map is not None:
        print("WARNING: world_points_conf not found, using uniform confidence")
        conf_map = np.ones(world_points_map.shape[:3], dtype=np.float32)

    if depth_conf is None and depth_map is not None:
        print("WARNING: depth_conf not found, using uniform confidence")
        depth_conf = np.ones(depth_map.shape[:3], dtype=np.float32)

    if extrinsics_cam is None:
        print("WARNING: extrinsic not found, cannot show camera poses")

    if intrinsics_cam is None:
        print("WARNING: intrinsic not found, using estimated focal length")

    # Compute world points from depth if not using point map
    if not use_point_map:
        # VGGT's unproject_depth_map_to_point_map expects depth with shape [S, H, W, 1]
        # Add channel dimension back if needed
        if depth_map is not None and depth_map.ndim == 3:
            depth_map = depth_map[..., None]  # [S, H, W] -> [S, H, W, 1]
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    colors = images.transpose(0, 2, 3, 1)
    S, H, W, _ = world_points.shape

    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)
    cam_to_world = cam_to_world_mat[:, :3, :]

    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    frame_indices = np.repeat(np.arange(S), H * W)

    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)

    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics, images_):
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        def attach_callback(frustum, frame):
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(range(S)):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud():
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)
        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    if extrinsics_cam is not None:
        visualize_frames(cam_to_world, images)
    else:
        print("Skipping camera visualization (no extrinsics available)")

    print("Starting viser server...")
    if background_mode:
        def server_loop():
            while True:
                time.sleep(0.001)
        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


def apply_sky_segmentation(conf, image_folder):
    """Apply sky segmentation to confidence scores."""
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        from visual_util import download_file_from_url
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            from visual_util import segment_sky
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    sky_mask_array = np.array(sky_mask_list)
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary
    print("Sky segmentation applied successfully")
    return conf


# =============================================================================
# Main: Quantized Inference + Viser
# =============================================================================

parser = argparse.ArgumentParser(description="VGGT quantized demo with viser for Mac")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/",
                    help="Path to folder containing images")
parser.add_argument("--load_predictions", type=str, default=None,
                    help="Load pre-computed predictions from directory (skips model inference)")
parser.add_argument("--save_predictions", type=str, default=None,
                    help="Save predictions to directory for later visualization")
parser.add_argument("--quantize", action="store_true",
                    help="Apply 4-bit quantization to model (recommended for Mac)")
parser.add_argument("--load_quantized_model", type=str, default=None,
                    help="Load pre-saved quantized model checkpoint")
parser.add_argument("--use_point_map", action="store_true",
                    help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true",
                    help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument("--conf_threshold", type=float, default=25.0,
                    help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation")
parser.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "mps", "cuda"], help="Device to use")
parser.add_argument("--compute_dtype", type=str, default="float16",
                    choices=["float32", "float16"], help="Computation dtype")
parser.add_argument("--image_size", type=int, default=294,
                    help="Image size (must be divisible by 14, e.g., 252, 294, 336)")
parser.add_argument("--max_images", type=int, default=None,
                    help="Maximum number of images to process")


def load_and_preprocess_images_safe(image_names, size=None):
    """Load images and optionally resize to target size."""
    try:
        images = load_and_preprocess_images(image_names)
    except Exception:
        # Fallback
        images = []
        for path in image_names:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            images.append(img_tensor)
        images = torch.stack(images)

    if size is not None and size != images.shape[-1]:
        images = torch.nn.functional.interpolate(
            images, size=(size, size), mode="bilinear", align_corners=False
        )

    return images


def run_quantized_inference(image_folder, device, compute_dtype, quantize, 
                            load_quantized_model, image_size, max_images):
    """Run VGGT inference with optional quantization."""

    if not HAS_VGGT:
        raise RuntimeError("VGGT not installed. Cannot run inference.")

    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    compute_dtype = dtype_map[compute_dtype]

    # Load images
    image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_names = [p for p in image_names if Path(p).suffix.lower() in 
                   (".jpg", ".jpeg", ".png", ".bmp", ".webp")]

    if max_images:
        image_names = image_names[:max_images]

    print(f"Found {len(image_names)} images")

    images = load_and_preprocess_images_safe(image_names, size=image_size)
    images = images.to(device)
    print(f"Image tensor shape: {images.shape}")

    # Load model
    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")

    if quantize:
        model = quantize_model_4bit(model, compute_dtype=compute_dtype)

    if load_quantized_model and os.path.exists(load_quantized_model):
        print(f"Loading quantized weights from {load_quantized_model}")
        state_dict = torch.load(load_quantized_model, map_location="cpu")
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=compute_dtype):
                predictions = model(images)
        else:
            if compute_dtype == torch.float16 and device == "mps":
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    predictions = model(images)
            else:
                predictions = model(images)

    print("Converting pose encoding to camera matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Move everything to CPU and squeeze batch dimension
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    # Also store original images for coloring
    predictions["images"] = images.cpu().numpy()

    return predictions


def load_saved_predictions(pred_dir):
    """Load pre-computed predictions from directory."""
    pred_dir = Path(pred_dir)
    predictions = {}

    # Mapping: viser expected name -> possible file names
    file_mappings = {
        "images": ["images.pt"],
        "world_points": ["world_points.pt", "point_maps.pt"],
        "world_points_conf": ["world_points_conf.pt"],
        "depth": ["depth.pt", "depth_maps.pt"],
        "depth_conf": ["depth_conf.pt"],
        "extrinsic": ["extrinsic.pt", "cameras.pt"],
        "intrinsic": ["intrinsic.pt"],
        "pose_enc": ["pose_enc.pt"],
    }

    for key, possible_files in file_mappings.items():
        for filename in possible_files:
            path = pred_dir / filename
            if path.exists():
                try:
                    data = torch.load(path, map_location="cpu", weights_only=True)
                    if isinstance(data, torch.Tensor):
                        data = data.numpy()
                    predictions[key] = data
                    print(f"Loaded {key} from {filename}: {data.shape}")
                    break
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue
        else:
            print(f"WARNING: Could not find file for '{key}'")

    return predictions


def save_predictions(predictions, save_dir):
    """Save predictions for later visualization."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for key, value in predictions.items():
        if isinstance(value, np.ndarray):
            torch.save(torch.from_numpy(value), save_dir / f"{key}.pt")
        elif isinstance(value, torch.Tensor):
            torch.save(value, save_dir / f"{key}.pt")

    print(f"Predictions saved to {save_dir}")


def main():
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
            print("Using MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            args.device = "cuda"
            print("Using CUDA")
        else:
            args.device = "cpu"
            print("Using CPU")

    # Validate image size
    if args.image_size % 14 != 0:
        valid_size = (args.image_size // 14) * 14
        print(f"WARNING: Image size {args.image_size} not divisible by 14. Using {valid_size}")
        args.image_size = valid_size

    # Load or compute predictions
    if args.load_predictions:
        print(f"Loading pre-computed predictions from {args.load_predictions}")
        predictions = load_saved_predictions(args.load_predictions)
    else:
        if not HAS_VGGT:
            print("ERROR: VGGT not installed and no --load_predictions provided.")
            print("Install VGGT or provide pre-computed predictions.")
            return

        predictions = run_quantized_inference(
            args.image_folder,
            args.device,
            args.compute_dtype,
            args.quantize,
            args.load_quantized_model,
            args.image_size,
            args.max_images,
        )

        if args.save_predictions:
            save_predictions(predictions, args.save_predictions)

    # Validate predictions
    required_keys = ["images", "world_points", "world_points_conf", "depth", "depth_conf", "extrinsic", "intrinsic"]
    for key in required_keys:
        if key not in predictions:
            print(f"WARNING: Missing key '{key}' in predictions. Viser may not work correctly.")

    # Ensure images are in (S, 3, H, W) format
    if "images" in predictions:
        imgs = predictions["images"]
        if imgs.ndim == 4 and imgs.shape[1] != 3:
            # Might be (S, H, W, 3)
            if imgs.shape[-1] == 3:
                imgs = imgs.transpose(0, 3, 1, 2)
                predictions["images"] = imgs

    print("\nStarting viser visualization...")
    print(f"Open your browser and go to: http://localhost:{args.port}")

    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )


if __name__ == "__main__":
    main()