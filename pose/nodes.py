"""
pose/nodes.py

ComfyUI node definitions for pose rendering and repair.
"""

import torch
import numpy as np
from PIL import Image

from .repair import repair_pose_keypoints
from .render import render_single_meta_to_image
from .utils import UPSCALE_METHODS


class MH_RenderPose:
    """
    ComfyUI node that renders pose keypoints to an image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "canvas_width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Width of the output canvas",
                    },
                ),
                "canvas_height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Height of the output canvas",
                    },
                ),
                "detect_resolution": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Resolution for output (0 = no resize)",
                    },
                ),
                "draw_body": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Draw body skeleton",
                    },
                ),
                "draw_hand": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Draw hand skeletons",
                    },
                ),
                "draw_face": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Draw face keypoints",
                    },
                ),
            },
            "optional": {
                "upscale_method": (UPSCALE_METHODS, {"default": "INTER_CUBIC"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "MH/Pose"

    def render(
        self,
        pose_keypoints,
        canvas_width,
        canvas_height,
        detect_resolution,
        draw_body,
        draw_hand,
        draw_face,
        upscale_method="INTER_CUBIC",
    ):
        if not pose_keypoints or len(pose_keypoints) == 0:
            print("[MH_RenderPose] No pose keypoints provided, returning empty image")
            empty = torch.zeros((1, canvas_height, canvas_width, 3), dtype=torch.float32)
            return (empty,)

        print(f"[MH_RenderPose] Rendering {len(pose_keypoints)} frames at {canvas_width}x{canvas_height}")

        images = []
        for meta in pose_keypoints:
            img, _ = render_single_meta_to_image(
                meta,
                canvas_H=canvas_height,
                canvas_W=canvas_width,
                detect_resolution=detect_resolution,
                upscale_method=upscale_method,
                output_type="pil",
                draw_body=draw_body,
                draw_hand=draw_hand,
                draw_face=draw_face,
            )
            
            # Convert PIL to numpy, then to tensor
            if isinstance(img, Image.Image):
                img_np = np.array(img).astype(np.float32) / 255.0
            else:
                img_np = img.astype(np.float32) / 255.0
            
            images.append(img_np)

        # Stack into batch tensor [B, H, W, C]
        batch = np.stack(images, axis=0)
        tensor = torch.from_numpy(batch)

        return (tensor,)


class MH_RepairDWPose:
    """
    ComfyUI node that repairs DWPose outputs by interpolating missing/low-confidence
    keypoints using temporal information from neighboring frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Keypoints with confidence below this are considered missing",
                    },
                ),
                "temporal_window": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Number of frames to search for valid keypoints",
                    },
                ),
                "heavy_occlusion_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": "If more than this % of keypoints are missing, replace entire pose",
                    },
                ),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("repaired_pose",)
    FUNCTION = "repair"
    CATEGORY = "MH/Pose"

    def repair(
        self,
        pose_keypoints,
        confidence_threshold,
        temporal_window,
        heavy_occlusion_threshold,
    ):
        if not pose_keypoints or len(pose_keypoints) == 0:
            print("[MH_RepairDWPose] No pose keypoints provided, returning empty")
            return (pose_keypoints,)

        print(
            f"[MH_RepairDWPose] Repairing {len(pose_keypoints)} frames with "
            f"confidence={confidence_threshold}, window={temporal_window}, "
            f"heavy_threshold={heavy_occlusion_threshold}"
        )

        repaired = repair_pose_keypoints(
            pose_metas=pose_keypoints,
            confidence_threshold=confidence_threshold,
            temporal_window=temporal_window,
            heavy_threshold=heavy_occlusion_threshold,
        )

        return (repaired,)
