"""
pose/nodes.py - ComfyUI node definitions for pose rendering and repair.
"""

import numpy as np
import torch
from PIL import Image

from .render import render_single_meta_to_image
from .repair import repair_pose_keypoints
from .scale import (
    MEASUREMENT_NAMES,
    compute_adjust_scale,
    retarget_pose_sequence,
)
from .utils import UPSCALE_METHODS


class MH_RenderPose:
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
        if pose_keypoints is None:
            print("[MH_RenderPose] No pose keypoints provided, returning empty image")
            empty = torch.zeros(
                (1, canvas_height, canvas_width, 3), dtype=torch.float32
            )
            return (empty,)

        if not isinstance(pose_keypoints, (list, tuple)):
            pose_keypoints = [pose_keypoints]

        if len(pose_keypoints) == 0:
            print("[MH_RenderPose] No pose keypoints provided, returning empty image")
            empty = torch.zeros(
                (1, canvas_height, canvas_width, 3), dtype=torch.float32
            )
            return (empty,)

        print(
            f"[MH_RenderPose] Rendering {len(pose_keypoints)} frames at {canvas_width}x{canvas_height}"
        )

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

            if isinstance(img, Image.Image):
                img_np = np.array(img).astype(np.float32) / 255.0
            else:
                img_np = img.astype(np.float32) / 255.0

            images.append(img_np)

        batch = np.stack(images, axis=0)
        tensor = torch.from_numpy(batch)

        return (tensor,)


class MH_RepairDWPose:
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
        if pose_keypoints is None:
            print("[MH_RepairDWPose] No pose keypoints provided, returning empty")
            return (pose_keypoints,)

        if not isinstance(pose_keypoints, (list, tuple)):
            pose_keypoints = [pose_keypoints]

        if len(pose_keypoints) == 0:
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


class MH_AutoPoseScaleCalculator:
    """
    Computes an adjust_scale ratio by comparing a skeleton measurement
    (e.g. torso length) between a source pose and a target pose.

    adjust_scale = target_distance / source_distance

    Connect the FLOAT output to the adjust_scale input of MH_PoseRetargeter
    (or WanViTPoseRetargeterToSrc).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_pose": ("POSE_KEYPOINT",),
                "target_pose": ("POSE_KEYPOINT",),
                "measurement": (
                    MEASUREMENT_NAMES,
                    {
                        "default": "torso",
                        "tooltip": (
                            "Which skeleton segment to compare. "
                            "'torso' = Neck→MidHip, 'legs' = MidHip→MidAnkle, "
                            "'torso_and_legs' = Neck→MidHip→MidAnkle, "
                            "'shoulders' = shoulder width"
                        ),
                    },
                ),
                "fallback_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": (
                            "Value to return if keypoints are insufficient "
                            "to compute the ratio automatically"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("adjust_scale",)
    FUNCTION = "calculate"
    CATEGORY = "MH/Pose"

    def calculate(
        self,
        source_pose,
        target_pose,
        measurement,
        fallback_scale,
    ):
        # Unwrap lists – we only need the first frame from each
        if isinstance(source_pose, (list, tuple)):
            if len(source_pose) == 0:
                print(
                    "[MH_AutoPoseScaleCalculator] source_pose is empty, using fallback"
                )
                return (fallback_scale,)
            source_meta = source_pose[0]
        else:
            source_meta = source_pose

        if isinstance(target_pose, (list, tuple)):
            if len(target_pose) == 0:
                print(
                    "[MH_AutoPoseScaleCalculator] target_pose is empty, using fallback"
                )
                return (fallback_scale,)
            target_meta = target_pose[0]
        else:
            target_meta = target_pose

        scale = compute_adjust_scale(source_meta, target_meta, measurement)

        if scale is None:
            print(
                f"[MH_AutoPoseScaleCalculator] Could not compute scale with "
                f"measurement='{measurement}' (missing keypoints). "
                f"Using fallback={fallback_scale}"
            )
            return (fallback_scale,)

        print(
            f"[MH_AutoPoseScaleCalculator] measurement='{measurement}' → "
            f"adjust_scale={scale:.4f}"
        )
        return (scale,)


class MH_PoseRetargeter:
    """
    Retargets a source pose sequence onto a target skeleton's proportions.

    For every frame the source skeleton is:
      1. Centered on its own anchor (mid-hip)
      2. Scaled by adjust_scale
      3. Repositioned at the target skeleton's anchor

    This is the Python equivalent of WanViTPoseRetargeterToSrc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_pose": ("POSE_KEYPOINT",),
                "target_pose": ("POSE_KEYPOINT",),
                "adjust_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": (
                            "Scale ratio applied to the source skeleton. "
                            "Connect the output of MH_AutoPoseScaleCalculator "
                            "here to automate this value."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("retargeted_pose",)
    FUNCTION = "retarget"
    CATEGORY = "MH/Pose"

    def retarget(
        self,
        source_pose,
        target_pose,
        adjust_scale,
    ):
        # Normalise inputs to lists
        if source_pose is None:
            print("[MH_PoseRetargeter] source_pose is None, returning empty")
            return ([],)

        if not isinstance(source_pose, (list, tuple)):
            source_pose = [source_pose]

        if len(source_pose) == 0:
            print("[MH_PoseRetargeter] source_pose is empty, returning empty")
            return ([],)

        # Target: take first frame only
        if isinstance(target_pose, (list, tuple)):
            if len(target_pose) == 0:
                print(
                    "[MH_PoseRetargeter] target_pose is empty, returning source unmodified"
                )
                return (source_pose,)
            target_meta = target_pose[0]
        else:
            target_meta = target_pose

        print(
            f"[MH_PoseRetargeter] Retargeting {len(source_pose)} frames "
            f"with adjust_scale={adjust_scale:.4f}"
        )

        retargeted = retarget_pose_sequence(
            source_metas=list(source_pose),
            target_meta=target_meta,
            adjust_scale=adjust_scale,
        )

        return (retargeted,)