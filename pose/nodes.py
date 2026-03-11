"""
pose/nodes.py - ComfyUI node definitions for pose rendering and repair.
"""

import numpy as np
import torch
from PIL import Image

from .keypoints import extract_all_keypoints
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


class MH_PoseAlignImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "image_to_align": ("IMAGE",),
                "pose_ref": ("POSE_KEYPOINT",),
                "pose_to_align": ("POSE_KEYPOINT",),
                "match_scale": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Scale image_to_align to match target pose height",
                    },
                ),
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Keypoint confidence threshold",
                    },
                ),
                "max_scale": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "min": 1.0,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": "Maximum allowed scale factor (clamps zoom-in)",
                    },
                ),
                "frame_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Frame index to use from pose sequences",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("aligned_image",)
    FUNCTION = "align"
    CATEGORY = "MH/Pose"

    def _select_meta(self, pose_keypoints, frame_index: int):
        if isinstance(pose_keypoints, (list, tuple)):
            if len(pose_keypoints) == 0:
                return None
            idx = min(max(int(frame_index), 0), len(pose_keypoints) - 1)
            return pose_keypoints[idx]
        return pose_keypoints

    def _bbox_from_meta(self, meta, canvas_w: int, canvas_h: int, conf: float):
        if meta is None:
            return None
        body_kps, left_hand_kps, right_hand_kps, face_kps = extract_all_keypoints(meta)
        arrays = [body_kps, left_hand_kps, right_hand_kps, face_kps]

        points = []
        for arr in arrays:
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.ndim == 1 and arr.size % 3 == 0:
                arr = arr.reshape(-1, 3)
            if arr.ndim == 2 and arr.shape[1] == 2:
                arr = np.concatenate(
                    [arr, np.ones((arr.shape[0], 1), dtype=np.float32)], axis=1
                )

            xs = arr[:, 0]
            ys = arr[:, 1]
            if (
                np.nanmax(xs) <= 1.5
                and np.nanmax(ys) <= 1.5
                and canvas_w > 0
                and canvas_h > 0
            ):
                arr = arr.copy()
                arr[:, 0] = arr[:, 0] * float(canvas_w)
                arr[:, 1] = arr[:, 1] * float(canvas_h)

            for x, y, c in arr:
                if float(c) > conf and not (float(x) == 0.0 and float(y) == 0.0):
                    points.append((float(x), float(y)))

        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _face_bbox_from_meta(self, meta, canvas_w: int, canvas_h: int, conf: float):
        if meta is None:
            return None
        _, _, _, face_kps = extract_all_keypoints(meta)
        if face_kps is None:
            return None

        arr = np.asarray(face_kps, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim == 1 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = np.concatenate(
                [arr, np.ones((arr.shape[0], 1), dtype=np.float32)], axis=1
            )

        xs = arr[:, 0]
        ys = arr[:, 1]
        if (
            np.nanmax(xs) <= 1.5
            and np.nanmax(ys) <= 1.5
            and canvas_w > 0
            and canvas_h > 0
        ):
            arr = arr.copy()
            arr[:, 0] = arr[:, 0] * float(canvas_w)
            arr[:, 1] = arr[:, 1] * float(canvas_h)

        points = []
        for x, y, c in arr:
            if float(c) > conf and not (float(x) == 0.0 and float(y) == 0.0):
                points.append((float(x), float(y)))

        if len(points) < 2:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _face_center_from_meta(self, meta, canvas_w: int, canvas_h: int, conf: float):
        bbox = self._face_bbox_from_meta(meta, canvas_w, canvas_h, conf)
        if bbox is None:
            return None
        min_x, min_y, max_x, max_y = bbox
        return (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

    def _hip_center_from_meta(self, meta, canvas_w: int, canvas_h: int, conf: float):
        if meta is None:
            return None
        body_kps, _, _, _ = extract_all_keypoints(meta)
        if body_kps is None:
            return None

        arr = np.asarray(body_kps, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim == 1 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = np.concatenate(
                [arr, np.ones((arr.shape[0], 1), dtype=np.float32)], axis=1
            )

        xs = arr[:, 0]
        ys = arr[:, 1]
        if (
            np.nanmax(xs) <= 1.5
            and np.nanmax(ys) <= 1.5
            and canvas_w > 0
            and canvas_h > 0
        ):
            arr = arr.copy()
            arr[:, 0] = arr[:, 0] * float(canvas_w)
            arr[:, 1] = arr[:, 1] * float(canvas_h)

        def _valid(idx: int):
            if idx >= arr.shape[0]:
                return None
            x, y, c = arr[idx]
            if float(c) > conf and not (float(x) == 0.0 and float(y) == 0.0):
                return float(x), float(y)
            return None

        mid = _valid(8)
        if mid is not None:
            return mid

        left = _valid(9)
        right = _valid(12)
        if left is not None and right is not None:
            return (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0
        if left is not None:
            return left
        if right is not None:
            return right
        return None

    def _feet_center_from_meta(self, meta, canvas_w: int, canvas_h: int, conf: float):
        if meta is None:
            return None
        body_kps, _, _, _ = extract_all_keypoints(meta)
        if body_kps is None:
            return None

        arr = np.asarray(body_kps, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim == 1 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = np.concatenate(
                [arr, np.ones((arr.shape[0], 1), dtype=np.float32)], axis=1
            )

        xs = arr[:, 0]
        ys = arr[:, 1]
        if (
            np.nanmax(xs) <= 1.5
            and np.nanmax(ys) <= 1.5
            and canvas_w > 0
            and canvas_h > 0
        ):
            arr = arr.copy()
            arr[:, 0] = arr[:, 0] * float(canvas_w)
            arr[:, 1] = arr[:, 1] * float(canvas_h)

        def _valid(idx: int):
            if idx >= arr.shape[0]:
                return None
            x, y, c = arr[idx]
            if float(c) > conf and not (float(x) == 0.0 and float(y) == 0.0):
                return float(x), float(y)
            return None

        # Ankle indices: 10 (left), 13 (right)
        left = _valid(10)
        right = _valid(13)
        if left is not None and right is not None:
            return (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0
        if left is not None:
            return left
        if right is not None:
            return right
        return None

    def _torso_bbox_from_meta(self, meta, canvas_w: int, canvas_h: int, conf: float):
        if meta is None:
            return None
        body_kps, _, _, _ = extract_all_keypoints(meta)
        if body_kps is None:
            return None

        arr = np.asarray(body_kps, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim == 1 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = np.concatenate(
                [arr, np.ones((arr.shape[0], 1), dtype=np.float32)], axis=1
            )

        xs = arr[:, 0]
        ys = arr[:, 1]
        if (
            np.nanmax(xs) <= 1.5
            and np.nanmax(ys) <= 1.5
            and canvas_w > 0
            and canvas_h > 0
        ):
            arr = arr.copy()
            arr[:, 0] = arr[:, 0] * float(canvas_w)
            arr[:, 1] = arr[:, 1] * float(canvas_h)

        torso_indices = [1, 2, 5, 8, 9, 12]
        points = []
        for idx in torso_indices:
            if idx >= arr.shape[0]:
                continue
            x, y, c = arr[idx]
            if float(c) > conf and not (float(x) == 0.0 and float(y) == 0.0):
                points.append((float(x), float(y)))

        if len(points) < 2:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def align(
        self,
        reference_image,
        image_to_align,
        pose_ref,
        pose_to_align,
        match_scale,
        confidence_threshold,
        max_scale,
        frame_index,
    ):
        if reference_image is None or image_to_align is None:
            print("[MH_PoseAlignImage] Missing images; returning input")
            return (image_to_align,)

        ref_meta = self._select_meta(pose_ref, frame_index)
        align_meta = self._select_meta(pose_to_align, frame_index)

        if ref_meta is None or align_meta is None:
            print("[MH_PoseAlignImage] Missing pose keypoints; returning input")
            return (image_to_align,)

        ref_h = int(reference_image.shape[1])
        ref_w = int(reference_image.shape[2])

        img_np = (image_to_align[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        w2, h2 = pil_img.size

        face_bbox1 = self._face_bbox_from_meta(
            ref_meta, ref_w, ref_h, confidence_threshold
        )
        face_bbox2 = self._face_bbox_from_meta(align_meta, w2, h2, confidence_threshold)

        if face_bbox1 is not None and face_bbox2 is not None:
            min_x1, min_y1, max_x1, max_y1 = face_bbox1
            min_x2, min_y2, max_x2, max_y2 = face_bbox2
        else:
            torso_bbox1 = self._torso_bbox_from_meta(
                ref_meta, ref_w, ref_h, confidence_threshold
            )
            torso_bbox2 = self._torso_bbox_from_meta(
                align_meta, w2, h2, confidence_threshold
            )

            if torso_bbox1 is None or torso_bbox2 is None:
                bbox1 = self._bbox_from_meta(
                    ref_meta, ref_w, ref_h, confidence_threshold
                )
                bbox2 = self._bbox_from_meta(align_meta, w2, h2, confidence_threshold)
                if bbox1 is None or bbox2 is None:
                    print("[MH_PoseAlignImage] No valid keypoints; returning input")
                    return (image_to_align,)
                min_x1, min_y1, max_x1, max_y1 = bbox1
                min_x2, min_y2, max_x2, max_y2 = bbox2
            else:
                min_x1, min_y1, max_x1, max_y1 = torso_bbox1
                min_x2, min_y2, max_x2, max_y2 = torso_bbox2

        cx1 = (min_x1 + max_x1) / 2.0
        cy1 = min_y1

        cx2 = (min_x2 + max_x2) / 2.0
        cy2 = min_y2

        scale_factor = 1.0
        if match_scale:
            face_center_1 = self._face_center_from_meta(
                ref_meta, ref_w, ref_h, confidence_threshold
            )
            face_center_2 = self._face_center_from_meta(
                align_meta, w2, h2, confidence_threshold
            )

            # Try feet first, fallback to hip
            feet_center_1 = self._feet_center_from_meta(
                ref_meta, ref_w, ref_h, confidence_threshold
            )
            feet_center_2 = self._feet_center_from_meta(
                align_meta, w2, h2, confidence_threshold
            )

            lower_center_1 = feet_center_1
            lower_center_2 = feet_center_2

            if lower_center_1 is None or lower_center_2 is None:
                lower_center_1 = self._hip_center_from_meta(
                    ref_meta, ref_w, ref_h, confidence_threshold
                )
                lower_center_2 = self._hip_center_from_meta(
                    align_meta, w2, h2, confidence_threshold
                )

            if (
                face_center_1 is not None
                and lower_center_1 is not None
                and face_center_2 is not None
                and lower_center_2 is not None
            ):
                dist1 = abs(face_center_1[1] - lower_center_1[1])
                dist2 = abs(face_center_2[1] - lower_center_2[1])
                if dist2 > 0:
                    scale_factor = dist1 / dist2
                    if scale_factor < 1.0:
                        scale_factor = 1.0
                    if max_scale is not None:
                        scale_factor = min(scale_factor, float(max_scale))

        if match_scale and scale_factor > 1.0 and face_bbox2 is not None:
            face_min_x2, face_min_y2, face_max_x2, face_max_y2 = face_bbox2
            sfx1 = face_min_x2 * scale_factor
            sfy1 = face_min_y2 * scale_factor
            sfx2 = face_max_x2 * scale_factor
            sfy2 = face_max_y2 * scale_factor

            left_tmp = int(round(cx2 * scale_factor - cx1))
            top_tmp = int(round(cy2 * scale_factor - cy1))
            right_tmp = left_tmp + ref_w
            bottom_tmp = top_tmp + ref_h

            if not (
                sfx1 >= left_tmp
                and sfx2 <= right_tmp
                and sfy1 >= top_tmp
                and sfy2 <= bottom_tmp
            ):
                scale_factor = 1.0

        if match_scale and scale_factor > 1.0:
            new_w2 = max(1, int(round(w2 * scale_factor)))
            new_h2 = max(1, int(round(h2 * scale_factor)))
            pil_img = pil_img.resize((new_w2, new_h2), Image.Resampling.LANCZOS)
            cx2 = cx2 * scale_factor
            cy2 = cy2 * scale_factor

        left = int(round(cx2 - cx1))
        top = int(round(cy2 - cy1))
        right = left + ref_w
        bottom = top + ref_h

        aligned = pil_img.crop((left, top, right, bottom))
        aligned_np = np.array(aligned).astype(np.float32) / 255.0
        aligned_tensor = torch.from_numpy(aligned_np).unsqueeze(0)

        return (aligned_tensor,)
