import copy
import math

import numpy as np


def align_pose_frames_to_vae(pose_metas, temporal_divisor=4):
    """
    Align pose frame count to match VAE's latent frame calculation.
    
    The WanVideo VAE compresses video temporally by a factor of 4, using ceil().
    If pose wrapper uses floor(), there's a mismatch. This function pads
    the pose frames to ensure they match the VAE's expected count.
    
    Args:
        pose_metas: List of pose metadata objects or dicts.
        temporal_divisor: VAE's temporal compression factor (default 4).
    
    Returns:
        Aligned pose_metas list with frame count that matches VAE expectations.
    """
    if not pose_metas or len(pose_metas) == 0:
        return pose_metas
    
    num_frames = len(pose_metas)
    
    # VAE uses ceil, so we need to pad to match
    target_latent_frames = math.ceil(num_frames / temporal_divisor)
    target_frames = target_latent_frames * temporal_divisor
    
    if num_frames == target_frames:
        # Already aligned
        return pose_metas
    
    frames_to_add = target_frames - num_frames
    
    if frames_to_add > 0:
        # Pad by repeating the last frame
        print(f"[PoseAlign] Padding {num_frames} frames to {target_frames} "
              f"(+{frames_to_add} frames) to match VAE latent calculation")
        
        last_frame = pose_metas[-1]
        for _ in range(frames_to_add):
            pose_metas.append(copy.deepcopy(last_frame))
    else:
        # Trim excess frames (rare case)
        print(f"[PoseAlign] Trimming {num_frames} frames to {target_frames}")
        pose_metas = pose_metas[:target_frames]
    
    return pose_metas


def repair_pose_keypoints(
    pose_metas,
    confidence_threshold=0.3,
    temporal_window=8,
    heavy_threshold=0.5,
):
    """
    Advanced occlusion repair with two strategies:
    1. Light occlusion (<50% missing): Interpolate individual keypoints
    2. Heavy occlusion (>50% missing): Replace entire pose with interpolation from good frames

    Args:
        pose_metas: List of pose metadata objects or dicts from ViTPose/DWPose.
        confidence_threshold: Minimum confidence score (0.0-1.0) to consider a keypoint valid.
        temporal_window: Number of frames to look back/forward for a "good" keypoint.
        heavy_threshold: If more than this % of body is missing, trigger Heavy Repair.

    Returns:
        repaired_metas: List of repaired pose objects.
    """
    num_frames = len(pose_metas)

    # Stats tracking
    stats = {"light_repairs": 0, "heavy_repairs": 0, "keypoints_fixed": 0}

    # First pass: extract all keypoints and calculate per-frame quality
    all_kps = []
    frame_quality = []  # Percentage of good keypoints per frame

    for meta in pose_metas:
        # Handle different data structures (Class objects vs Dictionaries)
        if hasattr(meta, "kps_body") and meta.kps_body is not None:
            # meta.kps_body is usually [N, 2], meta.kps_body_p is [N]
            kps = np.concatenate([meta.kps_body, meta.kps_body_p[:, None]], axis=1)
        elif hasattr(meta, "keypoints_body"):
            kps = np.array(meta.keypoints_body)
        elif isinstance(meta, dict) and "keypoints_body" in meta:
            kps = np.array(meta["keypoints_body"])
        else:
            kps = None
        all_kps.append(kps)

        # Calculate frame quality
        if kps is not None and len(kps) > 0:
            good_kps = np.sum(kps[:, 2] >= confidence_threshold)
            quality = good_kps / len(kps)
        else:
            quality = 0.0
        frame_quality.append(quality)

    # Identify good frames (for heavy occlusion interpolation)
    good_frame_indices = [
        i for i, q in enumerate(frame_quality) if q >= (1 - heavy_threshold)
    ]

    # Second pass: repair each frame
    repaired_metas = []

    for frame_idx, meta in enumerate(pose_metas):
        repaired_meta = copy.deepcopy(meta)
        kps = all_kps[frame_idx]

        if kps is None or len(kps) == 0:
            repaired_metas.append(repaired_meta)
            continue

        kps = kps.copy()
        num_keypoints = len(kps)
        missing_ratio = 1.0 - frame_quality[frame_idx]

        # ---------------------------------------------------------
        # STRATEGY A: HEAVY OCCLUSION (Replace entire pose)
        # ---------------------------------------------------------
        if missing_ratio > heavy_threshold:
            # Find nearest good frames before and after
            prev_good = None
            next_good = None

            for gi in good_frame_indices:
                if gi < frame_idx:
                    prev_good = gi
                elif gi > frame_idx and next_good is None:
                    next_good = gi
                    break

            if prev_good is not None and next_good is not None:
                # Interpolate entire pose between two good frames
                t = (frame_idx - prev_good) / (next_good - prev_good)
                prev_kps = all_kps[prev_good]
                next_kps = all_kps[next_good]

                if prev_kps is not None and next_kps is not None:
                    for kp_idx in range(min(len(kps), len(prev_kps), len(next_kps))):
                        kps[kp_idx][0] = prev_kps[kp_idx][0] + t * (
                            next_kps[kp_idx][0] - prev_kps[kp_idx][0]
                        )
                        kps[kp_idx][1] = prev_kps[kp_idx][1] + t * (
                            next_kps[kp_idx][1] - prev_kps[kp_idx][1]
                        )
                        kps[kp_idx][2] = 0.6  # Mark as interpolated (medium confidence)
                    stats["heavy_repairs"] += 1
            elif prev_good is not None:
                # Only past good frame - copy it (freeze frame)
                prev_kps = all_kps[prev_good]
                if prev_kps is not None:
                    for kp_idx in range(min(len(kps), len(prev_kps))):
                        kps[kp_idx] = prev_kps[kp_idx].copy()
                        kps[kp_idx][2] *= 0.7
                    stats["heavy_repairs"] += 1
            elif next_good is not None:
                # Only future good frame - copy it (reverse freeze)
                next_kps = all_kps[next_good]
                if next_kps is not None:
                    for kp_idx in range(min(len(kps), len(next_kps))):
                        kps[kp_idx] = next_kps[kp_idx].copy()
                        kps[kp_idx][2] *= 0.7
                    stats["heavy_repairs"] += 1

        # ---------------------------------------------------------
        # STRATEGY B: LIGHT OCCLUSION (Repair individual joints)
        # ---------------------------------------------------------
        else:
            frame_keypoints_fixed = 0
            for kp_idx in range(num_keypoints):
                if kps[kp_idx][2] < confidence_threshold:
                    # Find nearest good frame BEFORE
                    prev_frame = None
                    prev_kp = None
                    for offset in range(1, temporal_window + 1):
                        if frame_idx - offset >= 0:
                            past_kps = all_kps[frame_idx - offset]
                            if past_kps is not None and len(past_kps) > kp_idx:
                                if past_kps[kp_idx][2] >= confidence_threshold:
                                    prev_frame = frame_idx - offset
                                    prev_kp = past_kps[kp_idx].copy()
                                    break

                    # Find nearest good frame AFTER
                    next_frame = None
                    next_kp = None
                    for offset in range(1, temporal_window + 1):
                        if frame_idx + offset < num_frames:
                            future_kps = all_kps[frame_idx + offset]
                            if future_kps is not None and len(future_kps) > kp_idx:
                                if future_kps[kp_idx][2] >= confidence_threshold:
                                    next_frame = frame_idx + offset
                                    next_kp = future_kps[kp_idx].copy()
                                    break

                    # Interpolate or copy
                    if prev_kp is not None and next_kp is not None:
                        t = (frame_idx - prev_frame) / (next_frame - prev_frame)
                        kps[kp_idx][0] = prev_kp[0] + t * (next_kp[0] - prev_kp[0])
                        kps[kp_idx][1] = prev_kp[1] + t * (next_kp[1] - prev_kp[1])
                        kps[kp_idx][2] = (prev_kp[2] + next_kp[2]) / 2 * 0.8
                        stats["keypoints_fixed"] += 1
                        frame_keypoints_fixed += 1
                    elif prev_kp is not None:
                        kps[kp_idx] = prev_kp.copy()
                        kps[kp_idx][2] *= 0.7
                        stats["keypoints_fixed"] += 1
                        frame_keypoints_fixed += 1
                    elif next_kp is not None:
                        kps[kp_idx] = next_kp.copy()
                        kps[kp_idx][2] *= 0.7
                        stats["keypoints_fixed"] += 1
                        frame_keypoints_fixed += 1

            if frame_keypoints_fixed > 0:
                stats["light_repairs"] += 1

        # Update the meta with repaired keypoints
        if hasattr(repaired_meta, "kps_body"):
            repaired_meta.kps_body = kps[:, :2]
            repaired_meta.kps_body_p = kps[:, 2]
        elif hasattr(repaired_meta, "keypoints_body"):
            repaired_meta.keypoints_body = kps.tolist()
        elif isinstance(repaired_meta, dict):
            repaired_meta["keypoints_body"] = kps.tolist()

        repaired_metas.append(repaired_meta)

    # Print summary
    print(
        f"📊 Repair stats: {stats['light_repairs']} light repairs (glitches), "
        f"{stats['heavy_repairs']} heavy occlusions (replacements), "
        f"{stats['keypoints_fixed']} specific joints fixed"
    )

    return repaired_metas


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
                "temporal_divisor": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "VAE temporal compression factor for frame alignment (usually 4)",
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
        temporal_divisor,
    ):
        # pose_keypoints is typically a list of pose metadata per frame
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

        # Align frames to VAE expectations to prevent tensor size mismatch
        aligned = align_pose_frames_to_vae(repaired, temporal_divisor=temporal_divisor)

        return (aligned,)


class MH_AlignPoseFrames:
    """
    ComfyUI node that aligns pose frame count to match VAE's latent frame calculation.
    
    Fixes the RuntimeError caused by tensor size mismatch between Video VAE (uses ceil)
    and Pose Wrapper (uses floor) when frame count isn't divisible by 4.
    
    Example: 17 frames → VAE expects 5 latent frames, but pose gives 4 → crash.
    This node pads to 20 frames so both agree on 5 latent frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "temporal_divisor": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "VAE temporal compression factor (usually 4)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("aligned_pose",)
    FUNCTION = "align"
    CATEGORY = "MH/Pose"

    def align(self, pose_keypoints, temporal_divisor):
        if not pose_keypoints or len(pose_keypoints) == 0:
            print("[MH_AlignPoseFrames] No pose keypoints provided, returning empty")
            return (pose_keypoints,)

        aligned = align_pose_frames_to_vae(pose_keypoints, temporal_divisor)
        return (aligned,)


NODE_CLASS_MAPPINGS = {
    "MH_RepairDWPose": MH_RepairDWPose,
    "MH_AlignPoseFrames": MH_AlignPoseFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MH_RepairDWPose": "MH Repair DWPose",
    "MH_AlignPoseFrames": "MH Align Pose Frames",
}
