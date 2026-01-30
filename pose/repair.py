"""
pose/repair.py

Pose keypoint repair with temporal interpolation for occlusion handling.
"""

import copy
import numpy as np


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
