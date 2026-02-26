"""
pose/scale.py

Scale calculation utilities for pose keypoints.

Provides functions to:
- Measure skeleton segment distances (torso, legs) from POSE_KEYPOINT data
- Compute an adjust_scale ratio between a target and source skeleton
"""

from typing import Any, List, Optional

import numpy as np

from .keypoints import extract_all_keypoints

# ---------------------------------------------------------------------------
# OpenPose-18 (COCO + Neck) keypoint indices
# ---------------------------------------------------------------------------
# 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
# 5: LShoulder, 6: LElbow, 7: LWrist, 8: RHip, 9: RKnee,
# 10: RAnkle, 11: LHip, 12: LKnee, 13: LAnkle,
# 14: REye, 15: LEye, 16: REar, 17: LEar
# ---------------------------------------------------------------------------

IDX_NOSE = 0
IDX_NECK = 1
IDX_RSHOULDER = 2
IDX_LSHOULDER = 5
IDX_RHIP = 8
IDX_RANKLE = 10
IDX_LHIP = 11
IDX_LANKLE = 13


def _kp_valid(kps: np.ndarray, idx: int, confidence_threshold: float = 0.1) -> bool:
    """Return True if keypoint at *idx* exists and has sufficient confidence."""
    if idx >= kps.shape[0]:
        return False
    return float(kps[idx, 2]) >= confidence_threshold


def _kp_xy(kps: np.ndarray, idx: int) -> np.ndarray:
    """Return the (x, y) of keypoint *idx* as a 1-D array of length 2."""
    return kps[idx, :2].astype(np.float64)


def _midpoint(kps: np.ndarray, idx_a: int, idx_b: int) -> Optional[np.ndarray]:
    """Return the midpoint of two keypoints, or None if either is invalid."""
    if not (_kp_valid(kps, idx_a) and _kp_valid(kps, idx_b)):
        return None
    return (_kp_xy(kps, idx_a) + _kp_xy(kps, idx_b)) / 2.0


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _get_neck(kps: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the Neck coordinate.  Falls back to the midpoint of the two
    shoulders when the dedicated Neck keypoint (index 1) is missing.
    """
    if _kp_valid(kps, IDX_NECK):
        return _kp_xy(kps, IDX_NECK)
    return _midpoint(kps, IDX_RSHOULDER, IDX_LSHOULDER)


def _get_midhip(kps: np.ndarray) -> Optional[np.ndarray]:
    """Return the mid-hip coordinate (average of left and right hip)."""
    return _midpoint(kps, IDX_RHIP, IDX_LHIP)


def _get_mid_ankle(kps: np.ndarray) -> Optional[np.ndarray]:
    """Return the mid-ankle coordinate (average of left and right ankle)."""
    return _midpoint(kps, IDX_RANKLE, IDX_LANKLE)


def measure_torso(kps: np.ndarray) -> Optional[float]:
    """Distance from Neck to Mid-Hip (torso length)."""
    neck = _get_neck(kps)
    midhip = _get_midhip(kps)
    if neck is None or midhip is None:
        return None
    return _euclidean(neck, midhip)


def measure_legs(kps: np.ndarray) -> Optional[float]:
    """Distance from Mid-Hip to Mid-Ankle (leg length)."""
    midhip = _get_midhip(kps)
    mid_ankle = _get_mid_ankle(kps)
    if midhip is None or mid_ankle is None:
        return None
    return _euclidean(midhip, mid_ankle)


def measure_torso_and_legs(kps: np.ndarray) -> Optional[float]:
    """Combined distance: Neck -> Mid-Hip -> Mid-Ankle."""
    t = measure_torso(kps)
    lg = measure_legs(kps)
    if t is None or lg is None:
        return None
    return t + lg


def measure_shoulders(kps: np.ndarray) -> Optional[float]:
    """Distance between left and right shoulders."""
    if not (_kp_valid(kps, IDX_RSHOULDER) and _kp_valid(kps, IDX_LSHOULDER)):
        return None
    return _euclidean(_kp_xy(kps, IDX_RSHOULDER), _kp_xy(kps, IDX_LSHOULDER))


# Map user-facing measurement names to functions
MEASUREMENT_FUNCTIONS = {
    "torso": measure_torso,
    "legs": measure_legs,
    "torso_and_legs": measure_torso_and_legs,
    "shoulders": measure_shoulders,
}

MEASUREMENT_NAMES = list(MEASUREMENT_FUNCTIONS.keys())


# ---------------------------------------------------------------------------
# Public: extract body keypoints from a single POSE_KEYPOINT frame meta
# ---------------------------------------------------------------------------


def _body_kps_from_meta(meta: Any) -> Optional[np.ndarray]:
    """
    Extract body keypoints (Nx3) from a single POSE_KEYPOINT frame dict.
    Returns pixel-space coordinates as-is (no normalisation).
    """
    body_kps, _, _, _ = extract_all_keypoints(meta)
    return body_kps


# ---------------------------------------------------------------------------
# Public: compute adjust_scale
# ---------------------------------------------------------------------------


def compute_adjust_scale(
    source_meta: Any,
    target_meta: Any,
    measurement: str = "torso",
) -> Optional[float]:
    """
    Compute ``adjust_scale = target_distance / source_distance`` for the
    requested *measurement* anchor.

    Parameters
    ----------
    source_meta : POSE_KEYPOINT frame dict
        Pose extracted from the first frame of the driving video.
    target_meta : POSE_KEYPOINT frame dict
        Pose extracted from the reference (character) image.
    measurement : str
        One of ``"torso"``, ``"legs"``, ``"torso_and_legs"``, ``"shoulders"``.

    Returns
    -------
    float or None
        The ratio, or *None* if keypoints are insufficient.
    """
    measure_fn = MEASUREMENT_FUNCTIONS.get(measurement)
    if measure_fn is None:
        raise ValueError(
            f"Unknown measurement '{measurement}'. Choose from: {MEASUREMENT_NAMES}"
        )

    src_kps = _body_kps_from_meta(source_meta)
    tgt_kps = _body_kps_from_meta(target_meta)

    if src_kps is None or tgt_kps is None:
        return None

    src_dist = measure_fn(src_kps)
    tgt_dist = measure_fn(tgt_kps)

    if src_dist is None or tgt_dist is None:
        return None
    if src_dist < 1e-6:
        return None  # avoid division by zero

    return tgt_dist / src_dist


def _select_anchor(kps: np.ndarray) -> Optional[np.ndarray]:
    return _get_midhip(kps) or _get_neck(kps)


def _set_body_kps(meta: Any, body_kps: np.ndarray) -> Any:
    if meta is None:
        return meta

    if hasattr(meta, "kps_body"):
        try:
            meta.kps_body = body_kps[:, :2]
        except Exception:
            meta.kps_body = body_kps[:, :2].tolist()

        if hasattr(meta, "kps_body_p"):
            try:
                meta.kps_body_p = body_kps[:, 2]
            except Exception:
                meta.kps_body_p = body_kps[:, 2].tolist()
        return meta

    if hasattr(meta, "keypoints_body"):
        try:
            setattr(meta, "keypoints_body", body_kps)
        except Exception:
            setattr(meta, "keypoints_body", body_kps.tolist())
        return meta

    if isinstance(meta, dict):
        if "keypoints_body" in meta:
            meta["keypoints_body"] = body_kps.tolist()
        if "people" in meta and meta.get("people"):
            meta["people"][0]["pose_keypoints_2d"] = body_kps.reshape(-1).tolist()
        elif "pose_keypoints_2d" in meta:
            meta["pose_keypoints_2d"] = body_kps.reshape(-1).tolist()
        return meta

    return meta


def retarget_pose_sequence(
    source_metas: List[Any],
    target_meta: Any,
    adjust_scale: float = 1.0,
) -> List[Any]:
    if source_metas is None:
        return []

    target_kps = _body_kps_from_meta(target_meta)
    if target_kps is None:
        return list(source_metas)

    target_anchor = _select_anchor(target_kps)
    if target_anchor is None:
        return list(source_metas)

    retargeted: List[Any] = []

    for meta in source_metas:
        body_kps = _body_kps_from_meta(meta)
        if body_kps is None:
            retargeted.append(meta)
            continue

        src_anchor = _select_anchor(body_kps)
        if src_anchor is None:
            retargeted.append(meta)
            continue

        kps = np.array(body_kps, dtype=np.float64, copy=True)
        xy = kps[:, :2]
        valid = np.isfinite(xy).all(axis=1)

        if np.any(valid):
            xy_scaled = (xy[valid] - src_anchor) * float(adjust_scale) + target_anchor
            kps[valid, :2] = xy_scaled

        retargeted.append(_set_body_kps(meta, kps.astype(np.float32)))

    return retargeted
