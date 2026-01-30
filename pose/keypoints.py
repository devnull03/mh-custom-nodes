"""
pose/keypoints.py

Keypoint extraction, normalization, and data structures for pose handling.
"""

from typing import List, Optional, Any
import numpy as np


class SimpleKeypoint:
    """Simple keypoint with x, y coordinates and confidence score."""
    
    def __init__(self, x: float, y: float, score: float = 1.0):
        self.x = x
        self.y = y
        self.score = score


class SimpleBodyResult:
    """Container for body keypoints."""
    
    def __init__(self, keypoints: List[Optional[SimpleKeypoint]]):
        self.keypoints = keypoints
        self.total_score = float(sum([kp.score if kp is not None else 0.0 for kp in keypoints]))
        self.total_parts = sum([1 for kp in keypoints if kp is not None])


class SimplePoseResult:
    """Container for full pose (body, hands, face)."""
    
    def __init__(
        self,
        body: SimpleBodyResult,
        left_hand: Optional[List[SimpleKeypoint]] = None,
        right_hand: Optional[List[SimpleKeypoint]] = None,
        face: Optional[List[SimpleKeypoint]] = None,
    ):
        self.body = body
        self.left_hand = left_hand
        self.right_hand = right_hand
        self.face = face


def extract_kps_array(meta: Any) -> Optional[np.ndarray]:
    """
    Extract keypoints from various frame meta formats.
    
    Supported inputs:
      - Object with attributes: meta.kps_body (Nx2) and meta.kps_body_p (N,)
      - Object with attribute meta.keypoints_body (list/array Nx3)
      - Dict with "keypoints_body" entry (list Nx3)
      - OpenPose-style dict with "people" list
      
    Returns:
        Nx3 numpy array [x, y, conf] or None if nothing found.
    """
    if meta is None:
        return None

    # Object with kps_body + kps_body_p
    if hasattr(meta, "kps_body") and getattr(meta, "kps_body") is not None:
        kps_body = np.asarray(getattr(meta, "kps_body"))
        probs = np.asarray(getattr(meta, "kps_body_p", np.ones((kps_body.shape[0],))))
        
        if kps_body.ndim == 1 and kps_body.size == 2:
            kps_body = kps_body.reshape(1, 2)
        probs = probs.reshape(-1)
        
        if kps_body.shape[0] != probs.shape[0]:
            min_n = min(kps_body.shape[0], probs.shape[0])
            kps_body = kps_body[:min_n]
            probs = probs[:min_n]
        
        kps = np.concatenate([kps_body[:, :2], probs[:, None]], axis=1)
        return kps

    # Object with keypoints_body attribute
    if hasattr(meta, "keypoints_body"):
        arr = np.asarray(getattr(meta, "keypoints_body"))
        if arr.ndim == 1 and arr.size == 0:
            return None
        if arr.ndim == 1 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        return arr.astype(np.float32)

    # Dict with keypoints_body
    if isinstance(meta, dict):
        if "keypoints_body" in meta:
            arr = np.asarray(meta["keypoints_body"])
            if arr.ndim == 1 and arr.size % 3 == 0:
                arr = arr.reshape(-1, 3)
            return arr.astype(np.float32)
        
        # OpenPose-style dict with 'people' list
        if "people" in meta and isinstance(meta["people"], list) and len(meta["people"]) > 0:
            first = meta["people"][0]
            if "pose_keypoints_2d" in first:
                arr = np.asarray(first["pose_keypoints_2d"])
                if arr.ndim == 1 and arr.size % 3 == 0:
                    arr = arr.reshape(-1, 3)
                return arr.astype(np.float32)
    
    return None


def normalize_keypoints_array(
    kps: np.ndarray,
    canvas_H: int,
    canvas_W: int,
    expected_normalized: Optional[bool] = None,
) -> Optional[np.ndarray]:
    """
    Ensure keypoints are normalized to 0..1 range.
    
    Args:
        kps: Nx3 array of keypoints (x, y, conf)
        canvas_H: Canvas height for normalization
        canvas_W: Canvas width for normalization
        expected_normalized: If True/False, force that behavior. If None, auto-detect.
        
    Returns:
        Nx3 array with normalized coordinates.
    """
    if kps is None:
        return None
    
    arr = np.asarray(kps).astype(np.float32).copy()
    
    if arr.size == 0:
        return arr.reshape(0, 3)
    
    if arr.ndim == 1 and arr.size % 3 == 0:
        arr = arr.reshape(-1, 3)
    
    # If any x or y > 1.5, assume pixel coords
    xs = arr[:, 0]
    ys = arr[:, 1]
    
    if expected_normalized is None:
        need_norm = (np.nanmax(xs) > 1.5) or (np.nanmax(ys) > 1.5)
    else:
        need_norm = not expected_normalized
    
    if need_norm:
        if canvas_W <= 0 or canvas_H <= 0:
            raise ValueError("canvas_H and canvas_W must be > 0 to normalize pixel coords to 0..1")
        arr[:, 0] = arr[:, 0] / float(canvas_W)
        arr[:, 1] = arr[:, 1] / float(canvas_H)
    
    # Clamp to [0, 1]
    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)
    
    # Ensure conf column exists
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=np.float32)], axis=1)
    
    return arr


def build_simple_poses_from_kps_arrays(
    list_of_kps_arrays: List[np.ndarray],
    canvas_H: int,
    canvas_W: int,
) -> List[SimplePoseResult]:
    """
    Convert a list of Nx3 arrays into SimplePoseResult list.
    
    Args:
        list_of_kps_arrays: List of Nx3 arrays (x, y, conf) - normalized or pixel coords
        canvas_H: Canvas height for normalization
        canvas_W: Canvas width for normalization
        
    Returns:
        List of SimplePoseResult objects.
    """
    poses = []
    
    for arr in list_of_kps_arrays:
        arr = np.asarray(arr).astype(np.float32).copy()
        
        if arr.ndim == 1 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        
        if arr.shape[1] == 3:
            # Detect if pixel coords
            if (arr[:, 0].max() > 1.5) or (arr[:, 1].max() > 1.5):
                arr[:, 0] = arr[:, 0] / float(canvas_W)
                arr[:, 1] = arr[:, 1] / float(canvas_H)
        
        kps = []
        for x, y, c in arr:
            if c <= 0.0:
                kps.append(None)
            else:
                kps.append(SimpleKeypoint(float(x), float(y), float(c)))
        
        body = SimpleBodyResult(kps)
        pose = SimplePoseResult(body=body, left_hand=None, right_hand=None, face=None)
        poses.append(pose)
    
    return poses
