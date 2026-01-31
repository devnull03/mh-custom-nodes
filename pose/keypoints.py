"""
pose/keypoints.py

Keypoint extraction, normalization, and data structures for pose handling.
"""

from typing import List, Optional, Any, Tuple
import numpy as np


class SimpleKeypoint:
    def __init__(self, x: float, y: float, score: float = 1.0):
        self.x = x
        self.y = y
        self.score = score


class SimpleBodyResult:
    def __init__(self, keypoints: List[Optional[SimpleKeypoint]]):
        self.keypoints = keypoints
        self.total_score = float(sum([kp.score if kp is not None else 0.0 for kp in keypoints]))
        self.total_parts = sum([1 for kp in keypoints if kp is not None])


class SimplePoseResult:
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


def _extract_kps_from_source(source: Any, key_attr: str, key_dict: str) -> Optional[np.ndarray]:
    """Extract keypoints from either an object attribute or dict key."""
    arr = None
    
    if hasattr(source, key_attr):
        val = getattr(source, key_attr)
        if val is not None:
            arr = np.asarray(val)
    elif isinstance(source, dict) and key_dict in source:
        val = source[key_dict]
        if val is not None:
            arr = np.asarray(val)
    
    if arr is None or arr.size == 0:
        return None
    
    if arr.ndim == 1 and arr.size % 3 == 0:
        arr = arr.reshape(-1, 3)
    
    if arr.ndim == 2 and arr.shape[1] == 2:
        arr = np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)
    
    return arr.astype(np.float32)


def extract_all_keypoints(meta: Any) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Extract all keypoint types (body, left_hand, right_hand, face) from frame meta."""
    if meta is None:
        return None, None, None, None
    
    body_kps = None
    left_hand_kps = None
    right_hand_kps = None
    face_kps = None
    
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
        
        body_kps = np.concatenate([kps_body[:, :2], probs[:, None]], axis=1).astype(np.float32)
    else:
        body_kps = _extract_kps_from_source(meta, "keypoints_body", "keypoints_body")
    
    left_hand_kps = _extract_kps_from_source(meta, "keypoints_left_hand", "keypoints_left_hand")
    right_hand_kps = _extract_kps_from_source(meta, "keypoints_right_hand", "keypoints_right_hand")
    face_kps = _extract_kps_from_source(meta, "keypoints_face", "keypoints_face")
    
    if left_hand_kps is None:
        left_hand_kps = _extract_kps_from_source(meta, "kps_left_hand", "kps_left_hand")
    if right_hand_kps is None:
        right_hand_kps = _extract_kps_from_source(meta, "kps_right_hand", "kps_right_hand")
    if face_kps is None:
        face_kps = _extract_kps_from_source(meta, "kps_face", "kps_face")
    
    if isinstance(meta, dict) and "people" in meta and len(meta.get("people", [])) > 0:
        first = meta["people"][0]
        
        if body_kps is None and "pose_keypoints_2d" in first:
            arr = np.asarray(first["pose_keypoints_2d"])
            if arr.ndim == 1 and arr.size % 3 == 0:
                arr = arr.reshape(-1, 3)
            body_kps = arr.astype(np.float32)
        
        if left_hand_kps is None and "hand_left_keypoints_2d" in first:
            val = first["hand_left_keypoints_2d"]
            if val is not None:
                arr = np.asarray(val)
                if arr.size > 0:
                    if arr.ndim == 1 and arr.size % 3 == 0:
                        arr = arr.reshape(-1, 3)
                    left_hand_kps = arr.astype(np.float32)
        
        if right_hand_kps is None and "hand_right_keypoints_2d" in first:
            val = first["hand_right_keypoints_2d"]
            if val is not None:
                arr = np.asarray(val)
                if arr.size > 0:
                    if arr.ndim == 1 and arr.size % 3 == 0:
                        arr = arr.reshape(-1, 3)
                    right_hand_kps = arr.astype(np.float32)
        
        if face_kps is None and "face_keypoints_2d" in first:
            val = first["face_keypoints_2d"]
            if val is not None:
                arr = np.asarray(val)
                if arr.size > 0:
                    if arr.ndim == 1 and arr.size % 3 == 0:
                        arr = arr.reshape(-1, 3)
                    face_kps = arr.astype(np.float32)
    
    return body_kps, left_hand_kps, right_hand_kps, face_kps


def extract_kps_array(meta: Any) -> Optional[np.ndarray]:
    """Extract body keypoints only (legacy compatibility)."""
    body_kps, _, _, _ = extract_all_keypoints(meta)
    return body_kps


def normalize_keypoints_array(
    kps: np.ndarray,
    canvas_H: int,
    canvas_W: int,
    expected_normalized: Optional[bool] = None,
) -> Optional[np.ndarray]:
    """Normalize keypoints to 0..1 range if they appear to be pixel coordinates."""
    if kps is None:
        return None
    
    arr = np.asarray(kps).astype(np.float32).copy()
    
    if arr.size == 0:
        return arr.reshape(0, 3)
    
    if arr.ndim == 1 and arr.size % 3 == 0:
        arr = arr.reshape(-1, 3)
    
    xs = arr[:, 0]
    ys = arr[:, 1]
    
    if expected_normalized is None:
        need_norm = (np.nanmax(xs) > 1.5) or (np.nanmax(ys) > 1.5)
    else:
        need_norm = not expected_normalized
    
    if need_norm:
        if canvas_W <= 0 or canvas_H <= 0:
            raise ValueError("canvas_H and canvas_W must be > 0 to normalize pixel coords")
        arr[:, 0] = arr[:, 0] / float(canvas_W)
        arr[:, 1] = arr[:, 1] / float(canvas_H)
    
    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)
    
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=np.float32)], axis=1)
    
    return arr


def _array_to_keypoint_list(arr: Optional[np.ndarray], canvas_H: int, canvas_W: int) -> Optional[List[SimpleKeypoint]]:
    """Convert Nx3 array to list of SimpleKeypoint, normalizing if needed."""
    if arr is None or arr.size == 0:
        return None
    
    arr = normalize_keypoints_array(arr, canvas_H, canvas_W)
    if arr is None:
        return None
    
    kps = []
    for x, y, c in arr:
        if c <= 0.0:
            kps.append(None)
        else:
            kps.append(SimpleKeypoint(float(x), float(y), float(c)))
    
    return kps if len(kps) > 0 else None


def build_pose_from_meta(meta: Any, canvas_H: int, canvas_W: int) -> SimplePoseResult:
    """Build a SimplePoseResult from meta, extracting body, hands, and face."""
    body_kps, left_hand_kps, right_hand_kps, face_kps = extract_all_keypoints(meta)
    
    body_list = _array_to_keypoint_list(body_kps, canvas_H, canvas_W)
    if body_list is None:
        body_list = []
    body = SimpleBodyResult(body_list)
    
    left_hand = _array_to_keypoint_list(left_hand_kps, canvas_H, canvas_W)
    right_hand = _array_to_keypoint_list(right_hand_kps, canvas_H, canvas_W)
    face = _array_to_keypoint_list(face_kps, canvas_H, canvas_W)
    
    return SimplePoseResult(body=body, left_hand=left_hand, right_hand=right_hand, face=face)


def build_simple_poses_from_kps_arrays(
    list_of_kps_arrays: List[np.ndarray],
    canvas_H: int,
    canvas_W: int,
) -> List[SimplePoseResult]:
    """Convert a list of Nx3 body keypoint arrays into SimplePoseResult list (body only)."""
    poses = []
    
    for arr in list_of_kps_arrays:
        arr = np.asarray(arr).astype(np.float32).copy()
        
        if arr.ndim == 1 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        
        if arr.shape[1] == 3:
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
