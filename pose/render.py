"""
pose/render.py

Pose drawing and rendering functions.
"""

from typing import List, Optional, Sequence, Tuple, Any, Dict
import numpy as np
import cv2
from PIL import Image

from .keypoints import (
    SimpleKeypoint,
    SimplePoseResult,
    extract_kps_array,
    normalize_keypoints_array,
    build_simple_poses_from_kps_arrays,
)
from .utils import safe_HWC3, safe_resize_image_with_pad, HWC3, resize_image_with_pad

# Try to import original DWPose drawing code if available
try:
    from custom_controlnet_aux.dwpose import draw_poses as dw_draw_poses
    from custom_controlnet_aux.dwpose.types import (
        Keypoint as RepoKeypoint,
        BodyResult as RepoBodyResult,
        PoseResult as RepoPoseResult,
    )
    DWPOSE_DRAW_AVAILABLE = True
except Exception:
    DWPOSE_DRAW_AVAILABLE = False


# Body limb sequence (18-point layout, same as DWPose/OpenPose)
BODY_LIMB_SEQ = [
    [2, 3], [2, 6], [3, 4], [4, 5],
    [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14],
    [2, 1], [1, 15], [15, 17], [1, 16],
    [16, 18],
]

# Colors for body limbs
BODY_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
]

# Hand edges
HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]


def draw_bodypose_fallback(
    canvas: np.ndarray,
    keypoints: List[Optional[SimpleKeypoint]],
    xinsr_stick_scaling: bool = False,
) -> np.ndarray:
    """
    Draw body limbs and joints on canvas.
    
    Args:
        canvas: HxWx3 uint8 array
        keypoints: List of keypoints (normalized 0..1 coords)
        xinsr_stick_scaling: Unused, kept for API compatibility
        
    Returns:
        Canvas with body pose drawn.
    """
    if keypoints is None:
        return canvas
    
    H, W, _ = canvas.shape
    
    # Draw limbs
    for idx, (a, b) in enumerate(BODY_LIMB_SEQ):
        if a - 1 < 0 or b - 1 < 0 or a - 1 >= len(keypoints) or b - 1 >= len(keypoints):
            continue
        
        p1 = keypoints[a - 1]
        p2 = keypoints[b - 1]
        
        if p1 is None or p2 is None:
            continue
        
        x1, y1 = int(p1.x * W), int(p1.y * H)
        x2, y2 = int(p2.x * W), int(p2.y * H)
        color = BODY_COLORS[idx % len(BODY_COLORS)]
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=3)
    
    # Draw joints
    for kp in keypoints:
        if kp is None:
            continue
        x, y = int(kp.x * W), int(kp.y * H)
        cv2.circle(canvas, (x, y), radius=4, color=(255, 255, 255), thickness=-1)
    
    return canvas


def draw_handpose_fallback(
    canvas: np.ndarray,
    keypoints: Optional[List[SimpleKeypoint]],
) -> np.ndarray:
    """Draw hand skeleton on canvas."""
    if keypoints is None:
        return canvas
    
    H, W, _ = canvas.shape
    
    for ie, (e1, e2) in enumerate(HAND_EDGES):
        if e1 >= len(keypoints) or e2 >= len(keypoints):
            continue
        
        k1, k2 = keypoints[e1], keypoints[e2]
        if k1 is None or k2 is None:
            continue
        
        x1, y1 = int(k1.x * W), int(k1.y * H)
        x2, y2 = int(k2.x * W), int(k2.y * H)
        
        # HSV-based color
        hue = int(ie / len(HAND_EDGES) * 179)
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color = tuple(cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist())
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
    
    for kp in keypoints:
        if kp is None:
            continue
        x, y = int(kp.x * W), int(kp.y * H)
        cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)
    
    return canvas


def draw_facepose_fallback(
    canvas: np.ndarray,
    keypoints: Optional[List[SimpleKeypoint]],
) -> np.ndarray:
    """Draw face keypoints on canvas."""
    if keypoints is None:
        return canvas
    
    H, W, _ = canvas.shape
    
    for kp in keypoints:
        if kp is None:
            continue
        x, y = int(kp.x * W), int(kp.y * H)
        cv2.circle(canvas, (x, y), 2, (255, 255, 255), -1)
    
    return canvas


def draw_poses_fallback(
    poses: List[SimplePoseResult],
    H: int,
    W: int,
    draw_body: bool = True,
    draw_hand: bool = True,
    draw_face: bool = True,
    xinsr_stick_scaling: bool = False,
) -> np.ndarray:
    """
    Draw all poses on a blank canvas.
    
    Args:
        poses: List of SimplePoseResult objects
        H: Canvas height
        W: Canvas width
        draw_body: Whether to draw body pose
        draw_hand: Whether to draw hand poses
        draw_face: Whether to draw face keypoints
        xinsr_stick_scaling: Unused, kept for API compatibility
        
    Returns:
        HxWx3 uint8 canvas with poses drawn.
    """
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    
    for pose in poses:
        if draw_body:
            canvas = draw_bodypose_fallback(canvas, pose.body.keypoints, xinsr_stick_scaling)
        
        if draw_hand:
            if pose.left_hand is not None:
                canvas = draw_handpose_fallback(canvas, pose.left_hand)
            if pose.right_hand is not None:
                canvas = draw_handpose_fallback(canvas, pose.right_hand)
        
        if draw_face and pose.face is not None:
            canvas = draw_facepose_fallback(canvas, pose.face)
    
    return canvas


def render_keypoint_frames_to_image(
    frames_meta: Sequence[Any],
    canvas_H: int,
    canvas_W: int,
    detect_resolution: int = 512,
    upscale_method: str = "INTER_CUBIC",
    output_type: str = "pil",
    draw_body: bool = True,
    draw_hand: bool = True,
    draw_face: bool = True,
    xinsr_stick_scaling: bool = False,
    prefer_repo_draw: bool = True,
) -> Tuple[Any, List[Dict]]:
    """
    Render keypoint frames to images.
    
    Args:
        frames_meta: Sequence of frame metas (dicts or objects) or a single meta
        canvas_H, canvas_W: Canvas dimensions for keypoint normalization
        detect_resolution: Resolution for output (0 = no resize)
        upscale_method: OpenCV interpolation method
        output_type: "pil" or "np"
        draw_body, draw_hand, draw_face: What to draw
        xinsr_stick_scaling: Pass to drawing functions
        prefer_repo_draw: Use DWPose drawing if available
        
    Returns:
        (images, openpose_dicts): Rendered images and OpenPose-format dicts
    """
    # Accept single meta as well
    if not isinstance(frames_meta, (list, tuple)):
        frames_meta = [frames_meta]
    
    openpose_dicts = []
    images = []

    for meta in frames_meta:
        kps = extract_kps_array(meta)
        
        # Get canvas dimensions from meta if available
        if isinstance(meta, dict) and "canvas_height" in meta and "canvas_width" in meta:
            cH = int(meta["canvas_height"])
            cW = int(meta["canvas_width"])
        else:
            cH, cW = canvas_H, canvas_W
        
        if kps is None:
            # Produce empty canvas
            canvas = np.zeros((cH, cW, 3), dtype=np.uint8)
            openpose_dicts.append({"people": [], "canvas_height": cH, "canvas_width": cW})
            
            padded, remove_pad = safe_resize_image_with_pad(canvas, detect_resolution, upscale_method)
            cropped = remove_pad(padded)
            final = safe_HWC3(cropped)
            images.append(Image.fromarray(final) if output_type == "pil" else final)
            continue

        # Normalize keypoints
        kps_norm = normalize_keypoints_array(kps, cH, cW)
        poses_for_frame = build_simple_poses_from_kps_arrays([kps_norm], cH, cW)

        # Try DWPose drawing if available and preferred
        if DWPOSE_DRAW_AVAILABLE and prefer_repo_draw:
            try:
                repo_keypoints = []
                for x, y, c in kps_norm:
                    if c <= 0.0:
                        repo_keypoints.append(None)
                    else:
                        repo_keypoints.append(RepoKeypoint(x=float(x), y=float(y), score=float(c), id=-1))
                
                repo_body = RepoBodyResult(
                    keypoints=repo_keypoints,
                    total_score=float(np.nansum(kps_norm[:, 2])),
                    total_parts=sum(1 for p in repo_keypoints if p is not None),
                )
                repo_pose = RepoPoseResult(body=repo_body, left_hand=None, right_hand=None, face=None)
                
                repo_canvas = dw_draw_poses(
                    [repo_pose], cH, cW,
                    draw_body=draw_body, draw_hand=draw_hand, draw_face=draw_face,
                    xinsr_stick_scaling=xinsr_stick_scaling,
                )
                
                padded, remove_pad = safe_resize_image_with_pad(repo_canvas, detect_resolution, upscale_method)
                cropped = remove_pad(padded)
                final = safe_HWC3(cropped)
                images.append(Image.fromarray(final) if output_type == "pil" else final)
                
                # Build OpenPose dict
                people_list = [{
                    "pose_keypoints_2d": [float(v) for trip in kps for v in trip],
                    "face_keypoints_2d": None,
                    "hand_left_keypoints_2d": None,
                    "hand_right_keypoints_2d": None,
                }]
                openpose_dicts.append({"people": people_list, "canvas_height": cH, "canvas_width": cW})
                continue
            except Exception:
                pass  # Fall through to fallback

        # Use fallback drawing
        canvas = draw_poses_fallback(
            poses_for_frame, cH, cW,
            draw_body=draw_body, draw_hand=draw_hand, draw_face=draw_face,
            xinsr_stick_scaling=xinsr_stick_scaling,
        )
        
        padded, remove_pad = safe_resize_image_with_pad(canvas, detect_resolution, upscale_method)
        cropped = remove_pad(padded)
        final = safe_HWC3(cropped)
        images.append(Image.fromarray(final) if output_type == "pil" else final)
        
        openpose_dicts.append({
            "people": [{"pose_keypoints_2d": [float(v) for trip in kps for v in trip]}],
            "canvas_height": cH,
            "canvas_width": cW,
        })

    # Return single image if single meta was provided
    if len(images) == 1:
        return images[0], openpose_dicts
    return images, openpose_dicts


def render_single_meta_to_image(
    meta: Any,
    canvas_H: int,
    canvas_W: int,
    detect_resolution: int = 512,
    upscale_method: str = "INTER_CUBIC",
    output_type: str = "pil",
    draw_body: bool = True,
    draw_hand: bool = True,
    draw_face: bool = True,
    xinsr_stick_scaling: bool = False,
):
    """
    Convenience wrapper for rendering a single frame.
    
    Returns:
        (image, openpose_dict): Rendered image and OpenPose-format dict
    """
    img, dicts = render_keypoint_frames_to_image(
        [meta], canvas_H, canvas_W,
        detect_resolution=detect_resolution,
        upscale_method=upscale_method,
        output_type=output_type,
        draw_body=draw_body, draw_hand=draw_hand, draw_face=draw_face,
        xinsr_stick_scaling=xinsr_stick_scaling,
    )
    return img, (dicts[0] if isinstance(dicts, list) and len(dicts) > 0 else None)
