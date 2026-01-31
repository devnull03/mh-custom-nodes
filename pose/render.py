"""
pose/render.py - Pose drawing and rendering functions.
"""

from typing import List, Optional, Sequence, Tuple, Any, Dict
import numpy as np
import cv2
from PIL import Image

from .keypoints import (
    SimpleKeypoint,
    SimplePoseResult,
    extract_kps_array,
    extract_all_keypoints,
    normalize_keypoints_array,
    build_simple_poses_from_kps_arrays,
    build_pose_from_meta,
)
from .utils import safe_HWC3, safe_resize_image_with_pad

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


BODY_LIMB_SEQ = [
    [2, 3], [2, 6], [3, 4], [4, 5],
    [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14],
    [2, 1], [1, 15], [15, 17], [1, 16],
    [16, 18],
]

BODY_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
]

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
    if keypoints is None:
        return canvas
    
    H, W, _ = canvas.shape
    
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
    if not isinstance(frames_meta, (list, tuple)):
        frames_meta = [frames_meta]
    
    openpose_dicts = []
    images = []

    for meta in frames_meta:
        if isinstance(meta, dict) and "canvas_height" in meta and "canvas_width" in meta:
            cH = int(meta["canvas_height"])
            cW = int(meta["canvas_width"])
        else:
            cH, cW = canvas_H, canvas_W
        
        body_kps, left_hand_kps, right_hand_kps, face_kps = extract_all_keypoints(meta)
        
        if body_kps is None:
            canvas = np.zeros((cH, cW, 3), dtype=np.uint8)
            openpose_dicts.append({"people": [], "canvas_height": cH, "canvas_width": cW})
            
            padded, remove_pad = safe_resize_image_with_pad(canvas, detect_resolution, upscale_method)
            cropped = remove_pad(padded)
            final = safe_HWC3(cropped)
            images.append(Image.fromarray(final) if output_type == "pil" else final)
            continue

        pose = build_pose_from_meta(meta, cH, cW)

        if DWPOSE_DRAW_AVAILABLE and prefer_repo_draw:
            try:
                kps_norm = normalize_keypoints_array(body_kps, cH, cW)
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
                
                repo_left_hand = None
                repo_right_hand = None
                repo_face = None
                
                if left_hand_kps is not None:
                    lh_norm = normalize_keypoints_array(left_hand_kps, cH, cW)
                    repo_left_hand = [
                        RepoKeypoint(x=float(x), y=float(y), score=float(c), id=-1) if c > 0 else None
                        for x, y, c in lh_norm
                    ]
                
                if right_hand_kps is not None:
                    rh_norm = normalize_keypoints_array(right_hand_kps, cH, cW)
                    repo_right_hand = [
                        RepoKeypoint(x=float(x), y=float(y), score=float(c), id=-1) if c > 0 else None
                        for x, y, c in rh_norm
                    ]
                
                if face_kps is not None:
                    face_norm = normalize_keypoints_array(face_kps, cH, cW)
                    repo_face = [
                        RepoKeypoint(x=float(x), y=float(y), score=float(c), id=-1) if c > 0 else None
                        for x, y, c in face_norm
                    ]
                
                repo_pose = RepoPoseResult(
                    body=repo_body,
                    left_hand=repo_left_hand,
                    right_hand=repo_right_hand,
                    face=repo_face,
                )
                
                repo_canvas = dw_draw_poses(
                    [repo_pose], cH, cW,
                    draw_body=draw_body, draw_hand=draw_hand, draw_face=draw_face,
                    xinsr_stick_scaling=xinsr_stick_scaling,
                )
                
                padded, remove_pad = safe_resize_image_with_pad(repo_canvas, detect_resolution, upscale_method)
                cropped = remove_pad(padded)
                final = safe_HWC3(cropped)
                images.append(Image.fromarray(final) if output_type == "pil" else final)
                
                people_list = [{
                    "pose_keypoints_2d": [float(v) for trip in body_kps for v in trip],
                    "face_keypoints_2d": [float(v) for trip in face_kps for v in trip] if face_kps is not None else None,
                    "hand_left_keypoints_2d": [float(v) for trip in left_hand_kps for v in trip] if left_hand_kps is not None else None,
                    "hand_right_keypoints_2d": [float(v) for trip in right_hand_kps for v in trip] if right_hand_kps is not None else None,
                }]
                openpose_dicts.append({"people": people_list, "canvas_height": cH, "canvas_width": cW})
                continue
            except Exception:
                pass

        canvas = draw_poses_fallback(
            [pose], cH, cW,
            draw_body=draw_body, draw_hand=draw_hand, draw_face=draw_face,
            xinsr_stick_scaling=xinsr_stick_scaling,
        )
        
        padded, remove_pad = safe_resize_image_with_pad(canvas, detect_resolution, upscale_method)
        cropped = remove_pad(padded)
        final = safe_HWC3(cropped)
        images.append(Image.fromarray(final) if output_type == "pil" else final)
        
        openpose_dicts.append({
            "people": [{
                "pose_keypoints_2d": [float(v) for trip in body_kps for v in trip],
                "face_keypoints_2d": [float(v) for trip in face_kps for v in trip] if face_kps is not None else None,
                "hand_left_keypoints_2d": [float(v) for trip in left_hand_kps for v in trip] if left_hand_kps is not None else None,
                "hand_right_keypoints_2d": [float(v) for trip in right_hand_kps for v in trip] if right_hand_kps is not None else None,
            }],
            "canvas_height": cH,
            "canvas_width": cW,
        })

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
    img, dicts = render_keypoint_frames_to_image(
        [meta], canvas_H, canvas_W,
        detect_resolution=detect_resolution,
        upscale_method=upscale_method,
        output_type=output_type,
        draw_body=draw_body, draw_hand=draw_hand, draw_face=draw_face,
        xinsr_stick_scaling=xinsr_stick_scaling,
    )
    return img, (dicts[0] if isinstance(dicts, list) and len(dicts) > 0 else None)
