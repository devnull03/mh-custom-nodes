"""
pose/utils.py - Image utilities for pose rendering.
"""

import numpy as np
import cv2

try:
    from custom_controlnet_aux.util import (
        resize_image_with_pad as dw_resize_image_with_pad,
        HWC3 as dw_HWC3,
    )
    DWPOSE_UTILS_AVAILABLE = True
except Exception:
    DWPOSE_UTILS_AVAILABLE = False


UPSCALE_METHODS = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]


def get_upscale_method(method_str: str):
    if method_str not in UPSCALE_METHODS:
        raise ValueError(f"Method {method_str} not found in {UPSCALE_METHODS}")
    return getattr(cv2, method_str)


def HWC3(x: np.ndarray) -> np.ndarray:
    """
    Ensure image is uint8 HxWx3 format.
    
    Handles:
    - Grayscale (HxW or HxWx1) -> RGB
    - RGBA (HxWx4) -> RGB (alpha composited over white)
    """
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C in (1, 3, 4)
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    color = x[:, :, 0:3].astype(np.float32)
    alpha = x[:, :, 3:4].astype(np.float32) / 255.0
    y = color * alpha + 255.0 * (1.0 - alpha)
    return y.clip(0, 255).astype(np.uint8)


def pad64(x: int) -> int:
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def resize_image_with_pad(
    input_image: np.ndarray,
    resolution: int,
    upscale_method: str = "INTER_CUBIC",
    skip_hwc3: bool = False,
    mode: str = "edge",
):
    if not skip_hwc3:
        img = HWC3(input_image)
    else:
        img = input_image
    
    H_raw, W_raw, _ = img.shape
    
    if resolution == 0:
        return img, (lambda x: x)
    
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    
    interp = get_upscale_method(upscale_method) if k > 1 else cv2.INTER_AREA
    img = cv2.resize(img, (W_target, H_target), interpolation=interp)
    
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(x[:H_target, :W_target, ...]).copy()

    return np.ascontiguousarray(img_padded).copy(), remove_pad


def safe_HWC3(x: np.ndarray) -> np.ndarray:
    if DWPOSE_UTILS_AVAILABLE:
        return dw_HWC3(x)
    return HWC3(x)


def safe_resize_image_with_pad(
    input_image: np.ndarray,
    resolution: int,
    upscale_method: str = "INTER_CUBIC",
):
    if DWPOSE_UTILS_AVAILABLE:
        return dw_resize_image_with_pad(input_image, resolution, upscale_method)
    return resize_image_with_pad(input_image, resolution, upscale_method)
