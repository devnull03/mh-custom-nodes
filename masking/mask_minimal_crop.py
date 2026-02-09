import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

RESIZE_MODES = ["bicubic", "bilinear", "lanczos"]


def _tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    )


def _pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def _resize_bhwc(images, h, w, mode="bicubic"):
    """Resize [B,H,W,C] images. Supports bilinear, bicubic, lanczos."""
    if mode == "lanczos":
        results = []
        for i in range(images.shape[0]):
            pil_img = _tensor2pil(images[i])
            pil_img = pil_img.resize((w, h), Image.LANCZOS)
            results.append(_pil2tensor(pil_img))
        return torch.cat(results, dim=0)
    return F.interpolate(
        images.permute(0, 3, 1, 2),
        size=(h, w),
        mode=mode,
        align_corners=False,
        antialias=True,
    ).permute(0, 2, 3, 1)


def _resize_bhw(masks, h, w, mode="bicubic"):
    """Resize [B,H,W] masks. Supports bilinear, bicubic, lanczos."""
    if mode == "lanczos":
        results = []
        for i in range(masks.shape[0]):
            arr = (masks[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(arr, mode="L")
            pil_img = pil_img.resize((w, h), Image.LANCZOS)
            t = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            results.append(t.unsqueeze(0))
        return torch.cat(results, dim=0).clamp(0.0, 1.0)
    return (
        F.interpolate(
            masks.unsqueeze(1).float(),
            size=(h, w),
            mode=mode,
            align_corners=False,
            antialias=True,
        )
        .squeeze(1)
        .clamp(0.0, 1.0)
    )


def _round_to_divisible(value, divisor, max_value):
    """
    Round value to the nearest multiple of divisor that fits within max_value.
    Prefers rounding UP so we never shrink below the requested size.
    Falls back to rounding DOWN only when rounding up would exceed max_value.
    The result is always a multiple of divisor (minimum = divisor),
    unless divisor itself exceeds max_value, in which case max_value is returned.
    """
    if divisor <= 1:
        return min(max(1, int(value)), int(max_value))

    int_value = max(1, int(value))
    max_val = int(max_value)

    # If divisor is larger than the image dimension, just return max_value
    if divisor > max_val:
        return max_val

    # Try rounding up first — this preserves or expands the crop
    rounded_up = ((int_value + divisor - 1) // divisor) * divisor
    if rounded_up <= max_val:
        return rounded_up

    # Rounding up exceeds bounds — use the largest multiple that fits
    rounded_down = (max_val // divisor) * divisor
    return max(divisor, rounded_down)


def _clamp_box_to_image(x_min, y_min, width, height, img_w, img_h):
    """
    Clamp a box to image bounds. Ensures width/height do not exceed image size.
    Returns (x_min, y_min, x_max, y_max).
    """
    width = min(max(1, width), img_w)
    height = min(max(1, height), img_h)

    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_min + width > img_w:
        x_min = img_w - width
    if y_min + height > img_h:
        y_min = img_h - height

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    return x_min, y_min, x_min + width, y_min + height


def _make_bypass(img_w, img_h):
    """Create bypass crop_data and box for the full image."""
    crop_data = {
        "original_size": (img_w, img_h),
        "crop_box": (0, 0, img_w, img_h),
        "output_size": (img_w, img_h),
        "bypass": True,
    }
    return crop_data, (0, 0, img_w, img_h)


def _mask_bbox_from_batch(masks):
    """
    Per-frame bbox reduction — avoids materialising one huge nonzero tensor.
    Returns (x_min, y_min, x_max, y_max) or None if no mask pixels.
    """
    if len(masks.shape) == 2:
        masks = masks.unsqueeze(0)

    batch_size = masks.shape[0]
    y_min, y_max = masks.shape[1], -1
    x_min, x_max = masks.shape[2], -1

    for i in range(batch_size):
        nz = torch.nonzero(masks[i])
        if len(nz) == 0:
            continue
        y_min = min(y_min, nz[:, 0].min().item())
        y_max = max(y_max, nz[:, 0].max().item())
        x_min = min(x_min, nz[:, 1].min().item())
        x_max = max(x_max, nz[:, 1].max().item())

    if y_max < 0 or x_max < 0:
        return None

    return x_min, y_min, x_max, y_max


class mh_MaskMinimalCrop:
    """
    Static mask cropping — finds bounding box across ALL frames and applies same crop.
    Bypasses cropping when the mask covers most of the image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "divisible_by": (
                    "INT",
                    {"default": 16, "min": 1, "max": 256, "step": 1},
                ),
                "scale_mode": (["none", "original", "custom"], {"default": "none"}),
                "resolution": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
                "preserve_aspect": ("BOOLEAN", {"default": True}),
                "bypass_threshold": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "resize_mode": (RESIZE_MODES, {"default": "bicubic"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA", "BOX")
    RETURN_NAMES = ("images", "masks", "crop_data", "box")
    FUNCTION = "crop"
    CATEGORY = "MH/Crop"

    def crop(
        self,
        images,
        masks,
        padding=0,
        divisible_by=16,
        scale_mode="none",
        resolution=512,
        preserve_aspect=True,
        bypass_threshold=0.9,
        resize_mode="bicubic",
    ):
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)

        batch_size, img_h, img_w, _ = images.shape

        bbox = _mask_bbox_from_batch(masks)

        if bbox is None:
            print("Warning: MaskMinimalCrop found no mask pixels. Returning original.")
            crop_data, box = _make_bypass(img_w, img_h)
            return (images, masks, crop_data, box)

        x_min, y_min, x_max, y_max = bbox

        # Apply padding (x_max/y_max are inclusive pixel indices from bbox,
        # so +1 converts to exclusive, then +padding extends further)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_w, x_max + padding + 1)
        y_max = min(img_h, y_max + padding + 1)

        crop_w = x_max - x_min
        crop_h = y_max - y_min
        area_ratio = (crop_w * crop_h) / (img_w * img_h)

        # Bypass if crop covers most of the image
        if area_ratio >= bypass_threshold:
            print(
                f"[MaskMinimalCrop] Crop area {area_ratio:.1%} >= threshold {bypass_threshold:.1%}, bypassing."
            )
            crop_data, box = _make_bypass(img_w, img_h)
            return (images, masks, crop_data, box)

        # Compute crop dimensions
        if preserve_aspect:
            orig_aspect = img_w / img_h
            if crop_w / crop_h < orig_aspect:
                new_w = int(crop_h * orig_aspect)
                new_h = crop_h
            else:
                new_h = int(crop_w / orig_aspect)
                new_w = crop_w
        else:
            new_w, new_h = crop_w, crop_h

        # Round to divisible — prefers rounding UP so we never shrink below
        # the mask bounding box. Only rounds down if rounding up would exceed
        # the image dimension.
        new_w = _round_to_divisible(new_w, divisible_by, img_w)
        new_h = _round_to_divisible(new_h, divisible_by, img_h)

        # Center the crop on the bounding box center, clamped to image
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        x_min, y_min, x_max, y_max = _clamp_box_to_image(
            cx - new_w // 2, cy - new_h // 2, new_w, new_h, img_w, img_h
        )

        # Crop
        cropped_images = images[:, y_min:y_max, x_min:x_max, :]
        cropped_masks = masks[:, y_min:y_max, x_min:x_max]

        # Optional scaling
        if scale_mode != "none":
            target_w, target_h = self._compute_scale_target(
                scale_mode,
                cropped_images.shape[2],
                cropped_images.shape[1],
                img_w,
                img_h,
                resolution,
                preserve_aspect,
                divisible_by,
            )
            cropped_images = _resize_bhwc(
                cropped_images, target_h, target_w, resize_mode
            )
            cropped_masks = _resize_bhw(cropped_masks, target_h, target_w, resize_mode)

        out_w = int(cropped_images.shape[2])
        out_h = int(cropped_images.shape[1])

        crop_data = {
            "original_size": (img_w, img_h),
            "crop_box": (x_min, y_min, x_max, y_max),
            "output_size": (out_w, out_h),
            "bypass": False,
        }
        box = (x_min, y_min, x_max, y_max)
        print(f"[MaskMinimalCrop] Crop area {area_ratio:.1%}, box: {box}")
        return (cropped_images, cropped_masks, crop_data, box)

    @staticmethod
    def _compute_scale_target(
        mode, crop_w, crop_h, img_w, img_h, resolution, preserve_aspect, divisible_by
    ):
        if mode == "original":
            return img_w, img_h

        # Calculate the scale factor from the longest side so we can apply it
        # uniformly to both dimensions, preserving aspect ratio.
        if preserve_aspect:
            if crop_w >= crop_h:
                scale = resolution / crop_w
            else:
                scale = resolution / crop_h
            target_w = max(1, round(crop_w * scale))
            target_h = max(1, round(crop_h * scale))

            # Round both to divisible, then correct the shorter side to keep
            # the aspect ratio as close as possible.
            target_w = _round_to_divisible(target_w, divisible_by, resolution * 2)
            # Re-derive height from the rounded width to maintain AR
            target_h = max(1, round(target_w * crop_h / crop_w))
            target_h = _round_to_divisible(target_h, divisible_by, resolution * 2)
        else:
            target_w = target_h = resolution
            target_w = _round_to_divisible(target_w, divisible_by, resolution * 2)
            target_h = _round_to_divisible(target_h, divisible_by, resolution * 2)

        return target_w, target_h


class mh_MaskTrackingCrop:
    """
    Dynamic per-frame mask cropping with tracking.
    Uses the maximum bounding box across all frames for consistent output size,
    then crops each frame centered on its own mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "padding": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
                "divisible_by": (
                    "INT",
                    {"default": 16, "min": 1, "max": 256, "step": 1},
                ),
                "smoothing": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "scale_mode": (
                    "STRING",
                    {"default": "none", "choices": ["none", "original", "custom"]},
                ),
                "resolution": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
                "resize_mode": (RESIZE_MODES, {"default": "bicubic"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA_BATCH", "INT", "INT")
    RETURN_NAMES = ("images", "masks", "crop_data_batch", "crop_width", "crop_height")
    FUNCTION = "crop"
    CATEGORY = "MH/Crop"

    def crop(
        self,
        images,
        masks,
        padding=32,
        divisible_by=16,
        smoothing=0.0,
        scale_mode="none",
        resolution=512,
        resize_mode="bicubic",
    ):
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)

        batch_size, img_h, img_w, _ = images.shape

        # Per-frame bounding boxes: (center_x, center_y, width, height)
        per_frame_boxes = []
        max_w = max_h = 0

        for i in range(batch_size):
            frame_mask = masks[i] if i < masks.shape[0] else masks[-1]
            nz = torch.nonzero(frame_mask)

            if len(nz) == 0:
                per_frame_boxes.append((img_w // 2, img_h // 2, 0, 0))
            else:
                y_min, y_max = nz[:, 0].min().item(), nz[:, 0].max().item()
                x_min, x_max = nz[:, 1].min().item(), nz[:, 1].max().item()
                bw, bh = x_max - x_min + 1, y_max - y_min + 1
                per_frame_boxes.append(
                    ((x_min + x_max) // 2, (y_min + y_max) // 2, bw, bh)
                )
                max_w = max(max_w, bw)
                max_h = max(max_h, bh)

        # Fallback if all masks empty
        if max_w == 0 or max_h == 0:
            print("Warning: MaskTrackingCrop found no mask pixels in any frame.")
            max_w, max_h = img_w // 4, img_h // 4

        # Output size from max bbox + padding, rounded to divisible and clamped
        out_w = _round_to_divisible(max_w + padding * 2, divisible_by, img_w)
        out_h = _round_to_divisible(max_h + padding * 2, divisible_by, img_h)

        # Temporal smoothing
        smoothed_centers = []
        prev_cx = prev_cy = None
        for cx, cy, _, _ in per_frame_boxes:
            if smoothing > 0 and prev_cx is not None:
                cx = int(prev_cx * smoothing + cx * (1 - smoothing))
                cy = int(prev_cy * smoothing + cy * (1 - smoothing))
            smoothed_centers.append((cx, cy))
            prev_cx, prev_cy = cx, cy

        # Per-frame crops
        cropped_images_list = []
        cropped_masks_list = []
        crop_regions = []
        half_w, half_h = out_w // 2, out_h // 2

        for i in range(batch_size):
            cx, cy = smoothed_centers[i]
            x_min, y_min, x_max, y_max = _clamp_box_to_image(
                cx - half_w, cy - half_h, out_w, out_h, img_w, img_h
            )
            crop_regions.append((x_min, y_min, x_max, y_max))
            cropped_images_list.append(images[i : i + 1, y_min:y_max, x_min:x_max, :])
            mask_idx = min(i, masks.shape[0] - 1)
            cropped_masks_list.append(
                masks[mask_idx : mask_idx + 1, y_min:y_max, x_min:x_max]
            )

        cropped_images = torch.cat(cropped_images_list, dim=0)
        cropped_masks = torch.cat(cropped_masks_list, dim=0)

        # Optional scaling
        if scale_mode != "none":
            if scale_mode == "original":
                target_w, target_h = img_w, img_h
            else:
                # Scale to resolution preserving aspect ratio
                if out_w >= out_h:
                    scale = resolution / out_w
                else:
                    scale = resolution / out_h
                target_w = max(1, round(out_w * scale))
                target_h = max(1, round(out_h * scale))

                # Round the longer side, then re-derive shorter to keep AR
                target_w = _round_to_divisible(target_w, divisible_by, resolution * 2)
                target_h = max(1, round(target_w * out_h / out_w))
                target_h = _round_to_divisible(target_h, divisible_by, resolution * 2)

            cropped_images = _resize_bhwc(
                cropped_images, target_h, target_w, resize_mode
            )
            cropped_masks = _resize_bhw(cropped_masks, target_h, target_w, resize_mode)
            final_size = (target_w, target_h)
        else:
            final_size = (out_w, out_h)

        crop_data_batch = {
            "original_size": (img_w, img_h),
            "output_size": final_size,
            "crop_regions": crop_regions,
        }
        return (cropped_images, cropped_masks, crop_data_batch, out_w, out_h)


NODE_CLASS_MAPPINGS = {
    "mh_MaskMinimalCrop": mh_MaskMinimalCrop,
    "mh_MaskTrackingCrop": mh_MaskTrackingCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mh_MaskMinimalCrop": "MH Mask Minimal Crop",
    "mh_MaskTrackingCrop": "MH Mask Tracking Crop",
}
