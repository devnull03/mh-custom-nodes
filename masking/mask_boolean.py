import math

import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, device) -> torch.Tensor:
    radius = int(math.ceil(2 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur_mask(
    mask: torch.Tensor, radius: float, padding_mode: str = "replicate"
) -> torch.Tensor:
    if radius is None or radius <= 0:
        return mask
    sigma = float(radius)
    if sigma < 1e-3:
        return mask

    if padding_mode not in {"constant", "replicate", "reflect"}:
        raise ValueError(
            f"Invalid padding_mode: {padding_mode}. Use constant, replicate, or reflect."
        )

    kernel = _gaussian_kernel1d(sigma, mask.device)
    pad = kernel.numel() // 2

    mask_f = mask.float().unsqueeze(1)  # [B,1,H,W]
    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)

    if pad > 0:
        mask_f = F.pad(mask_f, (pad, pad, 0, 0), mode=padding_mode)
    mask_f = F.conv2d(mask_f, kernel_x, padding=0)

    if pad > 0:
        mask_f = F.pad(mask_f, (0, 0, pad, pad), mode=padding_mode)
    mask_f = F.conv2d(mask_f, kernel_y, padding=0)

    return mask_f.squeeze(1).clamp(0.0, 1.0)


class mh_MaskSubtract:
    """
    Subtracts mask_b from mask_a and clamps the result into [0, 1].
    Useful for removing overlapping regions (e.g., remove a ball from a person mask).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
            },
            "optional": {
                "blur_radius": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5},
                ),
                "blur_padding": (
                    ["constant", "replicate", "reflect"],
                    {"default": "replicate"},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "subtract"
    CATEGORY = "MH/Mask"

    def subtract(self, mask_a, mask_b, blur_radius=0.0, blur_padding="replicate"):
        # Ensure 3D tensors [B, H, W]
        if mask_a.dim() == 2:
            mask_a = mask_a.unsqueeze(0)
        if mask_b.dim() == 2:
            mask_b = mask_b.unsqueeze(0)

        # Broadcast single-frame mask to batch if needed
        if mask_a.shape[0] != mask_b.shape[0]:
            if mask_a.shape[0] == 1:
                mask_a = mask_a.repeat(mask_b.shape[0], 1, 1)
            elif mask_b.shape[0] == 1:
                mask_b = mask_b.repeat(mask_a.shape[0], 1, 1)
            else:
                raise ValueError(
                    f"Batch sizes differ and cannot be broadcast: {mask_a.shape} vs {mask_b.shape}"
                )

        # Validate spatial dimensions
        if mask_a.shape[1:] != mask_b.shape[1:]:
            raise ValueError(
                f"Mask spatial dimensions must match: {mask_a.shape[1:]} vs {mask_b.shape[1:]}"
            )

        original_dtype = mask_a.dtype
        if blur_radius and blur_radius > 0:
            mask_b = _gaussian_blur_mask(mask_b, blur_radius, blur_padding)

        result = (mask_a.float() - mask_b.float()).clamp(0.0, 1.0)
        result = result.to(original_dtype)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "mh_MaskSubtract": mh_MaskSubtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mh_MaskSubtract": "mh Mask Subtract",
}
