"""
MH node: compute absolute expand value (pixels) from a mask and a relative size change (%).
Useful for "Grow Mask With Blur" and similar nodes that take an expand in pixels:
input the mask and desired relative change (e.g. 0.5 for 50% larger, -0.5 for 50% smaller) to get the expand value.
"""
from __future__ import annotations

import torch


def _mask_area_and_perimeter(mask: torch.Tensor) -> tuple[float, float]:
    """Return (area, perimeter) for a 2D binary mask. Mask values > 0.5 are considered inside."""
    if mask.dim() == 3:
        mask = mask[0]
    m = (mask > 0.5).float()
    area = m.sum().item()
    if area <= 0:
        return 0.0, 0.0
    # 4-neighbor boundary: pixel is boundary iff it's 1 and at least one 4-neighbor is 0
    pad = torch.nn.functional.pad(
        m.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="constant", value=0.0
    ).squeeze()
    center = pad[1:-1, 1:-1]
    up, down = pad[:-2, 1:-1], pad[2:, 1:-1]
    left, right = pad[1:-1, :-2], pad[1:-1, 2:]
    eroded = torch.minimum(
        torch.minimum(torch.minimum(torch.minimum(center, up), down), left), right
    )
    boundary = (center - eroded).clamp(min=0.0)
    perimeter = boundary.sum().item()
    return area, perimeter


class mh_ExpandFromPercent:
    """
    Given a mask and a relative size change (fraction), returns the expand value in pixels
    to achieve that size increase or decrease (e.g. for Grow Mask With Blur).
    0.5 = 50% larger, -0.5 = 50% smaller.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "percent": (
                    "FLOAT",
                    {"default": 0.5, "min": -0.99, "max": 5.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("expand",)
    FUNCTION = "expand_from_percent"
    CATEGORY = "MH/Utils"

    def expand_from_percent(self, mask: torch.Tensor, percent: float) -> tuple[int,]:
        """
        Compute expand (pixels) so that expanding the mask by that amount
        changes its area by approximately the given fraction.

        percent: 0.5 = 50% larger, -0.5 = 50% smaller.
        Uses: expand ≈ (area * percent) / perimeter.
        """
        area, perimeter = _mask_area_and_perimeter(mask)
        if perimeter <= 0:
            return (0,)
        # delta_area = area * percent  =>  expand = delta_area / perimeter
        raw = area * percent / perimeter
        expand = int(round(raw))
        return (expand,)


NODE_CLASS_MAPPINGS = {"mh_ExpandFromPercent": mh_ExpandFromPercent}
NODE_DISPLAY_NAME_MAPPINGS = {"mh_ExpandFromPercent": "MH Expand From Percent"}
