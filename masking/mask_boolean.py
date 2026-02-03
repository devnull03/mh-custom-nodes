import torch


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
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "subtract"
    CATEGORY = "MH/Mask"

    def subtract(self, mask_a, mask_b):
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
        result = (mask_a.double() - mask_b.double()).clamp(0.0, 1.0)
        result = result.to(original_dtype)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "mh_MaskSubtract": mh_MaskSubtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mh_MaskSubtract": "mh Mask Subtract",
}
