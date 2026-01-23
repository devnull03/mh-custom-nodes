import torch


class mh_MaskMinimalCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA", "BOX")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_data", "box")
    FUNCTION = "crop"
    CATEGORY = "MH/Crop"

    def crop(self, image, mask, padding=0):
        # image shape: (B, H, W, C)
        # mask shape: (B, H, W) or (H, W)

        # Ensure mask is 3D (B, H, W)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        # Use first mask to determine bounding box (applies same crop to all batch items)
        rows, cols = torch.nonzero(mask[0], as_tuple=True)

        if len(rows) == 0:
            # Fallback: No mask found, return original
            print(
                "Warning: MaskMinimalCrop found no mask pixels. Returning original image."
            )
            h, w = image.shape[1], image.shape[2]
            return (image, mask, ((w, h), (0, 0, w, h)), (0, 0, w, h))

        # Determine bounding box
        y_min, y_max = rows.min().item(), rows.max().item()
        x_min, x_max = cols.min().item(), cols.max().item()

        # Apply padding
        h, w = mask.shape[1], mask.shape[2]

        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding + 1)
        y_max = min(h, y_max + padding + 1)

        # Crop all batch items
        # Image is [B, H, W, C], Mask is [B, H, W]
        cropped_image = image[:, y_min:y_max, x_min:x_max, :]
        cropped_mask = mask[:, y_min:y_max, x_min:x_max]

        # Prepare CROP_DATA: ((original_width, original_height), (left, top, right, bottom))
        crop_data = ((w, h), (x_min, y_min, x_max, y_max))

        # Box format: (x1, y1, x2, y2)
        box = (x_min, y_min, x_max, y_max)

        return (cropped_image, cropped_mask, crop_data, box)


NODE_CLASS_MAPPINGS = {"mh_MaskMinimalCrop": mh_MaskMinimalCrop}

NODE_DISPLAY_NAME_MAPPINGS = {"mh_MaskMinimalCrop": "MH Mask Minimal Crop"}
