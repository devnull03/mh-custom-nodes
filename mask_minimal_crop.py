import torch


class mh_MaskMinimalCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA", "BOX")
    RETURN_NAMES = ("images", "masks", "crop_data", "box")
    FUNCTION = "crop"
    CATEGORY = "MH/Crop"

    def crop(self, images, masks, padding=0):
        # images shape: (B, H, W, C)
        # masks shape: (B, H, W) or (H, W)

        # Ensure mask is 3D (B, H, W)
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)

        # Get original dimensions
        batch_size, img_height, img_width, channels = images.shape

        # Use first mask to determine bounding box (applies same crop to all batch items)
        rows, cols = torch.nonzero(masks[0], as_tuple=True)

        if len(rows) == 0:
            # Fallback: No mask found, return original
            print(
                "Warning: MaskMinimalCrop found no mask pixels. Returning original image."
            )
            crop_data = ((img_width, img_height), (0, 0, img_width, img_height))
            box = (0, 0, img_width, img_height)
            return (images, masks, crop_data, box)

        # Determine bounding box
        y_min, y_max = rows.min().item(), rows.max().item()
        x_min, x_max = cols.min().item(), cols.max().item()

        # Apply padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_width, x_max + padding + 1)
        y_max = min(img_height, y_max + padding + 1)

        # Crop all batch items
        # Image is [B, H, W, C], Mask is [B, H, W]
        cropped_images = images[:, y_min:y_max, x_min:x_max, :]
        cropped_masks = masks[:, y_min:y_max, x_min:x_max]

        # crop_data format: ((original_img_width, original_img_height), (left, top, right, bottom))
        crop_data = ((img_width, img_height), (x_min, y_min, x_max, y_max))

        # Box format: (x1, y1, x2, y2)
        box = (x_min, y_min, x_max, y_max)

        return (cropped_images, cropped_masks, crop_data, box)


NODE_CLASS_MAPPINGS = {"mh_MaskMinimalCrop": mh_MaskMinimalCrop}

NODE_DISPLAY_NAME_MAPPINGS = {"mh_MaskMinimalCrop": "MH Mask Minimal Crop"}
