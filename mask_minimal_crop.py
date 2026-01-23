import torch


class mh_MaskMinimalCrop:
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
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA", "BOX")
    RETURN_NAMES = ("images", "masks", "crop_data", "box")
    FUNCTION = "crop"
    CATEGORY = "MH/Crop"

    def crop(self, images, masks, padding=0, divisible_by=8):
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)

        batch_size, img_height, img_width, channels = images.shape

        nonzero = torch.nonzero(masks)

        if len(nonzero) == 0:
            print("Warning: MaskMinimalCrop found no mask pixels. Returning original.")
            crop_data = ((img_width, img_height), (0, 0, img_width, img_height))
            box = (0, 0, img_width, img_height)
            return (images, masks, crop_data, box)

        y_coords = nonzero[:, 1]
        x_coords = nonzero[:, 2]

        y_min = y_coords.min().item()
        y_max = y_coords.max().item()
        x_min = x_coords.min().item()
        x_max = x_coords.max().item()

        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_width, x_max + padding + 1)
        y_max = min(img_height, y_max + padding + 1)

        crop_width = x_max - x_min
        crop_height = y_max - y_min

        new_width = ((crop_width + divisible_by - 1) // divisible_by) * divisible_by
        new_height = ((crop_height + divisible_by - 1) // divisible_by) * divisible_by

        # Center the expansion
        expand_x = new_width - crop_width
        expand_y = new_height - crop_height

        x_min = max(0, x_min - expand_x // 2)
        y_min = max(0, y_min - expand_y // 2)
        x_max = min(img_width, x_min + new_width)
        y_max = min(img_height, y_min + new_height)

        # Adjust if we hit boundaries
        if x_max - x_min < new_width:
            x_min = max(0, x_max - new_width)
        if y_max - y_min < new_height:
            y_min = max(0, y_max - new_height)

        cropped_images = images[:, y_min:y_max, x_min:x_max, :]
        cropped_masks = masks[:, y_min:y_max, x_min:x_max]

        crop_data = ((img_width, img_height), (x_min, y_min, x_max, y_max))
        box = (x_min, y_min, x_max, y_max)

        return (cropped_images, cropped_masks, crop_data, box)


NODE_CLASS_MAPPINGS = {"mh_MaskMinimalCrop": mh_MaskMinimalCrop}

NODE_DISPLAY_NAME_MAPPINGS = {"mh_MaskMinimalCrop": "MH Mask Minimal Crop"}
