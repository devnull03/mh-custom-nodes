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

        cropped_images = images[:, y_min:y_max, x_min:x_max, :]
        cropped_masks = masks[:, y_min:y_max, x_min:x_max]

        crop_data = ((img_width, img_height), (x_min, y_min, x_max, y_max))
        box = (x_min, y_min, x_max, y_max)

        return (cropped_images, cropped_masks, crop_data, box)


NODE_CLASS_MAPPINGS = {"mh_MaskMinimalCrop": mh_MaskMinimalCrop}

NODE_DISPLAY_NAME_MAPPINGS = {"mh_MaskMinimalCrop": "MH Mask Minimal Crop"}
