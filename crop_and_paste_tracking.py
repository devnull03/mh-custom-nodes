import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class mh_Image_Paste_Crop_Tracking:
    """
    Pastes back per-frame tracked crops to their original positions.
    Works with CROP_DATA_BATCH from mh_MaskTrackingCrop.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "crop_images": ("IMAGE",),
                "crop_data_batch": ("CROP_DATA_BATCH",),
                "crop_blending": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "crop_sharpening": (
                    "INT",
                    {"default": 0, "min": 0, "max": 3, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "image_paste_crop_tracking"
    CATEGORY = "MH/Crop"

    def image_paste_crop_tracking(
        self,
        images,
        crop_images,
        crop_data_batch,
        crop_blending=0.25,
        crop_sharpening=0,
    ):
        if not crop_data_batch:
            print("Error: No valid crop data batch found!")
            batch_size = images.shape[0]
            h, w = images.shape[1], images.shape[2]
            empty_mask = torch.zeros((batch_size, h, w, 3), dtype=images.dtype)
            return (images, empty_mask)

        original_size = crop_data_batch["original_size"]
        output_size = crop_data_batch["output_size"]
        crop_regions = crop_data_batch["crop_regions"]

        batch_size = images.shape[0]
        crop_batch_size = crop_images.shape[0]

        result_images = []
        result_masks = []

        for i in range(batch_size):
            crop_idx = min(i, crop_batch_size - 1)
            region_idx = min(i, len(crop_regions) - 1)

            crop_region = crop_regions[region_idx]

            result_img, result_mask = self.paste_image(
                tensor2pil(images[i]),
                tensor2pil(crop_images[crop_idx]),
                original_size,
                crop_region,
                crop_blending,
                crop_sharpening,
            )
            result_images.append(result_img)
            result_masks.append(result_mask)

        return (torch.cat(result_images, dim=0), torch.cat(result_masks, dim=0))

    def paste_image(
        self,
        original_image,
        crop_image,
        original_size,
        crop_region,
        blend_amount=0.25,
        sharpen_amount=1,
    ):
        def lingrad(size, direction, white_ratio):
            grad_image = Image.new("RGB", size)
            draw = ImageDraw.Draw(grad_image)
            if direction == "vertical":
                black_end = int(size[1] * (1 - white_ratio))
                for y in range(size[1]):
                    if y <= black_end:
                        color = (0, 0, 0)
                    else:
                        color_val = int(((y - black_end) / (size[1] - black_end)) * 255)
                        color = (color_val, color_val, color_val)
                    draw.line([(0, y), (size[0], y)], fill=color)
            elif direction == "horizontal":
                black_end = int(size[0] * (1 - white_ratio))
                for x in range(size[0]):
                    if x <= black_end:
                        color = (0, 0, 0)
                    else:
                        color_val = int(((x - black_end) / (size[0] - black_end)) * 255)
                        color = (color_val, color_val, color_val)
                    draw.line([(x, 0), (x, size[1])], fill=color)
            return grad_image.convert("L")

        left, top, right, bottom = crop_region
        orig_width, orig_height = original_size

        target_width = right - left
        target_height = bottom - top

        # Resize crop image to target size if needed
        crop_image_resized = crop_image.resize(
            (target_width, target_height), Image.LANCZOS
        )

        if sharpen_amount > 0:
            for _ in range(int(sharpen_amount)):
                crop_image_resized = crop_image_resized.filter(ImageFilter.SHARPEN)

        blended_image = Image.new("RGBA", original_image.size, (0, 0, 0, 255))
        blended_mask = Image.new("L", original_image.size, 0)
        crop_padded = Image.new("RGBA", original_image.size, (0, 0, 0, 0))

        blended_image.paste(original_image, (0, 0))
        crop_padded.paste(crop_image_resized, (left, top))

        crop_mask = Image.new("L", crop_image_resized.size, 0)

        if top > 0:
            gradient_image = ImageOps.flip(
                lingrad(crop_image_resized.size, "vertical", blend_amount)
            )
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        if left > 0:
            gradient_image = ImageOps.mirror(
                lingrad(crop_image_resized.size, "horizontal", blend_amount)
            )
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        if right < original_image.width:
            gradient_image = lingrad(
                crop_image_resized.size, "horizontal", blend_amount
            )
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        if bottom < original_image.height:
            gradient_image = lingrad(crop_image_resized.size, "vertical", blend_amount)
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        crop_mask = ImageOps.invert(crop_mask)
        blended_mask.paste(crop_mask, (left, top))
        blended_mask = blended_mask.convert("L")

        blended_image.paste(crop_padded, (0, 0), blended_mask)

        return (
            pil2tensor(blended_image.convert("RGB")),
            pil2tensor(blended_mask.convert("RGB")),
        )


class mh_CropDataBatchInfo:
    """
    Extracts information from CROP_DATA_BATCH for debugging/inspection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_data_batch": ("CROP_DATA_BATCH",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "orig_width",
        "orig_height",
        "crop_width",
        "crop_height",
        "num_frames",
    )
    FUNCTION = "extract"
    CATEGORY = "MH/Crop"

    def extract(self, crop_data_batch):
        orig_width, orig_height = crop_data_batch["original_size"]
        crop_width, crop_height = crop_data_batch["output_size"]
        num_frames = len(crop_data_batch["crop_regions"])

        return (
            orig_width,
            orig_height,
            crop_width,
            crop_height,
            num_frames,
        )


NODE_CLASS_MAPPINGS = {
    "mh_Image_Paste_Crop_Tracking": mh_Image_Paste_Crop_Tracking,
    "mh_CropDataBatchInfo": mh_CropDataBatchInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mh_Image_Paste_Crop_Tracking": "MH Image Paste Crop Tracking",
    "mh_CropDataBatchInfo": "MH Crop Data Batch Info",
}
