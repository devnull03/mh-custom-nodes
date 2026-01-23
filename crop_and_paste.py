import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


def tensor2pil(image):
    """Convert a single image tensor (H, W, C) to PIL Image."""
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image):
    """Convert PIL Image to tensor (1, H, W, C)."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class mh_Image_Crop_Location:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "top": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "left": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "right": (
                    "INT",
                    {"default": 256, "max": 10000000, "min": 0, "step": 1},
                ),
                "bottom": (
                    "INT",
                    {"default": 256, "max": 10000000, "min": 0, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA")
    RETURN_NAMES = ("images", "crop_data")
    FUNCTION = "image_crop_location"
    CATEGORY = "MH/Crop"

    def image_crop_location(self, images, top=0, left=0, right=256, bottom=256):
        # images shape: (B, H, W, C)
        batch_size, img_height, img_width, channels = images.shape

        # Calculate and clamp coordinates
        crop_top = max(top, 0)
        crop_left = max(left, 0)
        crop_bottom = min(bottom, img_height)
        crop_right = min(right, img_width)

        # Validate dimensions
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        if crop_width <= 0 or crop_height <= 0:
            print("Warning: Invalid crop dimensions. Returning original image.")
            return (
                images,
                ((img_width, img_height), (0, 0, img_width, img_height)),
            )

        # Crop all batch items using tensor slicing
        cropped = images[:, crop_top:crop_bottom, crop_left:crop_right, :]

        # crop_data format: ((original_img_width, original_img_height), (left, top, right, bottom))
        crop_data = (
            (img_width, img_height),
            (crop_left, crop_top, crop_right, crop_bottom),
        )

        # Resize to 8-pixel multiple for diffusion models
        new_width = (crop_width // 8) * 8
        new_height = (crop_height // 8) * 8

        # Ensure at least 8 pixels
        new_width = max(new_width, 8)
        new_height = max(new_height, 8)

        if new_width != crop_width or new_height != crop_height:
            resized_list = []
            for i in range(batch_size):
                pil_img = tensor2pil(cropped[i])
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                resized_list.append(pil2tensor(pil_img))
            cropped = torch.cat(resized_list, dim=0)

        return (cropped, crop_data)


class mh_Image_Paste_Crop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # The original full canvas
                "crop_images": ("IMAGE",),  # The processed cropped part
                "crop_data": ("CROP_DATA",),
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
    FUNCTION = "image_paste_crop"
    CATEGORY = "MH/Crop"

    def image_paste_crop(
        self, images, crop_images, crop_data=None, crop_blending=0.25, crop_sharpening=0
    ):
        if not crop_data:
            print("Error: No valid crop data found!")
            batch_size = images.shape[0]
            h, w = images.shape[1], images.shape[2]
            empty_mask = torch.zeros((batch_size, h, w, 3), dtype=images.dtype)
            return (images, empty_mask)

        # Process each image in the batch
        batch_size = images.shape[0]
        crop_batch_size = crop_images.shape[0]

        result_images = []
        result_masks = []

        for i in range(batch_size):
            # Use corresponding crop image or last one if batch sizes don't match
            crop_idx = min(i, crop_batch_size - 1)

            result_img, result_mask = self.paste_image(
                tensor2pil(images[i]),
                tensor2pil(crop_images[crop_idx]),
                crop_data,
                crop_blending,
                crop_sharpening,
            )
            result_images.append(result_img)
            result_masks.append(result_mask)

        return (torch.cat(result_images, dim=0), torch.cat(result_masks, dim=0))

    def paste_image(
        self, original_image, crop_image, crop_data, blend_amount=0.25, sharpen_amount=1
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

        # Unpack crop data
        # crop_data format: ((original_img_width, original_img_height), (left, top, right, bottom))
        (orig_width, orig_height), (left, top, right, bottom) = crop_data

        # Calculate the target size for the crop (the hole we're filling)
        target_width = right - left
        target_height = bottom - top

        # Resize the processed crop back to fit the original hole
        crop_image_resized = crop_image.resize(
            (target_width, target_height), Image.LANCZOS
        )

        if sharpen_amount > 0:
            for _ in range(int(sharpen_amount)):
                crop_image_resized = crop_image_resized.filter(ImageFilter.SHARPEN)

        # Create output images at original size
        blended_image = Image.new("RGBA", original_image.size, (0, 0, 0, 255))
        blended_mask = Image.new("L", original_image.size, 0)
        crop_padded = Image.new("RGBA", original_image.size, (0, 0, 0, 0))

        # Paste original image as base
        blended_image.paste(original_image, (0, 0))

        # Paste the resized crop at the correct position
        crop_padded.paste(crop_image_resized, (left, top))

        # Generate Blending Mask
        crop_mask = Image.new("L", crop_image_resized.size, 0)

        # Create gradients at edges if they aren't touching the canvas border
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

        # Paste using the mask
        blended_image.paste(crop_padded, (0, 0), blended_mask)

        return (
            pil2tensor(blended_image.convert("RGB")),
            pil2tensor(blended_mask.convert("RGB")),
        )


NODE_CLASS_MAPPINGS = {
    "mh_Image_Crop_Location": mh_Image_Crop_Location,
    "mh_Image_Paste_Crop": mh_Image_Paste_Crop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mh_Image_Crop_Location": "MH Image Crop Location",
    "mh_Image_Paste_Crop": "MH Image Paste Crop",
}
