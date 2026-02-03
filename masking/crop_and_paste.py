import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage


def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def lingrad(size, direction, white_ratio):
    """Creates a linear gradient image for blending."""
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


def detect_drift_phase_correlation(original_crop, generated_crop, search_radius=16):
    """
    Detect spatial drift between original and generated crop using phase correlation.
    This is the fastest method (~2ms per frame) using FFT.
    
    Args:
        original_crop: PIL Image of the original cropped region
        generated_crop: PIL Image of the generated output
        search_radius: Maximum drift to detect in pixels
    
    Returns:
        (dx, dy): Detected drift offset to compensate
    """
    # Convert to grayscale numpy arrays
    orig_gray = np.array(original_crop.convert('L'), dtype=np.float32)
    gen_gray = np.array(generated_crop.convert('L'), dtype=np.float32)
    
    # Ensure same size
    if orig_gray.shape != gen_gray.shape:
        # Resize generated to match original
        generated_crop_resized = generated_crop.resize(original_crop.size, Image.LANCZOS)
        gen_gray = np.array(generated_crop_resized.convert('L'), dtype=np.float32)
    
    # Apply window function to reduce edge effects
    h, w = orig_gray.shape
    window_h = np.hanning(h).reshape(-1, 1)
    window_w = np.hanning(w).reshape(1, -1)
    window = window_h * window_w
    
    orig_windowed = orig_gray * window
    gen_windowed = gen_gray * window
    
    # Compute FFT
    fft_orig = np.fft.fft2(orig_windowed)
    fft_gen = np.fft.fft2(gen_windowed)
    
    # Cross-power spectrum
    cross_power = fft_orig * np.conj(fft_gen)
    cross_power_norm = cross_power / (np.abs(cross_power) + 1e-10)
    
    # Inverse FFT to get correlation
    correlation = np.fft.ifft2(cross_power_norm)
    correlation = np.abs(np.fft.fftshift(correlation))
    
    # Find peak within search radius
    center_y, center_x = h // 2, w // 2
    y_min = max(0, center_y - search_radius)
    y_max = min(h, center_y + search_radius + 1)
    x_min = max(0, center_x - search_radius)
    x_max = min(w, center_x + search_radius + 1)
    
    search_region = correlation[y_min:y_max, x_min:x_max]
    peak_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
    
    # Calculate offset from center
    dy = peak_idx[0] - (center_y - y_min)
    dx = peak_idx[1] - (center_x - x_min)
    
    return int(dx), int(dy)


class mh_Image_Crop_Location:
    """
    Crops images at a specific location.
    """

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
                "divisible_by": (
                    "INT",
                    {"default": 16, "min": 1, "max": 256, "step": 1},
                ),
                "scale_mode": (
                    "STRING",
                    {"default": "none", "choices": ["none", "original", "custom"]},
                ),
                "resolution": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA")
    RETURN_NAMES = ("images", "crop_data")
    FUNCTION = "image_crop_location"
    CATEGORY = "MH/Crop"

    def image_crop_location(
        self,
        images,
        top=0,
        left=0,
        right=256,
        bottom=256,
        divisible_by=8,
        scale_mode="none",
        resolution=512,
    ):
        batch_size, img_height, img_width, channels = images.shape

        crop_top = max(top, 0)
        crop_left = max(left, 0)
        crop_bottom = min(bottom, img_height)
        crop_right = min(right, img_width)

        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top

        if crop_width <= 0 or crop_height <= 0:
            print("Warning: Invalid crop dimensions. Returning original image.")
            return (
                images,
                (
                    (img_width, img_height),
                    (0, 0, img_width, img_height),
                    (img_width, img_height),
                ),
            )

        cropped = images[:, crop_top:crop_bottom, crop_left:crop_right, :]

        new_width = max((crop_width // divisible_by) * divisible_by, divisible_by)
        new_height = max((crop_height // divisible_by) * divisible_by, divisible_by)

        if new_width != crop_width or new_height != crop_height:
            resized_list = []
            for i in range(batch_size):
                pil_img = tensor2pil(cropped[i])
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                resized_list.append(pil2tensor(pil_img))
            cropped = torch.cat(resized_list, dim=0)

        if scale_mode != "none":
            if scale_mode == "original":
                target_w, target_h = img_width, img_height
            else:
                # Scale to resolution while preserving aspect ratio
                crop_w = int(cropped.shape[2])
                crop_h = int(cropped.shape[1])
                if crop_w >= crop_h:
                    target_w = resolution
                    target_h = int(resolution * crop_h / crop_w)
                else:
                    target_h = resolution
                    target_w = int(resolution * crop_w / crop_h)
                # Ensure divisible_by compliance
                target_w = max(divisible_by, (target_w // divisible_by) * divisible_by)
                target_h = max(divisible_by, (target_h // divisible_by) * divisible_by)

            scaled_list = []
            for i in range(batch_size):
                pil_img = tensor2pil(cropped[i])
                pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
                scaled_list.append(pil2tensor(pil_img))
            cropped = torch.cat(scaled_list, dim=0)

        output_height = int(cropped.shape[1])
        output_width = int(cropped.shape[2])

        crop_data = (
            (img_width, img_height),
            (crop_left, crop_top, crop_right, crop_bottom),
            (output_width, output_height),
        )

        return (cropped, crop_data)


class mh_Image_Paste_Crop:
    """
    Pastes cropped images back to their original position (static crop).
    Works with CROP_DATA from mh_MaskMinimalCrop or mh_Image_Crop_Location.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "crop_images": ("IMAGE",),
                "crop_data": ("CROP_DATA",),
                "crop_blending": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "crop_sharpening": (
                    "INT",
                    {"default": 0, "min": 0, "max": 3, "step": 1},
                ),
                "drift_correction": (
                    "BOOLEAN",
                    {"default": False},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "image_paste_crop"
    CATEGORY = "MH/Crop"

    def image_paste_crop(
        self, images, crop_images, crop_data=None, crop_blending=0.25, crop_sharpening=0, drift_correction=False
    ):
        if not crop_data:
            print("Error: No valid crop data found!")
            batch_size = images.shape[0]
            h, w = images.shape[1], images.shape[2]
            empty_mask = torch.zeros((batch_size, h, w, 3), dtype=images.dtype)
            return (images, empty_mask)

        batch_size = images.shape[0]
        crop_batch_size = crop_images.shape[0]
        
        # Use the smaller count - discard extra base frames if crop has fewer
        output_size = min(batch_size, crop_batch_size)
        
        if batch_size != crop_batch_size:
            print(f"[crop_and_paste] Frame count mismatch: images={batch_size}, crop_images={crop_batch_size}, outputting {output_size}")

        # Extract crop region for drift detection reference
        if len(crop_data) >= 3:
            (orig_width, orig_height), (left, top, right, bottom), (out_w, out_h) = crop_data
        else:
            (orig_width, orig_height), (left, top, right, bottom) = crop_data
            out_w, out_h = right - left, bottom - top
        
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)

        result_images = []
        result_masks = []
        
        drift_offsets = []

        for i in range(output_size):
            original_pil = tensor2pil(images[i])
            crop_pil = tensor2pil(crop_images[i])
            
            # Calculate drift correction if enabled
            dx, dy = 0, 0
            if drift_correction:
                # Extract original crop region for comparison
                original_crop_region = original_pil.crop((left, top, right, bottom))
                dx, dy = detect_drift_phase_correlation(original_crop_region, crop_pil)
                drift_offsets.append((dx, dy))
            
            result_img, result_mask = self.paste_image(
                original_pil,
                crop_pil,
                crop_data,
                crop_blending,
                crop_sharpening,
                drift_offset=(dx, dy),
            )
            result_images.append(result_img)
            result_masks.append(result_mask)
        
        if drift_correction and drift_offsets:
            # Log drift statistics
            avg_dx = sum(d[0] for d in drift_offsets) / len(drift_offsets)
            avg_dy = sum(d[1] for d in drift_offsets) / len(drift_offsets)
            max_dx = max(abs(d[0]) for d in drift_offsets)
            max_dy = max(abs(d[1]) for d in drift_offsets)
            print(f"[drift_correction] Avg offset: ({avg_dx:.1f}, {avg_dy:.1f}), Max: ({max_dx}, {max_dy})")

        return (torch.cat(result_images, dim=0), torch.cat(result_masks, dim=0))

    def paste_image(
        self, original_image, crop_image, crop_data, blend_amount=0.25, sharpen_amount=1, drift_offset=(0, 0)
    ):
        if len(crop_data) >= 3:
            (orig_width, orig_height), (left, top, right, bottom), (out_w, out_h) = (
                crop_data
            )
        else:
            (orig_width, orig_height), (left, top, right, bottom) = crop_data
            out_w, out_h = right - left, bottom - top

        # Ensure all coordinates are integers
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        
        # Apply drift correction offset
        dx, dy = drift_offset
        left -= dx
        top -= dy
        right -= dx
        bottom -= dy
        
        # Clamp to image bounds after drift correction
        left = max(0, min(left, original_image.width - 1))
        top = max(0, min(top, original_image.height - 1))
        right = max(left + 1, min(right, original_image.width))
        bottom = max(top + 1, min(bottom, original_image.height))
        
        crop_region_width = right - left
        crop_region_height = bottom - top
        target_width = max(1, crop_region_width)
        target_height = max(1, crop_region_height)

        if crop_image.size != (target_width, target_height):
            crop_image_resized = crop_image.resize(
                (target_width, target_height), Image.LANCZOS
            )
        else:
            crop_image_resized = crop_image

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
        crop_regions = crop_data_batch["crop_regions"]
        output_size = crop_data_batch.get("output_size", None)

        batch_size = images.shape[0]
        crop_batch_size = crop_images.shape[0]
        num_regions = len(crop_regions)
        
        # Use the smallest count - discard extra base frames if crop has fewer
        output_count = min(batch_size, crop_batch_size, num_regions)
        
        if batch_size != crop_batch_size or batch_size != num_regions:
            print(f"[crop_and_paste_tracking] Frame count mismatch: images={batch_size}, crop_images={crop_batch_size}, regions={num_regions}, outputting {output_count}")

        result_images = []
        result_masks = []

        for i in range(output_count):
            crop_region = crop_regions[i]

            result_img, result_mask = self.paste_image(
                tensor2pil(images[i]),
                tensor2pil(crop_images[i]),
                original_size,
                crop_region,
                output_size,
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
        output_size=None,
        blend_amount=0.25,
        sharpen_amount=1,
    ):
        left, top, right, bottom = crop_region
        
        # Ensure all coordinates are integers to handle floating point fps edge cases
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)

        crop_region_width = right - left
        crop_region_height = bottom - top
        target_width = max(1, crop_region_width)
        target_height = max(1, crop_region_height)

        if crop_image.size != (target_width, target_height):
            crop_image_resized = crop_image.resize(
                (target_width, target_height), Image.LANCZOS
            )
        else:
            crop_image_resized = crop_image

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


class mh_CropDataInfo:
    """
    Extracts information from CROP_DATA for debugging/inspection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_data": ("CROP_DATA",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "orig_width",
        "orig_height",
        "left",
        "top",
        "right",
        "bottom",
        "crop_width",
        "crop_height",
    )
    FUNCTION = "extract"
    CATEGORY = "MH/Crop"

    def extract(self, crop_data):
        if len(crop_data) >= 3:
            (
                (orig_width, orig_height),
                (left, top, right, bottom),
                (crop_width, crop_height),
            ) = crop_data
        else:
            (orig_width, orig_height), (left, top, right, bottom) = crop_data
            crop_width = right - left
            crop_height = bottom - top
        return (
            orig_width,
            orig_height,
            left,
            top,
            right,
            bottom,
            crop_width,
            crop_height,
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
    "mh_Image_Crop_Location": mh_Image_Crop_Location,
    "mh_Image_Paste_Crop": mh_Image_Paste_Crop,
    "mh_Image_Paste_Crop_Tracking": mh_Image_Paste_Crop_Tracking,
    "mh_CropDataInfo": mh_CropDataInfo,
    "mh_CropDataBatchInfo": mh_CropDataBatchInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mh_Image_Crop_Location": "MH Image Crop Location",
    "mh_Image_Paste_Crop": "MH Image Paste Crop",
    "mh_Image_Paste_Crop_Tracking": "MH Image Paste Crop Tracking",
    "mh_CropDataInfo": "MH Crop Data Info",
    "mh_CropDataBatchInfo": "MH Crop Data Batch Info",
}
