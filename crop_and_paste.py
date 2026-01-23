import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageChops, ImageFilter

# --- Helper Functions ---

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# --- Nodes ---

class mh_Image_Crop_Location:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "top": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "left": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "right": ("INT", {"default": 256, "max": 10000000, "min": 0, "step": 1}),
                "bottom": ("INT", {"default": 256, "max": 10000000, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA")
    FUNCTION = "image_crop_location"
    CATEGORY = "My Custom Nodes/Crop"

    def image_crop_location(self, image, top=0, left=0, right=256, bottom=256):
        image = tensor2pil(image)
        img_width, img_height = image.size

        # Calculate coordinates
        crop_top = max(top, 0)
        crop_left = max(left, 0)
        crop_bottom = min(bottom, img_height)
        crop_right = min(right, img_width)

        # Validate dimensions
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        if crop_width <= 0 or crop_height <= 0:
            print("Warning: Invalid crop dimensions. Returning original image.")
            return (pil2tensor(image), (image.size, (0, 0, image.size[0], image.size[1])))

        # Crop and resize to 8-pixel multiple (standard for diffusion models)
        crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        # This crop data contains the *original* crop size and coordinates
        crop_data = (crop.size, (crop_left, crop_top, crop_right, crop_bottom))
        
        # Resize for processing (optional, but WAS Suite does this)
        crop = crop.resize((((crop.size[0] // 8) * 8), ((crop.size[1] // 8) * 8)))

        return (pil2tensor(crop), crop_data)


class mh_Image_Paste_Crop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),       # The original full canvas
                "crop_image": ("IMAGE",),  # The processed cropped part
                "crop_data": ("CROP_DATA",),
                "crop_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_sharpening": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "image_paste_crop"
    CATEGORY = "My Custom Nodes/Crop"

    def image_paste_crop(self, image, crop_image, crop_data=None, crop_blending=0.25, crop_sharpening=0):
        if not crop_data:
            print("Error: No valid crop data found!")
            return (image, pil2tensor(Image.new("RGB", tensor2pil(image).size, (0,0,0))))

        result_image, result_mask = self.paste_image(
            tensor2pil(image), 
            tensor2pil(crop_image), 
            crop_data, 
            crop_blending, 
            crop_sharpening
        )

        return (result_image, result_mask)

    def paste_image(self, image, crop_image, crop_data, blend_amount=0.25, sharpen_amount=1):
        
        # Internal helper for gradient generation
        def lingrad(size, direction, white_ratio):
            image = Image.new('RGB', size)
            draw = ImageDraw.Draw(image)
            if direction == 'vertical':
                black_end = int(size[1] * (1 - white_ratio))
                for y in range(size[1]):
                    if y <= black_end:
                        color = (0, 0, 0)
                    else:
                        color_val = int(((y - black_end) / (size[1] - black_end)) * 255)
                        color = (color_val, color_val, color_val)
                    draw.line([(0, y), (size[0], y)], fill=color)
            elif direction == 'horizontal':
                black_end = int(size[0] * (1 - white_ratio))
                for x in range(size[0]):
                    if x <= black_end:
                        color = (0, 0, 0)
                    else:
                        color_val = int(((x - black_end) / (size[0] - black_end)) * 255)
                        color = (color_val, color_val, color_val)
                    draw.line([(x, 0), (x, size[1])], fill=color)
            return image.convert("L")

        # Unpack crop data
        # crop_size is the ORIGINAL size before any 8-pixel rounding
        crop_size, (left, top, right, bottom) = crop_data
        
        # Resize the processed crop back to the original hole size
        crop_image = crop_image.resize(crop_size)

        if sharpen_amount > 0:
            for _ in range(int(sharpen_amount)):
                crop_image = crop_image.filter(ImageFilter.SHARPEN)

        blended_image = Image.new('RGBA', image.size, (0, 0, 0, 255))
        blended_mask = Image.new('L', image.size, 0)
        crop_padded = Image.new('RGBA', image.size, (0, 0, 0, 0))
        
        blended_image.paste(image, (0, 0))
        crop_padded.paste(crop_image, (left, top))
        
        # Generate Blending Mask
        crop_mask = Image.new('L', crop_image.size, 0)

        # Create gradients at edges if they aren't touching the canvas border
        if top > 0:
            gradient_image = ImageOps.flip(lingrad(crop_image.size, 'vertical', blend_amount))
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        if left > 0:
            gradient_image = ImageOps.mirror(lingrad(crop_image.size, 'horizontal', blend_amount))
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        if right < image.width:
            gradient_image = lingrad(crop_image.size, 'horizontal', blend_amount)
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        if bottom < image.height:
            gradient_image = lingrad(crop_image.size, 'vertical', blend_amount)
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        crop_mask = ImageOps.invert(crop_mask)
        blended_mask.paste(crop_mask, (left, top))
        blended_mask = blended_mask.convert("L")
        
        # Paste using the mask
        blended_image.paste(crop_padded, (0, 0), blended_mask)

        return (pil2tensor(blended_image.convert("RGB")), pil2tensor(blended_mask.convert("RGB")))

# --- Node Mapping ---

NODE_CLASS_MAPPINGS = {
    "mh_Image_Crop_Location": mh_Image_Crop_Location,
    "mh_Image_Paste_Crop": mh_Image_Paste_Crop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mh_Image_Crop_Location": "MH Image Crop Location",
    "mh_Image_Paste_Crop": "MH Image Paste Crop"
}