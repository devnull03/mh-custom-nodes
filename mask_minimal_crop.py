class Mask_Minimal_Crop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",), # Optional: If you want to crop the image at the same time
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA", "BOX")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_data", "bbox_coords")
    FUNCTION = "execute"
    CATEGORY = "My Custom Nodes/Crop"

    def execute(self, mask, padding=0, image=None):
        # mask shape: [Batch, Height, Width]
        # image shape: [Batch, Height, Width, Channels]

        # 1. Find non-zero elements (white pixels)
        # Note: We use the first mask in the batch to determine the crop region
        # to ensure temporal consistency if used on video, or just singular logic.
        mask_2d = mask[0] if mask.dim() == 3 else mask
        
        non_zero_coords = torch.nonzero(mask_2d > 0)
        
        # Handle empty mask case
        if len(non_zero_coords) == 0:
            print("Warning: Empty mask. Returning full image/mask.")
            height, width = mask_2d.shape
            # Return full coordinates
            crop_data = ((width, height), (0, 0, width, height))
            bbox = (0, 0, width, height)
            return (image, mask, crop_data, bbox)

        # 2. Calculate Min/Max Bounds
        min_y = torch.min(non_zero_coords[:, 0]).item()
        max_y = torch.max(non_zero_coords[:, 0]).item()
        min_x = torch.min(non_zero_coords[:, 1]).item()
        max_x = torch.max(non_zero_coords[:, 1]).item()

        # 3. Apply Padding
        height, width = mask_2d.shape
        
        # Clamp values to be within image bounds
        min_y = max(0, min_y - padding)
        max_y = min(height, max_y + padding)
        min_x = max(0, min_x - padding)
        max_x = min(width, max_x + padding)

        # 4. Define Crop Coordinates (Left, Top, Right, Bottom)
        # Python slicing is exclusive on the upper bound, so we use max + 1 usually,
        # but for coordinates (right/bottom), strictly it's the pixel index.
        # WAS Suite uses (left, top, right, bottom) where right/bottom are the Slice End indices.
        
        crop_top = min_y
        crop_bottom = max_y + 1 # +1 to include the last pixel
        crop_left = min_x
        crop_right = max_x + 1

        # Ensure we didn't go out of bounds
        crop_bottom = min(height, crop_bottom)
        crop_right = min(width, crop_right)

        # 5. Crop Content
        # Crop Mask: [Batch, H, W] -> slice H and W
        cropped_mask = mask[:, crop_top:crop_bottom, crop_left:crop_right]

        cropped_image = None
        if image is not None:
            # Crop Image: [Batch, H, W, C]
            cropped_image = image[:, crop_top:crop_bottom, crop_left:crop_right, :]
        else:
            # Create a black placeholder if no image provided, just to satisfy output type
            c_h = crop_bottom - crop_top
            c_w = crop_right - crop_left
            cropped_image = torch.zeros((1, c_h, c_w, 3), dtype=torch.float32)

        # 6. Construct CROP_DATA
        # Structure: ( (width, height), (left, top, right, bottom) )
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        
        crop_data = ( (crop_width, crop_height), (crop_left, crop_top, crop_right, crop_bottom) )
        bbox = (crop_left, crop_top, crop_right, crop_bottom)

        return (cropped_image, cropped_mask, crop_data, bbox)
        

NODE_CLASS_MAPPINGS = {
    "My_Mask_Minimal_Crop": Mask_Minimal_Crop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "My_Mask_Minimal_Crop": "Mask Minimal Crop (Auto)"
}
