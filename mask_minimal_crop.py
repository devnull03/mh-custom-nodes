import torch
import torch.nn.functional as F


class mh_MaskMinimalCrop:
    """
    Static mask cropping - finds bounding box across ALL frames and applies same crop.
    """

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
                "scale_mode": (
                    ["none", "original", "custom"],
                    {"default": "none"},
                ),
                "target_width": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
                "target_height": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA", "BOX")
    RETURN_NAMES = ("images", "masks", "crop_data", "box")
    FUNCTION = "crop"
    CATEGORY = "MH/Crop"

    def crop(
        self,
        images,
        masks,
        padding=0,
        divisible_by=8,
        scale_mode="none",
        target_width=512,
        target_height=512,
    ):
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

        # Optional scaling of cropped outputs
        if scale_mode != "none":
            if scale_mode == "original":
                target_w, target_h = img_width, img_height
            else:
                target_w = max(1, int(target_width))
                target_h = max(1, int(target_height))

            cropped_images = self._resize_image_bhwc(cropped_images, target_h, target_w)
            cropped_masks = self._resize_mask_bhw(cropped_masks, target_h, target_w)

        crop_data = ((img_width, img_height), (x_min, y_min, x_max, y_max))
        box = (x_min, y_min, x_max, y_max)

        return (cropped_images, cropped_masks, crop_data, box)

    def _resize_image_bhwc(self, images, target_h, target_w):
        # images: [B, H, W, C]
        nchw = images.permute(0, 3, 1, 2)
        resized = F.interpolate(
            nchw, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return resized.permute(0, 2, 3, 1)

    def _resize_mask_bhw(self, masks, target_h, target_w):
        # masks: [B, H, W]
        nchw = masks.unsqueeze(1).float()
        resized = F.interpolate(
            nchw, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return resized.squeeze(1).clamp(0.0, 1.0)


class mh_MaskTrackingCrop:
    """
    Dynamic per-frame mask cropping with tracking.
    Calculates the maximum bounding box across all frames to determine output size,
    then crops each frame individually centered on its mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "padding": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
                "divisible_by": (
                    "INT",
                    {"default": 16, "min": 1, "max": 256, "step": 1},
                ),
                "smoothing": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "scale_mode": (
                    "STRING",
                    {"default": "none", "choices": ["none", "original", "custom"]},
                ),
                "target_width": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
                "target_height": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA_BATCH", "INT", "INT")
    RETURN_NAMES = ("images", "masks", "crop_data_batch", "crop_width", "crop_height")
    FUNCTION = "crop"
    CATEGORY = "MH/Crop"

    def crop(
        self,
        images,
        masks,
        padding=32,
        divisible_by=16,
        smoothing=0.0,
        scale_mode="none",
        target_width=512,
        target_height=512,
    ):
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)

        batch_size, img_height, img_width, channels = images.shape

        # Step 1: Calculate per-frame bounding boxes
        per_frame_boxes = []
        max_width = 0
        max_height = 0

        for i in range(batch_size):
            frame_mask = masks[i] if i < masks.shape[0] else masks[-1]
            nonzero = torch.nonzero(frame_mask)

            if len(nonzero) == 0:
                # No mask pixels - use center of image as fallback
                cx, cy = img_width // 2, img_height // 2
                per_frame_boxes.append(
                    (cx, cy, 0, 0)
                )  # center_x, center_y, width, height
            else:
                y_coords = nonzero[:, 0]
                x_coords = nonzero[:, 1]

                y_min = y_coords.min().item()
                y_max = y_coords.max().item()
                x_min = x_coords.min().item()
                x_max = x_coords.max().item()

                bbox_width = x_max - x_min + 1
                bbox_height = y_max - y_min + 1
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                per_frame_boxes.append((center_x, center_y, bbox_width, bbox_height))

                max_width = max(max_width, bbox_width)
                max_height = max(max_height, bbox_height)

        # Step 2: Calculate output size from max bbox + padding
        if max_width == 0 or max_height == 0:
            # All frames had empty masks
            print("Warning: MaskTrackingCrop found no mask pixels in any frame.")
            max_width = img_width // 4
            max_height = img_height // 4

        output_width = max_width + padding * 2
        output_height = max_height + padding * 2

        # Make divisible
        output_width = (
            (output_width + divisible_by - 1) // divisible_by
        ) * divisible_by
        output_height = (
            (output_height + divisible_by - 1) // divisible_by
        ) * divisible_by

        # Clamp to image size
        output_width = min(output_width, img_width)
        output_height = min(output_height, img_height)

        # Step 3: Apply temporal smoothing to centers if requested
        smoothed_centers = []
        prev_cx, prev_cy = None, None

        for i, (cx, cy, bw, bh) in enumerate(per_frame_boxes):
            if smoothing > 0 and prev_cx is not None:
                cx = int(prev_cx * smoothing + cx * (1 - smoothing))
                cy = int(prev_cy * smoothing + cy * (1 - smoothing))
            smoothed_centers.append((cx, cy))
            prev_cx, prev_cy = cx, cy

        # Step 4: Calculate per-frame crop regions and perform crops
        cropped_images_list = []
        cropped_masks_list = []
        crop_regions = []

        half_w = output_width // 2
        half_h = output_height // 2

        for i in range(batch_size):
            cx, cy = smoothed_centers[i]

            # Calculate crop bounds centered on mask
            x_min = cx - half_w
            y_min = cy - half_h
            x_max = x_min + output_width
            y_max = y_min + output_height

            # Adjust if out of bounds
            if x_min < 0:
                x_min = 0
                x_max = output_width
            if y_min < 0:
                y_min = 0
                y_max = output_height
            if x_max > img_width:
                x_max = img_width
                x_min = img_width - output_width
            if y_max > img_height:
                y_max = img_height
                y_min = img_height - output_height

            # Ensure bounds are valid
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)

            # Store crop region
            crop_regions.append((x_min, y_min, x_max, y_max))

            # Crop image
            cropped_img = images[i : i + 1, y_min:y_max, x_min:x_max, :]
            cropped_images_list.append(cropped_img)

            # Crop mask
            mask_idx = min(i, masks.shape[0] - 1)
            cropped_mask = masks[mask_idx : mask_idx + 1, y_min:y_max, x_min:x_max]
            cropped_masks_list.append(cropped_mask)

        # Concatenate results
        cropped_images = torch.cat(cropped_images_list, dim=0)
        cropped_masks = torch.cat(cropped_masks_list, dim=0)

        # Optional scaling of cropped outputs
        if scale_mode != "none":
            if scale_mode == "original":
                target_w, target_h = img_width, img_height
            else:
                target_w = max(1, int(target_width))
                target_h = max(1, int(target_height))
            cropped_images = self._resize_image_bhwc(cropped_images, target_h, target_w)
            cropped_masks = self._resize_mask_bhw(cropped_masks, target_h, target_w)
            scaled_size = (target_w, target_h)
        else:
            scaled_size = None

        # Create crop data batch
        crop_data_batch = {
            "original_size": (img_width, img_height),
            "output_size": (output_width, output_height),
            "crop_regions": crop_regions,  # List of (x_min, y_min, x_max, y_max) per frame
            "scaled_size": scaled_size,
        }

        return (
            cropped_images,
            cropped_masks,
            crop_data_batch,
            output_width,
            output_height,
        )

    def _resize_image_bhwc(self, images, target_h, target_w):
        # images: [B, H, W, C]
        nchw = images.permute(0, 3, 1, 2)
        resized = F.interpolate(
            nchw, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return resized.permute(0, 2, 3, 1)

    def _resize_mask_bhw(self, masks, target_h, target_w):
        # masks: [B, H, W]
        nchw = masks.unsqueeze(1).float()
        resized = F.interpolate(
            nchw, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return resized.squeeze(1).clamp(0.0, 1.0)


NODE_CLASS_MAPPINGS = {
    "mh_MaskMinimalCrop": mh_MaskMinimalCrop,
    "mh_MaskTrackingCrop": mh_MaskTrackingCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mh_MaskMinimalCrop": "MH Mask Minimal Crop",
    "mh_MaskTrackingCrop": "MH Mask Tracking Crop",
}
