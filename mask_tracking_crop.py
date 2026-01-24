import torch


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
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA_BATCH", "INT", "INT")
    RETURN_NAMES = ("images", "masks", "crop_data_batch", "crop_width", "crop_height")
    FUNCTION = "crop"
    CATEGORY = "MH/Crop"

    def crop(self, images, masks, padding=32, divisible_by=16, smoothing=0.0):
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

        # Create crop data batch
        crop_data_batch = {
            "original_size": (img_width, img_height),
            "output_size": (output_width, output_height),
            "crop_regions": crop_regions,  # List of (x_min, y_min, x_max, y_max) per frame
        }

        return (
            cropped_images,
            cropped_masks,
            crop_data_batch,
            output_width,
            output_height,
        )


NODE_CLASS_MAPPINGS = {"mh_MaskTrackingCrop": mh_MaskTrackingCrop}

NODE_DISPLAY_NAME_MAPPINGS = {"mh_MaskTrackingCrop": "MH Mask Tracking Crop"}
