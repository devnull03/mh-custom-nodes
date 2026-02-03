# Image Paste Crop

Pastes cropped images back to their original position using static crop data.

## Parameters

- **images**: Original image batch (background to paste onto)
- **crop_images**: Processed cropped images to paste back
- **crop_data**: Crop metadata from `mh_Image_Crop_Location` or `mh_MaskMinimalCrop`
- **crop_blending**: Edge blend amount (0.0-1.0) for smooth transitions
- **crop_sharpening**: Number of sharpening passes (0-3) applied before pasting

## Outputs

- **images**: Composited result with cropped images pasted back
- **masks**: Blend masks showing where pasting occurred

## Frame Handling

- Uses 1:1 frame mapping when `crop_images` has enough frames
- Extra frames in `crop_images` beyond `images` batch size are ignored
- If `crop_images` has fewer frames, clamps to last available frame (with warning)