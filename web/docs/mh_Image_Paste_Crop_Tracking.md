# Image Paste Crop Tracking

Pastes per-frame tracked crops back to their original positions. Works with dynamic crop data where each frame may have a different crop region.

## Parameters

- **images**: Original image batch (background to paste onto)
- **crop_images**: Processed cropped images to paste back
- **crop_data_batch**: Per-frame crop metadata from `mh_MaskTrackingCrop`
- **crop_blending**: Edge blend amount (0.0-1.0) for smooth transitions
- **crop_sharpening**: Number of sharpening passes (0-3) applied before pasting

## Outputs

- **images**: Composited result with cropped images pasted back at their tracked positions
- **masks**: Blend masks showing where pasting occurred

## Frame Handling

- Uses 1:1 frame mapping when `crop_images` has enough frames
- Extra frames in `crop_images` beyond `images` batch size are ignored
- If `crop_images` has fewer frames, clamps to last available frame (with warning)
- Region indices are handled similarly for `crop_data_batch`