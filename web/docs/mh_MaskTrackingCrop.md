# Mask Tracking Crop

Dynamic per-frame mask cropping with tracking. Calculates the maximum bounding box across all frames to determine a consistent output size, then crops each frame individually centered on its mask.

## Parameters

- **images**: Input image batch to crop
- **masks**: Mask batch defining the region of interest per frame
- **padding**: Extra pixels to add around the detected bounding boxes
- **divisible_by**: Ensures output dimensions are divisible by this value (useful for model requirements like 16 for VAE)
- **smoothing**: Temporal smoothing factor (0.0-1.0) to reduce jitter between frames
- **scale_mode**: `none` keeps crop size, `original` scales to original image size, `custom` scales to specified resolution
- **resolution**: Target resolution when using `custom` scale mode

## Outputs

- **images**: Cropped image batch with per-frame tracking
- **masks**: Cropped mask batch
- **crop_data_batch**: Per-frame metadata for pasting back with `mh_Image_Paste_Crop_Tracking`
- **crop_width**: Output crop width
- **crop_height**: Output crop height