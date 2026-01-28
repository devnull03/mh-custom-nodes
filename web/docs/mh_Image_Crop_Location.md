# Image Crop Location

Crops images at a specific location with optional scaling.

## Parameters

- **images**: Input image batch to crop
- **top**: Top boundary of the crop region (pixels)
- **left**: Left boundary of the crop region (pixels)
- **right**: Right boundary of the crop region (pixels)
- **bottom**: Bottom boundary of the crop region (pixels)
- **divisible_by**: Ensures output dimensions are divisible by this value (useful for model requirements)
- **scale_mode**: `none` keeps crop size, `original` scales to original image size, `custom` scales to specified resolution
- **resolution**: Target resolution when using `custom` scale mode

## Outputs

- **images**: Cropped image batch
- **crop_data**: Metadata for pasting back with `mh_Image_Paste_Crop`
