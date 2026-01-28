# Mask Minimal Crop

Finds the bounding box of a mask across all frames and crops images to that region. Uses a static crop region for the entire batch.

## Parameters

- **images**: Input image batch to crop
- **masks**: Mask batch defining the region of interest
- **padding**: Extra pixels to add around the detected bounding box
- **divisible_by**: Ensures output dimensions are divisible by this value (useful for model requirements like 16 for VAE)
- **scale_mode**: `none` keeps crop size, `original` scales to original image size, `custom` scales to specified resolution
- **resolution**: Target resolution when using `custom` scale mode

## Outputs

- **images**: Cropped image batch
- **masks**: Cropped mask batch
- **crop_data**: Metadata for pasting back with `mh_Image_Paste_Crop`
- **box**: Bounding box coordinates (x_min, y_min, x_max, y_max)