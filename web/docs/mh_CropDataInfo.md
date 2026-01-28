# Crop Data Info

Extracts information from CROP_DATA for debugging and inspection.

## Parameters

- **crop_data**: Crop metadata from `mh_Image_Crop_Location` or `mh_MaskMinimalCrop`

## Outputs

- **orig_width**: Original image width before cropping
- **orig_height**: Original image height before cropping
- **left**: Left boundary of the crop region
- **top**: Top boundary of the crop region
- **right**: Right boundary of the crop region
- **bottom**: Bottom boundary of the crop region
- **crop_width**: Width of the cropped output
- **crop_height**: Height of the cropped output