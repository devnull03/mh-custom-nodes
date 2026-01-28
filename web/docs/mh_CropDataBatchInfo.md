# Crop Data Batch Info

Extracts information from CROP_DATA_BATCH for debugging and inspection of tracked crops.

## Parameters

- **crop_data_batch**: Per-frame crop metadata from `mh_MaskTrackingCrop`

## Outputs

- **orig_width**: Original image width before cropping
- **orig_height**: Original image height before cropping
- **crop_width**: Width of the cropped output
- **crop_height**: Height of the cropped output
- **num_frames**: Number of frames in the batch