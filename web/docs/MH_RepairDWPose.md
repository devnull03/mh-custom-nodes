# Repair DWPose

Repairs DWPose/ViTPose outputs by interpolating missing or low-confidence keypoints using temporal information from neighboring frames. Handles both light occlusions (individual joints) and heavy occlusions (full pose replacement).

## Parameters

- **pose_keypoints**: Pose keypoint data from DWPose or ViTPose nodes
- **confidence_threshold**: Keypoints with confidence below this value are considered missing (0.0-1.0, default: 0.3)
- **temporal_window**: Number of frames to search forward and backward for valid keypoints (1-32, default: 8)
- **heavy_occlusion_threshold**: If more than this percentage of keypoints are missing, replace the entire pose (0.1-0.9, default: 0.5)

## Outputs

- **repaired_pose**: Pose keypoints with missing data interpolated from neighboring frames

## Repair Strategies

**Light Occlusion** (< 50% keypoints missing):
- Interpolates individual missing keypoints from the nearest valid frames before and after
- Preserves original keypoints that have sufficient confidence

**Heavy Occlusion** (> 50% keypoints missing):
- Replaces the entire pose by interpolating between the nearest "good" frames
- Falls back to copying from the nearest good frame if only one direction is available

## Notes

- Repaired keypoints are marked with reduced confidence scores to indicate interpolation
- Prints repair statistics showing the number of light repairs, heavy occlusions, and specific joints fixed