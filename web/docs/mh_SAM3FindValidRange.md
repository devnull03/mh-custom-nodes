# SAM3 Find Valid Frame Range

Analyzes SAM3 propagation masks to find the frame range where the tracked object actually exists. Uses mask pixel coverage as the primary signal — scores can report 1.0 even on frames where the mask is entirely empty.

## Parameters

- **sam3_masks** *(required)*: Mask dict from SAM3 video propagation. Each frame maps to a boolean array; empty arrays or all-False frames are treated as "no object".
- **padding**: Extra buffer frames to keep around the detected range (default: 0)
- **min_pixel_ratio**: Minimum fraction of mask pixels that must be True for a frame to count as valid. Set to 0 to accept any True pixel (default: 0.0001)
- **video_state** *(optional)*: SAM3 video state, provides `num_frames` for a reliable upper bound when clamping padding
- **sam3_scores** *(optional)*: Fallback score dict, used only when masks are not available
- **score_threshold** *(optional)*: Minimum confidence score for the scores-based fallback (default: 0.3)

## Outputs

- **start_frame**: First frame index where the object is detected (with padding applied)
- **end_frame**: Last frame index where the object is detected (with padding applied, clamped to total frames)
- **frame_count**: Total frames in the output range (end - start + 1)

## Notes

- Masks are the ground truth — scores alone are unreliable because SAM3 can report high confidence on frames with zero mask pixels
- Empty mask arrays (shape `(0, H, W)`) and all-False masks are both treated as "object not present"
- When `video_state` is connected, `end_frame + padding` is clamped to `num_frames - 1`; otherwise clamps to the max key in the mask/score dicts
- Returns `(0, 0, 0)` if no valid frames are found
