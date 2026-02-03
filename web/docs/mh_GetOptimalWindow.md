# Get Optimal Window

Calculates the optimal frame window size for video processing pipelines. Ensures efficient chunking with minimal overlap while respecting alignment constraints.

## Parameters

- **video_frames**: Total number of frames in the video
- **base_window**: Base window size to target (default: 81)
- **alpha**: Allowable expansion factor as a fraction (default: 0.2 = 20% larger allowed)
- **align**: Window size must be divisible by this value (default: 4)

## Outputs

- **optimal_window**: Calculated window size (chunk size) for processing

## Notes

- If `video_frames` is less than `base_window * (1 + alpha)`, returns the video length directly
- Otherwise, calculates the minimum number of chunks needed and distributes frames evenly
- Final result is rounded up to meet the alignment requirement