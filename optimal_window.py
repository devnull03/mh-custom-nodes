import math


class mh_GetOptimalWindow:
    """
    Calculates the optimal frame window size based on video length.
    Ensures efficient chunking for video generation pipelines.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": (
                    "INT",
                    {"default": 81, "min": 1, "max": 10000, "step": 1},
                ),
                "base_window": (
                    "INT",
                    {"default": 81, "min": 1, "max": 1000, "step": 1},
                ),
                "alpha": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "align": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("optimal_window",)
    FUNCTION = "get_optimal_window"
    CATEGORY = "MH/Utils"

    def get_optimal_window(self, video_frames, base_window=81, alpha=0.2, align=4):
        """
        Calculate optimal window size for video frame processing.

        Args:
            video_frames: Total number of frames in the video
            base_window: Base window size (default 81)
            alpha: Allowable expansion factor (default 0.2 = 20% larger allowed)
            align: Alignment requirement for the window size (default 4)

        Returns:
            Optimal window size (chunk size) for processing the video.
            For short videos (<= max_allowed), returns exactly video_frames.
            For longer videos, returns an aligned chunk size.
        """
        max_allowed = base_window * (1 + alpha)

        if video_frames <= max_allowed:
            # Short video: video_frames + align buffer
            optimal = video_frames + align
        else:
            # Long video: calculate chunk size, then add align buffer
            num_chunks = math.ceil(video_frames / max_allowed)
            chunk_size = math.ceil(video_frames / num_chunks)
            optimal = chunk_size + align

        optimal = min(99, optimal)

        return (optimal,)


NODE_CLASS_MAPPINGS = {"mh_GetOptimalWindow": mh_GetOptimalWindow}

NODE_DISPLAY_NAME_MAPPINGS = {"mh_GetOptimalWindow": "MH Get Optimal Window"}
