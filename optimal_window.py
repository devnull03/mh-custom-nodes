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
            Optimal window size that evenly divides the video with minimal overlap
        """
        max_allowed = base_window * (1 + alpha)

        if video_frames <= max_allowed:
            optimal = video_frames
        else:
            num_chunks = math.ceil(video_frames / max_allowed)
            optimal = math.ceil(video_frames / num_chunks)

        remainder = optimal % align
        if remainder != 0:
            optimal += align - remainder

        return (optimal,)


NODE_CLASS_MAPPINGS = {"mh_GetOptimalWindow": mh_GetOptimalWindow}

NODE_DISPLAY_NAME_MAPPINGS = {"mh_GetOptimalWindow": "MH Get Optimal Window"}
