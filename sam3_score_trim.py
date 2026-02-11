import torch


class mh_SAM3FindValidRange:
    """
    Analyzes SAM3 propagation scores to find the frame range where the
    tracked object actually exists. Outputs start/end frame indices and
    total count, useful for trimming empty frames before cropping.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_scores": ("SAM3_VIDEO_SCORES",),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Minimum confidence score to consider object present",
                    },
                ),
                "padding": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 120,
                        "step": 1,
                        "tooltip": "Extra buffer frames to keep around the valid range",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("start_frame", "end_frame", "frame_count")
    FUNCTION = "calculate_range"
    CATEGORY = "MH/SAM3"

    def calculate_range(self, sam3_scores, threshold, padding):
        valid_frames = []

        for frame_idx, scores in sam3_scores.items():
            if isinstance(scores, torch.Tensor):
                max_score = scores.max().item() if scores.numel() > 0 else 0.0
            elif isinstance(scores, (list, tuple)):
                max_score = max(scores) if scores else 0.0
            elif isinstance(scores, (int, float)):
                max_score = float(scores)
            else:
                max_score = 0.0

            if max_score >= threshold:
                valid_frames.append(int(frame_idx))

        if not valid_frames:
            print(
                "[MH SAM3 Trim] Warning: no frames found above threshold "
                f"{threshold:.2f}. Returning zeros."
            )
            return (0, 0, 0)

        valid_frames.sort()
        raw_start = valid_frames[0]
        raw_end = valid_frames[-1]

        start_idx = max(0, raw_start - padding)
        end_idx = raw_end + padding

        count = end_idx - start_idx + 1

        print(
            f"[MH SAM3 Trim] Object detected frames {raw_start}-{raw_end} "
            f"({len(valid_frames)}/{len(sam3_scores)} above {threshold:.2f}). "
            f"Output range {start_idx}-{end_idx} (count {count})"
        )

        return (start_idx, end_idx, count)


NODE_CLASS_MAPPINGS = {
    "mh_SAM3FindValidRange": mh_SAM3FindValidRange,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mh_SAM3FindValidRange": "MH SAM3 Find Valid Frame Range",
}
