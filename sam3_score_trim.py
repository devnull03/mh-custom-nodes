import torch
import numpy as np


class mh_SAM3FindValidRange:
    """
    Analyzes SAM3 propagation masks (primary) or scores (fallback) to find
    the frame range where the tracked object actually exists. Outputs
    start/end frame indices and total count, useful for trimming empty
    frames before cropping.

    Masks are the ground truth — scores can report 1.0 even on frames
    where the mask is entirely False (no object pixels).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_masks": ("SAM3_VIDEO_MASKS",),
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
                "min_pixel_ratio": (
                    "FLOAT",
                    {
                        "default": 0.0001,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.0001,
                        "tooltip": (
                            "Minimum fraction of mask pixels that must be True "
                            "for a frame to count as valid. 0 means any True pixel counts."
                        ),
                    },
                ),
            },
            "optional": {
                "video_state": ("SAM3_VIDEO_STATE",),
                "sam3_scores": ("SAM3_VIDEO_SCORES",),
                "score_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Fallback: minimum confidence score when masks "
                            "are not available"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("start_frame", "end_frame", "frame_count")
    FUNCTION = "calculate_range"
    CATEGORY = "MH/SAM3"

    def calculate_range(
        self,
        sam3_masks,
        padding,
        min_pixel_ratio,
        video_state=None,
        sam3_scores=None,
        score_threshold=0.3,
    ):
        valid_frames = []

        # ---- primary: use masks ----
        if sam3_masks is not None and len(sam3_masks) > 0:
            for frame_idx, mask in sam3_masks.items():
                mask_arr = np.asarray(mask)

                # empty array → no object
                if mask_arr.size == 0:
                    continue

                true_ratio = mask_arr.sum() / mask_arr.size

                if true_ratio >= min_pixel_ratio and mask_arr.any():
                    valid_frames.append(int(frame_idx))

            method = "masks"

        # ---- fallback: use scores ----
        elif sam3_scores is not None and len(sam3_scores) > 0:
            for frame_idx, scores in sam3_scores.items():
                if isinstance(scores, torch.Tensor):
                    max_score = (
                        scores.max().item() if scores.numel() > 0 else 0.0
                    )
                elif isinstance(scores, (list, tuple)):
                    max_score = max(scores) if scores else 0.0
                elif isinstance(scores, (int, float)):
                    max_score = float(scores)
                else:
                    max_score = 0.0

                if max_score >= score_threshold:
                    valid_frames.append(int(frame_idx))

            method = "scores"

        else:
            print(
                "[MH SAM3 Trim] Warning: neither masks nor scores provided. "
                "Returning zeros."
            )
            return (0, 0, 0)

        # ---- compute range ----
        if not valid_frames:
            total = len(sam3_masks or sam3_scores or {})
            print(
                f"[MH SAM3 Trim] Warning: no valid frames found via {method} "
                f"(checked {total} frames). Returning zeros."
            )
            return (0, 0, 0)

        valid_frames.sort()
        raw_start = valid_frames[0]
        raw_end = valid_frames[-1]

        # use video_state.num_frames if available, else infer from keys
        if video_state is not None and hasattr(video_state, "num_frames"):
            last_frame = video_state.num_frames - 1
        else:
            last_frame = max(
                (max(sam3_masks.keys()) if sam3_masks else 0),
                (max(sam3_scores.keys()) if sam3_scores else 0),
            )

        start_idx = max(0, raw_start - padding)
        end_idx = min(raw_end + padding, last_frame)

        count = end_idx - start_idx + 1

        print(
            f"[MH SAM3 Trim] ({method}) Object detected frames "
            f"{raw_start}–{raw_end} "
            f"({len(valid_frames)}/{len(sam3_masks or sam3_scores)} valid). "
            f"Output range {start_idx}–{end_idx} (count {count})"
        )

        return (start_idx, end_idx, count)


NODE_CLASS_MAPPINGS = {
    "mh_SAM3FindValidRange": mh_SAM3FindValidRange,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mh_SAM3FindValidRange": "MH SAM3 Find Valid Frame Range",
}
