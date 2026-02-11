import json
import torch


class mh_BBoxMaskSelector:
    """Select one or more bounding boxes (and their matching masks) by index.

    Indexes are given as a comma-separated string, e.g. "0,2,4".
    Boxes input is a JSON string containing a list of [x1, y1, x2, y2] arrays.
    Masks input is a dict keyed by frame index (matching the boxes ordering).

    Output is a SAM3 boxes prompt dict:
        {"boxes": [[x1,y1,x2,y2], ...], "labels": [true, ...]}
    Labels are true for positive boxes, false for negative.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": ("STRING", {
                    "multiline": True,
                    "tooltip": (
                        "JSON array of bounding boxes, each [x1, y1, x2, y2]. "
                        "Example: [[10, 20, 100, 200], [50, 60, 150, 250]]"
                    ),
                }),
                "indexes": ("STRING", {
                    "default": "0",
                    "tooltip": (
                        "Comma-separated indices to select, e.g. '0,2,4'. "
                        "Indices are 0-based."
                    ),
                }),
                "box_type": (["positive", "negative"], {
                    "default": "positive",
                    "tooltip": (
                        "Positive boxes label the object to segment. "
                        "Negative boxes label regions to exclude."
                    ),
                }),
            },
            "optional": {
                "masks": ("MASK",),
            },
        }

    RETURN_TYPES = ("SAM3_BOXES_PROMPT", "MASK")
    RETURN_NAMES = ("sam3_boxes_prompt", "selected_masks")
    FUNCTION = "select"
    CATEGORY = "MH/SAM3"

    def select(self, boxes, indexes, box_type, masks=None):
        empty_prompt = json.dumps({"boxes": [], "labels": []})

        # ── parse boxes ──────────────────────────────────────────────────
        try:
            all_boxes = json.loads(boxes)
        except json.JSONDecodeError as e:
            print(f"[MH BBox Selector] Error: could not parse boxes JSON — {e}")
            return (empty_prompt, torch.zeros(0))

        if not isinstance(all_boxes, list):
            print("[MH BBox Selector] Error: boxes must be a JSON array")
            return (empty_prompt, torch.zeros(0))

        # ── parse indexes ────────────────────────────────────────────────
        try:
            selected_idxs = [int(i.strip()) for i in indexes.split(",") if i.strip()]
        except ValueError as e:
            print(f"[MH BBox Selector] Error: bad index value — {e}")
            return (empty_prompt, torch.zeros(0))

        # ── validate indexes ─────────────────────────────────────────────
        num_boxes = len(all_boxes)
        valid_idxs = []
        for idx in selected_idxs:
            if 0 <= idx < num_boxes:
                valid_idxs.append(idx)
            else:
                print(
                    f"[MH BBox Selector] Warning: index {idx} out of range "
                    f"(0–{num_boxes - 1}), skipping."
                )

        if not valid_idxs:
            print("[MH BBox Selector] Warning: no valid indexes selected.")
            return (empty_prompt, torch.zeros(0))

        # ── build SAM3 boxes prompt ──────────────────────────────────────
        label_value = box_type == "positive"  # True for +ve, False for -ve
        selected_boxes = [all_boxes[i] for i in valid_idxs]
        prompt = {
            "boxes": selected_boxes,
            "labels": [label_value] * len(selected_boxes),
        }
        prompt_json = json.dumps(prompt)

        # ── select masks ─────────────────────────────────────────────────
        selected_masks = None
        if masks is not None:
            try:
                selected_masks = masks[valid_idxs]  # tensor index with list
            except (IndexError, RuntimeError) as e:
                print(f"[MH BBox Selector] Warning: mask selection failed — {e}")
                selected_masks = torch.zeros(0, *masks.shape[1:])

        if selected_masks is None:
            selected_masks = torch.zeros(0)

        print(
            f"[MH BBox Selector] Selected {len(valid_idxs)} of {num_boxes} "
            f"boxes (indexes: {valid_idxs}, type: {box_type})"
        )

        return (prompt_json, selected_masks)


NODE_CLASS_MAPPINGS = {
    "mh_BBoxMaskSelector": mh_BBoxMaskSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mh_BBoxMaskSelector": "MH BBox & Mask Selector",
}
