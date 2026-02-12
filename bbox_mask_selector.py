import json
import torch


class mh_BBoxMaskSelector:
    """Select bounding boxes by index and convert to SAM3 prompt format.

    Input boxes are absolute pixel [x1, y1, x2, y2] from a detection node.
    Output is a SAM3-compatible prompt with normalized [cx, cy, w, h] coords.

    Conversion pipeline:
        pixel [x1,y1,x2,y2]  →  normalize by image size  →  [cx,cy,w,h]
        → wrap in {"boxes": [...], "labels": [...]}
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "JSON array of bounding boxes in pixel coords, "
                        "each [x1, y1, x2, y2]. Connect from a detection node."
                    ),
                }),
                "image_width": ("INT", {
                    "default": 576,
                    "min": 1,
                    "tooltip": "Width of the source image in pixels (for normalization)",
                }),
                "image_height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "tooltip": "Height of the source image in pixels (for normalization)",
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

    @staticmethod
    def _pixel_to_sam3(box, img_w, img_h):
        """Convert pixel [x1, y1, x2, y2] → normalized [cx, cy, w, h]."""
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h
        return [cx, cy, w, h]

    def select(self, boxes, image_width, image_height, indexes, box_type, masks=None):
        empty_prompt = {"boxes": [], "labels": []}

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

        # ── convert & build SAM3 boxes prompt ────────────────────────────
        label_value = box_type == "positive"
        converted_boxes = [
            self._pixel_to_sam3(all_boxes[i], image_width, image_height)
            for i in valid_idxs
        ]

        prompt = {
            "boxes": converted_boxes,
            "labels": [label_value] * len(converted_boxes),
        }

        # log both raw and converted for debugging
        for i, idx in enumerate(valid_idxs):
            raw = all_boxes[idx]
            conv = converted_boxes[i]
            print(
                f"[MH BBox Selector]   #{idx}: "
                f"pixel [{raw[0]:.1f}, {raw[1]:.1f}, {raw[2]:.1f}, {raw[3]:.1f}] → "
                f"norm  [{conv[0]:.4f}, {conv[1]:.4f}, {conv[2]:.4f}, {conv[3]:.4f}]"
            )

        # ── select masks ─────────────────────────────────────────────────
        selected_masks = None
        if masks is not None:
            try:
                selected_masks = masks[valid_idxs]
            except (IndexError, RuntimeError) as e:
                print(f"[MH BBox Selector] Warning: mask selection failed — {e}")
                selected_masks = torch.zeros(0, *masks.shape[1:])

        if selected_masks is None:
            selected_masks = torch.zeros(0)

        print(
            f"[MH BBox Selector] Selected {len(valid_idxs)} of {num_boxes} "
            f"boxes (indexes: {valid_idxs}, type: {box_type}, "
            f"img: {image_width}x{image_height})"
        )

        return (prompt, selected_masks)


NODE_CLASS_MAPPINGS = {
    "mh_BBoxMaskSelector": mh_BBoxMaskSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mh_BBoxMaskSelector": "MH BBox & Mask Selector",
}
