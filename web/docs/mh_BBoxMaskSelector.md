# BBox & Mask Selector

Select one or more bounding boxes (and their matching masks) by index from a detection output. Outputs a SAM3-compatible boxes prompt with labels.

## Parameters

- **boxes**: JSON string containing an array of `[x1, y1, x2, y2]` bounding boxes
- **indexes**: Comma-separated 0-based indices to select, e.g. `"0,2,4"`
- **box_type**: `positive` or `negative` — controls whether the output labels are `true` (include region) or `false` (exclude region)
- **masks** *(optional)*: Standard mask tensor `[N, H, W]` matching the boxes ordering. Selected masks are returned as a subset.

## Outputs

- **sam3_boxes_prompt**: SAM3 boxes prompt dict with `boxes` and `labels` arrays, e.g. `{"boxes": [[x1,y1,x2,y2]], "labels": [true]}`
- **selected_masks**: Mask tensor containing only the masks at the selected indices

## Notes

- Out-of-range indices are warned and skipped
- Returns empty prompt and empty masks if no valid indices are selected
- Boxes are passed through as-is (no coordinate normalization)
