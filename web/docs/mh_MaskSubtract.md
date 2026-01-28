# Mask Subtract

Subtracts one mask from another and clamps the result to [0, 1]. Useful for removing overlapping regions, such as removing a ball from a person mask.

## Parameters

- **mask_a**: Primary mask (the one to subtract from)
- **mask_b**: Mask to subtract from mask_a

## Outputs

- **mask**: Result of (mask_a - mask_b), clamped to [0, 1]

## Notes

- Automatically broadcasts single-frame masks to match batch sizes
- Both masks must have matching spatial dimensions