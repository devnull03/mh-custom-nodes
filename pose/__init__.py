"""
pose/__init__.py

MH Pose package - Pose rendering and repair nodes for ComfyUI.
"""

from .nodes import MH_RenderPose, MH_RepairDWPose

NODE_CLASS_MAPPINGS = {
    "MH_RenderPose": MH_RenderPose,
    "MH_RepairDWPose": MH_RepairDWPose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MH_RenderPose": "MH Render Pose",
    "MH_RepairDWPose": "MH Repair DWPose",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    # Re-export utilities for external use
    "MH_RenderPose",
    "MH_RepairDWPose",
]
