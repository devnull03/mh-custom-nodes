"""
pose/__init__.py

MH Pose package - Pose rendering, repair, and scale calculation nodes for ComfyUI.
"""

from .nodes import (
    MH_AutoPoseScaleCalculator,
    MH_RenderPose,
    MH_RepairDWPose,
)

NODE_CLASS_MAPPINGS = {
    "MH_RenderPose": MH_RenderPose,
    "MH_RepairDWPose": MH_RepairDWPose,
    "MH_AutoPoseScaleCalculator": MH_AutoPoseScaleCalculator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MH_RenderPose": "MH Render Pose",
    "MH_RepairDWPose": "MH Repair DWPose",
    "MH_AutoPoseScaleCalculator": "MH Auto Pose Scale Calculator",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    # Re-export utilities for external use
    "MH_RenderPose",
    "MH_RepairDWPose",
    "MH_AutoPoseScaleCalculator",
]
