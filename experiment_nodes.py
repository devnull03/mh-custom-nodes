import os
import time

import folder_paths
import numpy as np
from PIL import Image


class ValueHook:
    """
    The 'Hook'. Acts as a passthrough for the value, but also registers
    the start time of the execution and provides a reference for the Controller.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "FLOAT",
                    {"default": 1.0, "step": 0.01, "min": -10000.0, "max": 10000.0},
                ),
                "name": ("STRING", {"default": "param_1"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("FLOAT", "INT", "HOOK_REF")
    RETURN_NAMES = ("float", "int", "hook_ref")
    FUNCTION = "do_hook"
    CATEGORY = "MH/Experiment"

    def do_hook(self, value, name, unique_id=None):
        """
        Pass through the value while capturing metadata for experiment tracking.
        Returns float, int (truncated), and a hook reference dict.
        """
        return (
            float(value),
            int(value),
            {"name": name, "value": value, "time": time.time(), "unique_id": unique_id},
        )


class ExperimentController:
    """
    The 'Hub'. Accepts multiple Hook connections.
    Logic mostly lives in JS for UI interactions.
    Python side acts as an anchor for connections and passes through data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "hook_1": ("HOOK_REF",),
                "hook_2": ("HOOK_REF",),
                "hook_3": ("HOOK_REF",),
                "hook_4": ("HOOK_REF",),
                "hook_5": ("HOOK_REF",),
                "hook_6": ("HOOK_REF",),
                "hook_7": ("HOOK_REF",),
                "hook_8": ("HOOK_REF",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "process"
    CATEGORY = "MH/Experiment"
    OUTPUT_NODE = True

    def process(self, **kwargs):
        """
        Process connected hooks. The actual experiment orchestration
        happens in JavaScript; this just validates connections.
        """
        connected_hooks = {k: v for k, v in kwargs.items() if v is not None}

        if connected_hooks:
            print(
                f"[ExperimentController] Connected hooks: {list(connected_hooks.keys())}"
            )
            for name, hook_ref in connected_hooks.items():
                if isinstance(hook_ref, dict):
                    print(
                        f"  - {name}: {hook_ref.get('name', 'unknown')} = {hook_ref.get('value', 'N/A')}"
                    )

        return {}


class ExperimentReporter:
    """
    The 'Sink'. Calculates generation time and writes Markdown report.
    Receives the final image and metadata from connected hooks.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Exp_Result"}),
                "primary_hook": ("HOOK_REF",),
            },
            "optional": {
                "hook_2": ("HOOK_REF",),
                "hook_3": ("HOOK_REF",),
                "hook_4": ("HOOK_REF",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_report"
    CATEGORY = "MH/Experiment"

    def save_report(
        self,
        images,
        filename_prefix,
        primary_hook,
        hook_2=None,
        hook_3=None,
        hook_4=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        """
        Save the image and append a row to the Markdown report.
        """
        # 1. Calculate Generation Time
        start_time = (
            primary_hook.get("time", time.time())
            if isinstance(primary_hook, dict)
            else time.time()
        )
        duration = time.time() - start_time

        # 2. Collect all hook parameters
        all_hooks = [primary_hook, hook_2, hook_3, hook_4]
        params = {}
        for hook in all_hooks:
            if hook and isinstance(hook, dict):
                name = hook.get("name", "unknown")
                value = hook.get("value", "N/A")
                params[name] = value

        # 3. Extract Experiment Metadata (injected by JS into extra_pnginfo)
        exp_data = {}
        if extra_pnginfo and isinstance(extra_pnginfo, dict):
            exp_data = extra_pnginfo.get("experiment_params", {})

        # Merge params from hooks and extra_pnginfo
        merged_params = {**params, **exp_data}

        # 4. Determine experiment ID and create output directory
        exp_id = merged_params.pop("experiment_id", None)
        if not exp_id:
            exp_id = f"Exp_{time.strftime('%Y%m%d_%H%M%S')}"

        sub_dir = os.path.join(self.output_dir, "Experiments", exp_id)
        os.makedirs(sub_dir, exist_ok=True)

        # 5. Save Image
        timestamp = int(time.time() * 1000)
        img_name = f"{filename_prefix}_{timestamp}.png"
        img_path = os.path.join(sub_dir, img_name)

        # Convert tensor to PIL and save
        # images shape: (B, H, W, C) where B is batch size
        img_array = 255.0 * images[0].cpu().numpy()
        img_pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        img_pil.save(img_path, compress_level=4)

        print(f"[ExperimentReporter] Saved image: {img_path}")

        # 6. Update Markdown Report
        md_path = os.path.join(sub_dir, "results.md")

        # Build column headers and values
        cols = ["GenTime (s)"] + list(merged_params.keys())
        vals = [f"{duration:.2f}"] + [str(v) for v in merged_params.values()]

        # Create header if file is new
        if not os.path.exists(md_path):
            header = "| Image | " + " | ".join(cols) + " |"
            separator = "| --- | " + " | ".join(["---"] * len(cols)) + " |"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Experiment Report: {exp_id}\n\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"{header}\n{separator}\n")

        # Append row
        row = f"| ![]({img_name}) | " + " | ".join(vals) + " |"
        with open(md_path, "a", encoding="utf-8") as f:
            f.write(row + "\n")

        print(f"[ExperimentReporter] Updated report: {md_path}")

        # Return UI data for ComfyUI preview
        return {
            "ui": {
                "images": [
                    {
                        "filename": img_name,
                        "subfolder": f"Experiments/{exp_id}",
                        "type": "output",
                    }
                ]
            }
        }


NODE_CLASS_MAPPINGS = {
    "MH_ValueHook": ValueHook,
    "MH_ExperimentController": ExperimentController,
    "MH_ExperimentReporter": ExperimentReporter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MH_ValueHook": "MH Value Hook",
    "MH_ExperimentController": "MH Experiment Controller",
    "MH_ExperimentReporter": "MH Experiment Reporter",
}
