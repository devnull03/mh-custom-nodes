class MH_ValueHook:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "FLOAT",
                    {"default": 1.0, "step": 0.01, "min": -10000.0, "max": 10000.0},
                ),
                "name": ("STRING", {"default": "param"}),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT", "HOOK_REF")
    RETURN_NAMES = ("float", "int", "hook_ref")
    FUNCTION = "do_hook"
    CATEGORY = "MH/Experiment"

    def do_hook(self, value, name):
        return (
            float(value),
            int(value),
            {"name": name, "value": value},
        )


class MH_ExperimentHub:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_runs": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
            },
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

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("run_index",)
    FUNCTION = "process"
    CATEGORY = "MH/Experiment"

    def process(self, num_runs, **kwargs):
        connected_hooks = {k: v for k, v in kwargs.items() if v is not None}

        if connected_hooks:
            print(
                f"[MH_ExperimentHub] num_runs={num_runs}, hooks={list(connected_hooks.keys())}"
            )

        return (0,)


NODE_CLASS_MAPPINGS = {
    "MH_ValueHook": MH_ValueHook,
    "MH_ExperimentHub": MH_ExperimentHub,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MH_ValueHook": "MH Value Hook",
    "MH_ExperimentHub": "MH Experiment Hub",
}
