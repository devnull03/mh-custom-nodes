import importlib
import os
import pkgutil

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

package_dir = os.path.dirname(__file__)

# Subpackages to always load explicitly (so they are detected even if
# iter_modules order or package naming differs, e.g. mh-custom-nodes vs mh_custom_nodes).
_subpackages = ["pose", "masking"]

for sub in _subpackages:
    try:
        module = importlib.import_module(f".{sub}", package=__name__)
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        print(f"[MH] Failed to import {sub}: {e}")

for module_info in pkgutil.iter_modules([package_dir]):
    if module_info.name.startswith("_") or module_info.name == "web" or module_info.name in _subpackages:
        continue

    try:
        module = importlib.import_module(f".{module_info.name}", package=__name__)

        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)

        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

    except Exception as e:
        print(f"[MH] Failed to import {module_info.name}: {e}")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
