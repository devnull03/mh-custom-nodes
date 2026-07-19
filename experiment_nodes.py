import json


class AnyType(str):
    """A string subclass that compares equal to every other type.

    ComfyUI decides whether two node slots may connect by comparing their type
    strings. By making ``__ne__`` always return ``False`` (and ``__eq__`` always
    return ``True``) an ``AnyType("*")`` output is treated as type-compatible with
    any downstream input (INT, FLOAT, STRING, a COMBO widget-input, etc.). This is
    the standard ComfyUI "wildcard" trick and is what lets a single
    ``MH_BatchInputSpecifier`` feed whatever parameter you wire it into.
    """

    def __ne__(self, _other):
        return False

    def __eq__(self, _other):
        return True

    def __hash__(self):
        return hash(str(self))


ANY = AnyType("*")


class MH_BatchInputSpecifier:
    """Marks a single workflow parameter for a client-side batch sweep.

    The real work happens in the browser (see web/experimentRunner.js): the
    frontend clones the *connected* input's widget so every swept value is valid
    by construction, then queues one fully-resolved prompt per combination.

    On the Python side this node only needs to:
      * expose a wildcard output so it can be linked into any widget-input, and
      * produce one valid value for a plain single run (i.e. when you just press
        "Queue Prompt" without going through "Run Experiment Grid").

    All of the node's state lives in a single serialized ``payload`` widget so it
    round-trips through workflow save/load. The frontend keeps it in sync as:
        {"type": "INT" | "FLOAT" | "STRING" | "COMBO", "values": [...]}
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # A JSON blob maintained entirely by the frontend. It is a real
                # (serialized) widget so that (a) it persists in the saved
                # workflow and (b) the backend can read the sweep values here.
                # The JS hides it visually (non-multiline: a canvas-drawn widget
                # with no DOM textarea is trivial to collapse).
                "payload": ("STRING", {"default": "{}", "multiline": False}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "passthrough"
    CATEGORY = "MH/Experiment"

    def passthrough(self, payload):
        # For a normal single run we emit the FIRST value in the list, coerced to
        # the declared type. During "Run Experiment Grid" the frontend overwrites
        # the *downstream* node's input directly for each combination, which
        # orphans this node in those payloads (ComfyUI then prunes it) -- so this
        # value only ever matters for a plain single queue.
        try:
            data = json.loads(payload) if payload else {}
        except (ValueError, TypeError):
            data = {}

        value_type = data.get("type", "STRING")
        values = data.get("values") or []
        first = values[0] if values else None

        return (self._coerce(first, value_type),)

    @staticmethod
    def _coerce(value, value_type):
        # COMBO values are already strings from a fixed option set; leave as-is.
        if value_type == "INT":
            try:
                return int(round(float(value)))
            except (ValueError, TypeError):
                return 0
        if value_type == "FLOAT":
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        # STRING / COMBO / unknown
        return "" if value is None else str(value)


NODE_CLASS_MAPPINGS = {
    "MH_BatchInputSpecifier": MH_BatchInputSpecifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MH_BatchInputSpecifier": "MH Batch Input Specifier",
}
