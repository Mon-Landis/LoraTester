from __future__ import annotations


# This field name is shared by the compositor API and the future sampler node.
SHOW_LORA_DETAILS_FIELD = "show_lora_details"

SHOW_LORA_DETAILS_INPUT = (
    "BOOLEAN",
    {
        "default": True,
        "label_on": "show names and min/max weights",
        "label_off": "hide names and min/max weights",
        "tooltip": "Show the original A/B/C LoRA names and configured minimum/maximum weights in the footer.",
    },
)

COLOR_MODE_INPUT = (
    ["black", "white", "custom"],
    {
        "default": "black",
        "tooltip": "Select the comparison-sheet background and text contrast preset.",
    },
)

FUTURE_NODE_COMPOSITOR_FIELDS = {
    SHOW_LORA_DETAILS_FIELD: SHOW_LORA_DETAILS_INPUT,
    "color_mode": COLOR_MODE_INPUT,
}


__all__ = [
    "COLOR_MODE_INPUT",
    "FUTURE_NODE_COMPOSITOR_FIELDS",
    "SHOW_LORA_DETAILS_FIELD",
    "SHOW_LORA_DETAILS_INPUT",
]
