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

LOG_TEST_DETAILS_FIELD = "log_test_details"

LOG_TEST_DETAILS_INPUT = (
    "BOOLEAN",
    {
        "default": True,
        "advanced": True,
        "label_on": "log test details",
        "label_off": "hide test details from log",
        "tooltip": (
            "Log each comparison cell with its active LoRA files, artist tags, "
            "weights, route, and observable cache state."
        ),
    },
)

USE_ANIMA_ARTIST_MIXER_FIELD = "use_anima_artist_mixer"

USE_ANIMA_ARTIST_MIXER_INPUT = (
    "BOOLEAN",
    {
        "default": True,
        "advanced": True,
        "label_on": "use Anima Artist Mixer",
        "label_off": "use native artist tags",
        "tooltip": (
            "Use the optional Anima Artist Mixer for multi-artist Anima tests. "
            "Disable for non-Anima models or to force native artist-tag encoding."
        ),
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
    LOG_TEST_DETAILS_FIELD: LOG_TEST_DETAILS_INPUT,
    USE_ANIMA_ARTIST_MIXER_FIELD: USE_ANIMA_ARTIST_MIXER_INPUT,
    "color_mode": COLOR_MODE_INPUT,
}


__all__ = [
    "COLOR_MODE_INPUT",
    "FUTURE_NODE_COMPOSITOR_FIELDS",
    "LOG_TEST_DETAILS_FIELD",
    "LOG_TEST_DETAILS_INPUT",
    "USE_ANIMA_ARTIST_MIXER_FIELD",
    "USE_ANIMA_ARTIST_MIXER_INPUT",
    "SHOW_LORA_DETAILS_FIELD",
    "SHOW_LORA_DETAILS_INPUT",
]
