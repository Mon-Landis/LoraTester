from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.node_contract import (
    FUTURE_NODE_COMPOSITOR_FIELDS,
    LOG_TEST_DETAILS_FIELD,
    LOG_TEST_DETAILS_INPUT,
    SHOW_LORA_DETAILS_FIELD,
    USE_ANIMA_ARTIST_MIXER_FIELD,
    USE_ANIMA_ARTIST_MIXER_INPUT,
)
from lora_tester.nodes import LoraStackNode, LoraTesterSampler


class NodeContractTests(unittest.TestCase):
    def test_lora_detail_toggle_is_reserved(self) -> None:
        self.assertEqual(SHOW_LORA_DETAILS_FIELD, "show_lora_details")
        input_type, options = FUTURE_NODE_COMPOSITOR_FIELDS[SHOW_LORA_DETAILS_FIELD]
        self.assertEqual(input_type, "BOOLEAN")
        self.assertTrue(options["default"])

    def test_log_detail_toggle_is_advanced_and_enabled_by_default(self) -> None:
        self.assertEqual(LOG_TEST_DETAILS_FIELD, "log_test_details")
        self.assertIs(FUTURE_NODE_COMPOSITOR_FIELDS[LOG_TEST_DETAILS_FIELD], LOG_TEST_DETAILS_INPUT)
        input_type, options = LOG_TEST_DETAILS_INPUT
        self.assertEqual(input_type, "BOOLEAN")
        self.assertTrue(options["default"])
        self.assertTrue(options["advanced"])

    def test_color_modes_are_stable(self) -> None:
        modes, options = FUTURE_NODE_COMPOSITOR_FIELDS["color_mode"]
        self.assertEqual(modes, ["black", "white", "custom"])
        self.assertEqual(options["default"], "black")

    def test_anima_mixer_toggle_is_advanced_and_enabled_by_default(self) -> None:
        self.assertEqual(USE_ANIMA_ARTIST_MIXER_FIELD, "use_anima_artist_mixer")
        self.assertIs(
            FUTURE_NODE_COMPOSITOR_FIELDS[USE_ANIMA_ARTIST_MIXER_FIELD],
            USE_ANIMA_ARTIST_MIXER_INPUT,
        )
        input_type, options = USE_ANIMA_ARTIST_MIXER_INPUT
        self.assertEqual(input_type, "BOOLEAN")
        self.assertTrue(options["default"])
        self.assertTrue(options["advanced"])

    def test_dynamic_file_validators_explicitly_claim_every_combo_slot(self) -> None:
        direct = inspect.getfullargspec(LoraTesterSampler.VALIDATE_INPUTS)
        stack = inspect.getfullargspec(LoraStackNode.VALIDATE_INPUTS)
        self.assertIsNone(direct.varkw)
        self.assertIsNone(stack.varkw)
        self.assertTrue(
            {"lora_a_name", "lora_b_name", "lora_c_name"}.issubset(direct.args)
        )
        self.assertTrue(
            {f"lora_{index}_name" for index in range(1, 17)}.issubset(stack.args)
        )


if __name__ == "__main__":
    unittest.main()
