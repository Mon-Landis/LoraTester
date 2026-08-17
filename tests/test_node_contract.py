from __future__ import annotations

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
)


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


if __name__ == "__main__":
    unittest.main()
