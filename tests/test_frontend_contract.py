from __future__ import annotations

import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.nodes import LoraTesterSampler, LoraTesterStyleNode


class FrontendContractTests(unittest.TestCase):
    def test_plugin_exports_web_directory(self) -> None:
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))
        plugin = importlib.import_module(ROOT.name)
        self.assertEqual(plugin.WEB_DIRECTORY, "./web")

    def test_dynamic_extension_tracks_all_optional_lora_widgets(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn('const TARGET_NODE = "LoraTesterSampler"', source)
        self.assertIn("lora_b_name", source)
        self.assertIn("lora_b_trigger", source)
        self.assertIn("lora_b_max_strength", source)
        self.assertIn("lora_c_name", source)
        self.assertIn("lora_c_trigger", source)
        self.assertIn("lora_c_max_strength", source)
        self.assertIn("originalOnConfigure", source)
        self.assertIn("updateLoraGroups", source)
        self.assertIn("HIDDEN_WIDGET_TYPE", source)
        self.assertIn("node.graph?.incrementVersion?.()", source)
        self.assertIn("resizeNodeToWidgets(node)", source)
        self.assertIn("refreshWidgetViews(node)", source)
        self.assertIn("canvas.selectItems?.(selected, false)", source)

    def test_frontend_localizes_stable_enum_values_without_changing_them(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn('get?.("Comfy.Locale")', source)
        self.assertIn('getSettingValue?.("Comfy.Locale")', source)
        self.assertIn("options.getOptionLabel", source)
        self.assertIn("installWidgetTranslations", source)

        expected_values = {
            "color_mode": ("black", "white", "custom"),
            "background_fit": ("cover", "contain", "stretch", "tile"),
            "decorator": ("none", "technical"),
        }
        for field, values in expected_values.items():
            with self.subTest(field=field):
                self.assertIn(f"{field}:", source)
                for value in values:
                    self.assertIn(f"{value}:", source)

        self.assertIn("show_lora_details", source)
        self.assertIn("options.label_on", source)
        self.assertIn("options.label_off", source)
        self.assertIn("options.on", source)
        self.assertIn("options.off", source)

    def test_english_and_chinese_locales_cover_both_nodes(self) -> None:
        with (
            patch("lora_tester.nodes._get_lora_names", return_value=["A.safetensors"]),
            patch("lora_tester.nodes._get_sampler_names", return_value=("sampler",)),
            patch("lora_tester.nodes._get_scheduler_names", return_value=("scheduler",)),
        ):
            sampler_inputs = LoraTesterSampler.INPUT_TYPES()
        sampler_names = set(sampler_inputs["required"]) | set(sampler_inputs["optional"])
        style_inputs = LoraTesterStyleNode.INPUT_TYPES()
        style_names = set(style_inputs["required"]) | set(style_inputs["optional"])

        for locale in ("en", "zh"):
            with self.subTest(locale=locale):
                locale_root = ROOT / "locales" / locale
                main = json.loads((locale_root / "main.json").read_text(encoding="utf-8"))
                node_defs = json.loads(
                    (locale_root / "nodeDefs.json").read_text(encoding="utf-8")
                )
                self.assertIn("Lora Tester", main["nodeCategories"])
                self.assertEqual(
                    set(node_defs),
                    {"LoraTesterSampler", "LoraTesterStyle"},
                )
                self.assertEqual(
                    set(node_defs["LoraTesterSampler"]["inputs"]),
                    sampler_names,
                )
                self.assertEqual(
                    set(node_defs["LoraTesterStyle"]["inputs"]),
                    style_names,
                )
                self.assertTrue(node_defs["LoraTesterSampler"]["display_name"])
                self.assertTrue(node_defs["LoraTesterStyle"]["display_name"])
                self.assertEqual(
                    set(node_defs["LoraTesterSampler"]["inputs"]["color_mode"]["values"]),
                    set(sampler_inputs["required"]["color_mode"][0]),
                )
                self.assertEqual(
                    set(node_defs["LoraTesterStyle"]["inputs"]["background_fit"]["values"]),
                    set(style_inputs["required"]["background_fit"][0]),
                )
                localized_decorators = set(
                    node_defs["LoraTesterStyle"]["inputs"]["decorator"]["values"]
                )
                self.assertEqual(localized_decorators, {"none", "technical"})
                self.assertLessEqual(
                    localized_decorators,
                    set(style_inputs["required"]["decorator"][0]),
                )


if __name__ == "__main__":
    unittest.main()
