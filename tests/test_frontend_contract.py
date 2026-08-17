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

from lora_tester.nodes import (
    NODE_CLASS_MAPPINGS,
    LoraTesterSampler,
    LoraTesterStyleNode,
)


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
        self.assertIn("lora_b_min_strength", source)
        self.assertIn("lora_b_max_strength", source)
        self.assertIn("lora_c_name", source)
        self.assertIn("lora_c_trigger", source)
        self.assertIn("lora_c_min_strength", source)
        self.assertIn("lora_c_max_strength", source)
        self.assertIn("originalOnConfigure", source)
        self.assertIn("originalOnAfterGraphConfigured", source)
        self.assertIn("updateLoraGroups", source)
        self.assertIn("HIDDEN_WIDGET_TYPE", source)
        self.assertIn("Reapply every target", source)
        self.assertIn("node.graph?.incrementVersion?.()", source)
        self.assertIn("resizeNodeToWidgets(node)", source)
        self.assertIn("refreshWidgetViews(node)", source)
        self.assertIn("canvas.selectItems?.(selected, false)", source)

    def test_node2_workflow_tabs_and_missing_loras_are_supported(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn("preserveUnavailableLoraValues", source)
        self.assertIn("values.includes(value)", source)
        self.assertIn("options.values = [...values, value]", source)
        self.assertIn("values.push(value)", source)
        self.assertIn("loadedGraphNode(node)", source)
        self.assertIn("const originalSetGraph = canvas.setGraph", source)
        self.assertIn("scheduleGraphNodeUi(this.graph)", source)
        self.assertIn("node[property] = []", source)
        self.assertIn("node[property] = snapshot", source)
        self.assertIn("if (!state.hiddenByLoraTester) {", source)
        self.assertIn("visible && !widget[WIDGET_STATE]", source)
        self.assertIn("delete options.hidden", source)
        self.assertIn("Visibility for these optional groups", source)

    def test_multi_prompt_widgets_keep_readable_layout_and_input_hints(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn("const MULTI_PROMPT_MIN_WIDTH = 480", source)
        self.assertIn("installMultiPromptLayout(node)", source)
        self.assertIn("options.placeholder = label", source)
        self.assertIn('document.querySelectorAll("[node-id][node-type]")', source)
        self.assertIn('input.setAttribute("aria-label", label)', source)

    def test_artist_mode_labels_warning_and_upstream_stack_tracking(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn('const ARTIST_TAG_MODE = "__lora_tester_artist_tag__"', source)
        self.assertIn("ARTIST_MODE_OPTION_LABELS", source)
        self.assertIn("artistModeLabels(node, nodeName)", source)
        self.assertIn("Artist ${title} Tag Weight", source)
        self.assertIn("countIndependentArtistTags", source)
        self.assertIn('"independent_artist_tags"', source)
        self.assertIn("independentArtists > 1", source)
        self.assertIn("stackArtists + independentArtists > 1", source)
        self.assertIn("stackArtistCountsFromNode", source)
        self.assertIn('registeredNodeAvailable("AnimaArtistPack")', source)
        self.assertIn('"AnimaArtistAdapterMixer"', source)
        self.assertIn('widgetValue(node, "use_anima_artist_mixer") !== false', source)
        self.assertIn("disable the advanced Mixer switch for non-Anima models", source)
        self.assertIn("updateMixerWarning(node, nodeName)", source)
        self.assertIn("node.addDOMWidget", source)
        self.assertIn("serialize: false", source)
        self.assertIn("node.onWidgetChanged = function", source)

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
        self.assertIn("log_test_details", source)
        self.assertIn("use_anima_artist_mixer", source)
        self.assertIn("options.label_on", source)
        self.assertIn("options.label_off", source)
        self.assertIn("options.on", source)
        self.assertIn("options.off", source)
        self.assertIn("installNodeLabels", source)
        self.assertIn('zh: "BASE 对照列间距"', source)
        self.assertIn("positive_prompt_", source)
        self.assertIn("LoRA 组合", source)

    def test_english_and_chinese_locales_cover_all_nodes(self) -> None:
        with (
            patch("lora_tester.nodes._get_lora_names", return_value=["A.safetensors"]),
            patch("lora_tester.nodes._get_sampler_names", return_value=("sampler",)),
            patch("lora_tester.nodes._get_scheduler_names", return_value=("scheduler",)),
        ):
            sampler_inputs = LoraTesterSampler.INPUT_TYPES()
            all_node_inputs = {
                node_name: node_class.INPUT_TYPES()
                for node_name, node_class in NODE_CLASS_MAPPINGS.items()
            }
        sampler_names = set(sampler_inputs["required"]) | set(sampler_inputs["optional"])
        style_inputs = LoraTesterStyleNode.INPUT_TYPES()
        style_names = set(style_inputs["required"]) | set(style_inputs["optional"])

        expected_display_names = {
            "en": {
                "LoraTesterSampler": "Style Component Tester",
                "MultiPromptSample": "Style Combination Tester",
            },
            "zh": {
                "LoraTesterSampler": "风格组件测试器",
                "MultiPromptSample": "风格组合测试器",
            },
        }
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
                    set(NODE_CLASS_MAPPINGS),
                )
                for node_name, input_types in all_node_inputs.items():
                    with self.subTest(locale=locale, node_name=node_name):
                        expected_inputs = set(input_types.get("required", {})) | set(
                            input_types.get("optional", {})
                        )
                        localized_inputs = node_defs[node_name].get("inputs", {})
                        self.assertEqual(set(localized_inputs), expected_inputs)
                        for translation in localized_inputs.values():
                            self.assertTrue(translation["name"])
                            self.assertTrue(translation["tooltip"])

                        expected_outputs = {
                            str(index)
                            for index in range(
                                len(NODE_CLASS_MAPPINGS[node_name].RETURN_TYPES)
                            )
                        }
                        localized_outputs = node_defs[node_name].get("outputs", {})
                        self.assertEqual(set(localized_outputs), expected_outputs)
                        for translation in localized_outputs.values():
                            self.assertTrue(translation["name"])
                            self.assertTrue(translation["tooltip"])

                self.assertEqual(
                    set(node_defs["LoraTesterSampler"]["inputs"]), sampler_names
                )
                self.assertEqual(set(node_defs["LoraTesterStyle"]["inputs"]), style_names)
                self.assertTrue(node_defs["LoraTesterSampler"]["display_name"])
                self.assertTrue(node_defs["LoraTesterStyle"]["display_name"])
                for node_name, display_name in expected_display_names[locale].items():
                    self.assertEqual(node_defs[node_name]["display_name"], display_name)
                for node_name in (
                    "LoraStack",
                    "LoraStackSplitter",
                    "LoraStackLister",
                    "MultiPromptSample",
                ):
                    self.assertTrue(node_defs[node_name]["display_name"])
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
