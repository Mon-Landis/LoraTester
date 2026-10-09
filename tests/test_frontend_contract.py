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
        self.assertIn("const DYNAMIC_LAYOUT_STATE", source)
        self.assertIn("const WORKFLOW_SIZE_RESTORED", source)
        self.assertIn("const MIN_WIDTH_APPLIED", source)
        self.assertIn("const shouldResize = semanticChanged", source)
        self.assertIn("const semanticChanged =", source)
        self.assertIn("if (!widget && !warning) return", source)
        self.assertIn("const inputsChanged =", source)

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
        self.assertIn('installDynamicCount(node, "prompt_count", PROMPT_GROUPS, 16)', source)
        self.assertIn('positive_prompt_${index + 1}', source)
        self.assertIn('Prompt ${match[1]}', source)
        self.assertNotIn('custom_separator', source)
        self.assertNotIn('separator_mode', source)
        self.assertIn("options.placeholder = label", source)
        self.assertIn('document.querySelectorAll("[node-id][node-type]")', source)
        self.assertIn('input.setAttribute("aria-label", label)', source)
        self.assertIn("workflow owns its saved dimensions", source)

    def test_artist_mode_labels_warning_and_upstream_stack_tracking(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn('const ARTIST_TAG_MODE = "__lora_tester_artist_tag__"', source)
        self.assertIn("ARTIST_MODE_OPTION_LABELS", source)
        self.assertIn("artistModeLabels(node, nodeName)", source)
        self.assertIn("Artist ${title} Tag Weight", source)
        self.assertIn('language === "zh" ? `风格 ${title}` : `Style ${title}`', source)
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
        self.assertIn("风格组合", source)

    def test_xy_frontend_tracks_axis_overrides_warnings_and_all_controls(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn('const AXIS_COMPOSER_NODE = "LoraTesterAxisComposer"', source)
        self.assertIn("function axisMetadataFromSource", source)
        self.assertIn('axis: { en: "Axis", zh: "轴" }', source)
        self.assertIn('StyleStack', source)
        self.assertIn("function setWidgetDisabled(widget, disabled, node = null)", source)
        self.assertIn('querySelectorAll?.("textarea, input, select, button")', source)
        self.assertIn("element.inert = disabled ? true", source)
        self.assertIn("const XY_DOM_OVERRIDES = new Map()", source)
        self.assertIn("new MutationObserver", source)
        self.assertIn("xyDomApplyScheduled", source)
        self.assertIn("function axisMetadata(node, inputName)", source)
        self.assertIn("function updateXyAxisState(node)", source)
        self.assertIn("function installXySourceObservers(node, nodeName)", source)
        self.assertIn('querySelector(selector)', source)
        self.assertIn("const originalConnectionsChange = node.onConnectionsChange", source)
        self.assertIn("const XY_WARNING_WIDGET =", source)
        self.assertIn("Large XY queue", source)
        self.assertIn("X and Y both modify", source)
        self.assertIn("refreshReactiveCollection(node, \"inputs\")", source)
        self.assertIn("refreshReactiveCollection(node, \"outputs\")", source)

    def test_random_generator_seed_has_localized_sentinel_display(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn("function installRandomGeneratorSeedDisplay(node)", source)
        self.assertIn("Number(widget.value) === -1", source)
        self.assertIn('activeLanguage() === "zh" ? "随机" : "Random"', source)
        self.assertIn("data-lora-tester-seed-display", source)
        self.assertIn("installRandomGeneratorSeedDisplay(node)", source)

    def test_anima_remap_warning_tracks_static_and_runtime_diagnosis(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn('const ANIMA_REMAP_WARNING_WIDGET =', source)
        self.assertIn("function createAnimaRemapWarningWidget(node, widgetName = ANIMA_REMAP_WARNING_WIDGET)", source)
        self.assertIn("function anima29BModelConnected(node)", source)
        self.assertIn("function anyLoraSelected(node, nodeName)", source)
        self.assertIn("function updateAnimaRemapWarning(node, nodeName)", source)
        self.assertIn("function xySamplerHasMultiArtistTest(node)", source)
        self.assertIn(
            'if (![TARGET_NODE, MULTI_PROMPT_NODE, XY_SAMPLER_NODE, FLOW_XY_NODE].includes(nodeName)) return;',
            source,
        )
        self.assertIn("__loraTesterAnimaRemapMessage", source)
        self.assertIn("message?.lora_tester_anima_remap?.[0]?.message", source)
        self.assertIn("originalOnExecuted", source)
        self.assertIn("[TARGET_NODE, MULTI_PROMPT_NODE, XY_SAMPLER_NODE, FLOW_XY_NODE]", source)

    def test_flattener_frontend_labels_and_axis_metadata(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn('const STACK_FLATTENER_NODE = "LoraStackFlattener"', source)
        self.assertEqual(source.count("LoraStackFlattener: {"), 3)
        self.assertNotIn('label_on: "加入原始组合"', source)
        self.assertNotIn('label_off: "仅输出独立单项"', source)
        self.assertIn("sourceName === STACK_FLATTENER_NODE", source)
        self.assertIn("styleStackCountFromSource(rawSource)", source)
        self.assertIn("flattenedStackHasSeparateOriginal(source, new Set(visited))", source)
        self.assertIn("function flattenedStackHasSeparateOriginal(node, visited = new Set())", source)
        self.assertIn("nodeName === STACK_FLATTENER_NODE", source)
        self.assertIn('new Set(["include_original", "weight_mode"])', source)
        self.assertIn('inherit: { en: "Inherit", zh: "继承" }', source)
        self.assertIn('normalize: { en: "Normalize", zh: "归一化" }', source)
        self.assertIn('dual: { en: "Dual", zh: "双行" }', source)
        self.assertIn("flattenedStackChildren(source, new Set(visited))", source)
        self.assertIn("if (entries.length === 1)", source)
        self.assertIn("name|trigger|strength", source)
        self.assertIn("entry.strength !== 1", source)

    def test_name_nodes_keep_localization_and_upstream_artist_tracking(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn('const STACK_NAME_NODE = "LoraStackName"', source)
        self.assertIn('const STACK_LIST_NAME_NODE = "LoraStackListName"', source)
        self.assertIn("if (sourceName === STACK_LIST_NAME_NODE)", source)
        self.assertIn("if (nodeName === STACK_NAME_NODE || nodeName === STACK_MIXER_STRENGTH_NODE)", source)
        self.assertIn('return stackEntryDataFromSource(firstSourceForInput(node, "lora_stack"), visited)', source)
        self.assertIn('custom_name: { en: "Style Name", zh: "风格名称" }', source)
        self.assertIn('index: { en: "Index (0-based; negative = all)"', source)
        self.assertIn("function updateReplacerAdvancedInput(node, nodeName)", source)
        self.assertIn('"Comfy.VueNodes.Enabled.change"', source)
        self.assertIn("properties.loraTesterReplacerAdvanced === true", source)
        self.assertIn("toggle.serialize = false", source)

    def test_artist_text_and_replacement_nodes_use_shared_frontend_paths(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn('const ARTIST_TEXT_NODE = "ArtistTagTextParser"', source)
        self.assertIn('const ARTIST_REPLACER_NODE = "ArtistTagReplacer"', source)
        self.assertIn('replace: { en: "Replace", zh: "替换" }', source)
        self.assertIn('multiply: { en: "Multiply", zh: "倍率" }', source)
        self.assertIn('lora_1_name: { en: "Replacement LoRA / Artist Mode"', source)
        self.assertIn("artistEntriesFromStackSource", source)
        self.assertIn("normalizeArtistMatchTag", source)
        self.assertIn('new Set(["artist_text"])', source)
        self.assertIn('new Set(["match_tag", "lora_1_name", "lora_1_trigger", "lora_1_strength", "strength_mode"])', source)
        self.assertIn("GLOBAL_PROMPT_APPEND_NODE, PROMPT_LIST_NAME_NODE, ARTIST_TEXT_NODE", source)
        self.assertIn('installOptionLabels(widget, ARTIST_MODE_OPTION_LABELS)', source)
        self.assertIn("widgetOptionTargets(widget)", source)

    def test_prompt_names_use_shared_multiline_layout_and_axis_metadata(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn('const PROMPT_LIST_NAME_NODE = "LoraTesterPromptListName"', source)
        self.assertIn('nodeName === PROMPT_LIST_NAME_NODE && widget.name === "names"', source)
        self.assertIn('names: { en: "Names (One per Line)", zh: "名称（每行一项）" }', source)
        self.assertIn("rawName === PROMPT_LIST_NAME_NODE", source)
        self.assertIn("const hasLongPromptLayout = [MULTI_PROMPT_INPUT_NODE, GLOBAL_PROMPT_APPEND_NODE, PROMPT_LIST_NAME_NODE", source)

    def test_axis_preview_uses_readonly_multiline_widgets_and_preserves_outputs(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        self.assertIn('const AXIS_PREVIEW_NODE = "LoraTesterAxisPreview"', source)
        self.assertIn('import { ComfyWidgets } from "../../scripts/widgets.js"', source)
        self.assertIn("function installAxisPreview(node)", source)
        self.assertIn("function updateAxisPreview(node, message)", source)
        self.assertIn("widget.serialize = false", source)
        self.assertIn("options.read_only = true", source)
        self.assertIn('options.wrap = "off"', source)
        self.assertIn("options.minNodeSize = [560, 340]", source)
        self.assertIn("textarea.readOnly = true", source)
        self.assertIn('whiteSpace: "pre"', source)
        self.assertIn("loraTesterAxisPreviewText", source)
        self.assertIn("if (nodeData.name === AXIS_PREVIEW_NODE) updateAxisPreview(this, message)", source)
        self.assertIn("onNodeOutputsUpdated(nodeOutputs)", source)
        self.assertIn('axisMetadataFromSource(firstSourceForInput(source, "axis"), visited)', source)

    def test_anima_flow_nodes_share_xy_lifecycle_and_missing_dependency_ui(self) -> None:
        source = (ROOT / "web" / "lora_tester.js").read_text(encoding="utf-8")
        for contract in (
            'const FLOW_XY_NODE = "LoraTesterAnimaFlowXYSampler"',
            'const FLOW_AXIS_NODE = "LoraTesterAnimaFlowParameterAxis"',
            'api.fetchApi("/lora_tester/anima_flow/status")',
            "function updateAnimaFlowWarning(node, nodeName)",
            'node.color = "#7f1d1d"',
            "link.href = FLOW_DEPENDENCY_URL",
            "loraTesterFlowOriginalColors",
            "const isFlowCombo",
            'nodeData.name === XY_SAMPLER_NODE || nodeData.name === FLOW_XY_NODE',
            "function updateAnimaFlowAxisHint(node)",
        ):
            self.assertIn(contract, source)

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
                self.assertIn("Axes", main["nodeCategories"])
                self.assertIn("Deprecated", main["nodeCategories"])
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
                    "LoraStackFlattener",
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
