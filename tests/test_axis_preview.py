from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.artist import ARTIST_TAG_MODE, ArtistTagTemplate
from lora_tester.axis_preview import format_axis_preview
from lora_tester.nodes import AxisPreviewNode, NODE_CLASS_MAPPINGS
from lora_tester.stack import LoraStack, LoraStackItem, LoraStackList, flatten_lora_stack
from lora_tester.xy import (
    MAX_SEED, AxisEntry, AxisParameter, DetailBlock, PromptEntry, PromptList,
    SeedList, XYAxis, build_lora_stack_axis, build_prompt_axis, build_seed_axis,
    concatenate_axes, cross_merge_axes,
)


class AxisPreviewTests(unittest.TestCase):
    def test_prompt_preview_preserves_multiline_text_and_artist_boundary(self):
        prompt = PromptEntry("portrait\n细节第二行", "@ordinary_prefix", "suffix", "(@wlop:0.35)")
        axis = build_prompt_axis(PromptList((prompt,)), title="提示词检查")
        text = format_axis_preview(axis)
        self.assertIn("轴内容预览: 提示词检查", text)
        self.assertIn("总项数: 1", text)
        self.assertIn("正文: portrait ⏎ 细节第二行", text)
        self.assertIn("前置: @ordinary_prefix", text)
        self.assertIn("后置: suffix", text)
        self.assertIn("独立画师: (@wlop:0.35)", text)
        self.assertEqual(len(text.splitlines()), 3)
        self.assertTrue(text.splitlines()[2].startswith("\t└─ [01]"))
        self.assertNotIn("PromptEntry(", text)

    def test_style_preview_expands_base_files_triggers_artist_weights_and_template(self):
        template = ArtistTagTemplate("@{tag}", "(@{tag}:{weight})")
        stack = LoraStack((
            LoraStackItem("styles/A.safetensors", "@ordinary_trigger\nsecond", 1.2),
            LoraStackItem(ARTIST_TAG_MODE, "@wlop, @ask_(askzy)", 0.3),
        ), template)
        axis = build_lora_stack_axis(LoraStackList((stack,)), include_base=True)
        text = format_axis_preview(axis)
        self.assertIn("分组数: 2", text)
        self.assertIn("BASE / 沿用基础设置", text)
        self.assertIn("文件: styles/A.safetensors", text)
        self.assertIn("触发词: @ordinary_trigger ⏎ second", text)
        self.assertIn("权重: 1.2", text)
        self.assertIn("权重: 0.3", text)
        self.assertIn("画师: @wlop, @ask_(askzy)", text)
        self.assertIn("weighted_template: (@{tag}:{weight})", text)
        self.assertIn("STYLE SOURCES", text)
        self.assertNotIn("LoraStack(items=", text)
        element = next(line for line in text.splitlines() if "lora_stack:" in line)
        for expected in ("styles/A.safetensors", "@ordinary_trigger", "@wlop", "@ask_(askzy)", "权重: 1.2", "权重: 0.3", "画师模板:"):
            self.assertIn(expected, element)
        self.assertIs(axis.entries[1].parameters[0].value, stack)

    def test_seed_preview_keeps_exact_unsigned_64_bit_values(self):
        axis = build_seed_axis(SeedList((0, 42, MAX_SEED)))
        text = format_axis_preview(axis, "en")
        self.assertIn("Axis Content Preview", text)
        self.assertIn("Entries: 3", text)
        self.assertIn(f"seed: {MAX_SEED}", text)
        self.assertIn(f"SEED: {MAX_SEED}", text)

    def test_combined_axes_keep_groups_order_and_all_parameters(self):
        prompt = build_prompt_axis(PromptList((PromptEntry("one"), PromptEntry("two"))))
        seeds = build_seed_axis(SeedList((10, 20)))
        mixed = cross_merge_axes(prompt, seeds, title="MIXED")
        text = format_axis_preview(mixed, "en")
        self.assertIn("Entries: 4", text)
        self.assertIn("Parameters: prompt, seed", text)
        self.assertLess(text.index("[01]"), text.index("[04]"))
        grouped = concatenate_axes(seeds, build_seed_axis(SeedList((30,))))
        self.assertIn("Group 02 (1 Entry)", format_axis_preview(grouped, "en"))

    def test_generic_axis_formats_all_sampler_parameters_and_nested_custom_values(self):
        @dataclass(frozen=True)
        class CustomValue:
            name: str
            tags: tuple[str, ...]

        parameters = (
            AxisParameter("steps", 24), AxisParameter("cfg", 5.5),
            AxisParameter("sampler_name", "euler"), AxisParameter("scheduler", "normal"),
            AxisParameter("denoise", 0.65), AxisParameter("custom", {"enabled": True, "values": [0, 2], "entry": CustomValue("中文", ("a", "b")), "none": None}),
        )
        axis = XYAxis("CUSTOM", ((AxisEntry("sample", parameters),),))
        text = format_axis_preview(axis, "en")
        for expected in ("steps: 24", "cfg: 5.5", "sampler_name: euler", "scheduler: normal", "denoise: 0.65", "enabled: True", "entry: CustomValue", "name: 中文", "none: None", "tags: [a, b]"):
            self.assertIn(expected, text)
        self.assertEqual(len(text.splitlines()), 3)

    def test_detail_blocks_preserve_multiline_cells_headers_and_text(self):
        axis = XYAxis("DETAILS", ((AxisEntry("BASE", ()),),), (
            DetailBlock("TABLE", "table", ("CODE", "VALUE"), (("A", "first\nsecond"),)),
            DetailBlock("NOTES", "text", text=("line 1\nline 2", "中文")),
            DetailBlock("EMPTY TABLE", "table", ("ONLY HEADER",), ()),
        ))
        text = format_axis_preview(axis, "en")
        self.assertIn("VALUE: first ⏎ second", text)
        self.assertIn("\t\t├─ line 1 ⏎ line 2", text)
        self.assertIn("ONLY HEADER", text)

    def test_dual_style_preview_preserves_child_order_and_original_weights(self):
        original = LoraStack.from_values((("A", "alpha", 1.2), ("B", "beta", 1.0)))
        axis = build_lora_stack_axis(flatten_lora_stack(original, True, "dual"), include_base=False)
        text = format_axis_preview(axis, "en")
        self.assertIn("Entries: 4", text)
        self.assertEqual(text.count("Weight: 1.2"), 2)
        self.assertLess(text.index("[02]"), text.index("[03]"))

    def test_single_line_elements_escape_tabs_and_newlines_without_losing_order(self):
        axis = XYAxis("title\nline", ((AxisEntry("label\tline", (
            AxisParameter("custom", {"note": "first\r\nsecond\tlast"}),
        ), "detail\nnext"),),))
        text = format_axis_preview(axis, "en")
        self.assertEqual(len(text.splitlines()), 3)
        self.assertIn("title ⏎ line", text)
        self.assertIn("label ⇥ line", text)
        self.assertIn("first ⏎ second ⇥ last", text)
        self.assertIn("Note: detail ⏎ next", text)

    def test_recursive_custom_data_is_printable_and_empty_parameters_remain_base(self):
        recursive = []
        recursive.append(recursive)
        axis = XYAxis("custom", ((AxisEntry("recursive", (AxisParameter("data", recursive),)), AxisEntry("BASE", ())),))
        text = format_axis_preview(axis, "en")
        self.assertIn("data: [<cycle>]", text)
        self.assertIn("BASE / sampler defaults", text)

    def test_node_contract_returns_text_ui_and_identical_axis_without_extra_inputs(self):
        self.assertIs(NODE_CLASS_MAPPINGS["LoraTesterAxisPreview"], AxisPreviewNode)
        self.assertTrue(AxisPreviewNode.OUTPUT_NODE)
        self.assertEqual(AxisPreviewNode.INPUT_TYPES()["required"]["axis"][0], "XY_AXIS")
        self.assertEqual(AxisPreviewNode.RETURN_TYPES, ("XY_AXIS", "STRING"))
        axis = build_seed_axis(SeedList((99,)))
        outputs = AxisPreviewNode.preview_axis(axis)
        self.assertIs(outputs["result"][0], axis)
        self.assertEqual(outputs["ui"]["text"], [outputs["result"][1]])
        self.assertEqual(outputs["result"][1], format_axis_preview(axis))
        self.assertEqual(format_axis_preview(axis), format_axis_preview(axis))

    def test_bad_axis_and_language_fail_with_clear_errors(self):
        with self.assertRaisesRegex(TypeError, "XYAxis"):
            AxisPreviewNode.preview_axis("flattened text")
        with self.assertRaisesRegex(ValueError, "language"):
            format_axis_preview(build_seed_axis(SeedList((0,))), "invalid")


if __name__ == "__main__":
    unittest.main()
