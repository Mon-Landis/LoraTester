from __future__ import annotations

import json
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.artist import ARTIST_TAG_MODE, ArtistTagTemplate
from lora_tester.axis_preview import format_axis_preview
from lora_tester.nodes import (
    ArtistTagReplacerNode, ArtistTagTextParserNode, AxisComposerNode,
    LoraStackAxisNode, LoraStackListNameNode, LoraStackListerNode,
    LoraStackNameNode, LoraStackNode, NODE_CLASS_MAPPINGS, _xy_stack_key,
)
from lora_tester.stack import (
    LoraStack, LoraStackItem, LoraStackList, flatten_lora_stack,
    parse_artist_stack, rename_lora_stack, rename_lora_stack_list,
    replace_stack_artist, split_lora_stack,
)
from lora_tester.xy import (
    PromptEntry, PromptList, build_lora_stack_axis, build_prompt_axis,
    concatenate_axes, cross_merge_axes,
)


class StackNameTests(unittest.TestCase):
    def setUp(self):
        self.template = ArtistTagTemplate("@{tag}", "(@{tag}:{weight})")
        self.stack = LoraStack((
            LoraStackItem("a.safetensors", "alpha", 0.8),
            LoraStackItem(ARTIST_TAG_MODE, "@wlop", 0.3),
        ), self.template, "原组合")

    def test_name_is_optional_normalized_and_immutable(self):
        self.assertIsNone(LoraStack(self.stack.items).custom_name)
        for name in (None, "", "   "):
            with self.subTest(name=name):
                self.assertIsNone(replace(self.stack, custom_name=name).custom_name)
        self.assertEqual(replace(self.stack, custom_name="  柔和厚涂  ").custom_name, "柔和厚涂")
        with self.assertRaises(FrozenInstanceError):
            self.stack.custom_name = "changed"
        for name in ("line\nbreak", "line\rbreak", "tab\tname"):
            with self.assertRaisesRegex(ValueError, "single line"):
                rename_lora_stack(self.stack, name)
        with self.assertRaisesRegex(TypeError, "text"):
            rename_lora_stack(self.stack, 123)

    def test_single_rename_changes_only_metadata_and_reset_is_deferred(self):
        renamed = rename_lora_stack(self.stack, "测试{i}")
        self.assertEqual(renamed.custom_name, "测试{i}")
        self.assertIs(renamed.items, self.stack.items)
        self.assertIs(renamed.artist_template, self.template)
        self.assertEqual(renamed.label, self.stack.label)
        self.assertEqual(renamed.signature(), self.stack.signature())
        self.assertEqual(_xy_stack_key(renamed), _xy_stack_key(self.stack))
        reset = rename_lora_stack(renamed, "")
        self.assertIsNone(reset.custom_name)
        self.assertEqual(self.stack.custom_name, "原组合")
        self.assertIs(rename_lora_stack(self.stack, "原组合"), self.stack)

    def test_builders_accept_names_and_legacy_calls_keep_defaults(self):
        values = dict(lora_1_name=ARTIST_TAG_MODE, lora_1_trigger="@wlop", lora_1_strength=0.8)
        builder = LoraStackNode()
        named = builder.build_stack(1, self.template, custom_name="配置组合", **values)[0]
        self.assertEqual(named.custom_name, "配置组合")
        self.assertIs(named.artist_template, self.template)
        self.assertIsNone(builder.build_stack(1, **values)[0].custom_name)
        parsed = ArtistTagTextParserNode.parse_text("(@wlop:0.8)", self.template, "解析组合")[0]
        self.assertEqual(parsed.custom_name, "解析组合")
        self.assertEqual(parsed.artist_entries, (("wlop", 0.8),))
        self.assertIsNone(ArtistTagTextParserNode.parse_text("@wlop")[0].custom_name)

    def test_new_builder_inputs_are_optional_and_appended(self):
        with patch("lora_tester.nodes._get_lora_names", return_value=[ARTIST_TAG_MODE]):
            builder = LoraStackNode.INPUT_TYPES()
        self.assertNotIn("custom_name", builder["required"])
        self.assertEqual(list(builder["optional"])[-1], "custom_name")
        self.assertEqual(list(ArtistTagTextParserNode.INPUT_TYPES()["optional"])[-1], "custom_name")
        with patch("lora_tester.nodes._get_lora_names", return_value=[ARTIST_TAG_MODE]):
            rename_input = ArtistTagReplacerNode.INPUT_TYPES()["optional"]["custom_name"]
        self.assertEqual(rename_input[0], "STRING")
        self.assertTrue(rename_input[1]["advanced"])
        self.assertFalse(rename_input[1]["dynamicPrompts"])
        self.assertEqual(rename_input[1]["default"], "")

    def test_list_selects_zero_based_position_without_mutating_shared_stack(self):
        source = LoraStackList((self.stack, self.stack, self.stack))
        output = rename_lora_stack_list(source, 1, "方案-{i}")
        self.assertEqual([stack.custom_name for stack in output.stacks], ["原组合", "方案-1", "原组合"])
        self.assertIs(output.stacks[0], self.stack)
        self.assertIs(output.stacks[2], self.stack)
        self.assertEqual([stack.custom_name for stack in source.stacks], ["原组合"] * 3)

    def test_constructor_and_replacer_cache_fingerprints_include_custom_names(self):
        with patch("lora_tester.nodes._lora_input_fingerprint", return_value=(("file", "state"),)):
            for node in (LoraStackNode, ArtistTagReplacerNode):
                with self.subTest(node=node.__name__):
                    default = node.IS_CHANGED(custom_name="")
                    first = node.IS_CHANGED(custom_name="First")
                    second = node.IS_CHANGED(custom_name="Second")
                    self.assertNotEqual(default, first)
                    self.assertNotEqual(first, second)
                    self.assertEqual(node.IS_CHANGED(custom_name="   "), default)
                    self.assertEqual(default, (("file", "state"),))

    def test_all_negative_indices_name_all_and_use_input_indices(self):
        source = LoraStackList((self.stack,) * 3)
        for index in (-1, -2, -123):
            with self.subTest(index=index):
                output = rename_lora_stack_list(source, index, "测试-{i}-{i}")
                self.assertEqual([stack.custom_name for stack in output.stacks], ["测试-0-0", "测试-1-1", "测试-2-2"])

    def test_out_of_range_and_empty_lists_are_noops(self):
        source = LoraStackList((self.stack,))
        for index in (1, 100):
            self.assertIs(rename_lora_stack_list(source, index, "ignored"), source)
        empty = LoraStackList(())
        self.assertIs(rename_lora_stack_list(empty, -1, "方案-{i}"), empty)

    def test_list_blank_names_clear_only_selected_positions(self):
        source = LoraStackList((self.stack,) * 3)
        single = rename_lora_stack_list(source, 0, "  ")
        self.assertEqual([stack.custom_name for stack in single.stacks], [None, "原组合", "原组合"])
        self.assertTrue(all(stack.custom_name is None for stack in rename_lora_stack_list(source, -1, "").stacks))

    def test_list_name_escaping_is_single_pass_and_backslashes_are_preserved(self):
        source = LoraStackList((self.stack,) * 3)
        cases = (
            (r"风格-{i}-\{i}", "风格-2-{i}"),
            (r"风格-\\{i}", "风格-\\2"),
            (r"风格-\\\{i}", "风格-\\{i}"),
            (r"\{i\}", "{i}"),
            (r"\{{i}\}", "{2}"),
            (r"{name}-{i:02d}-{i}", "{name}-{i:02d}-2"),
            ("C:\\styles\\draft-{i}\\", "C:\\styles\\draft-2\\"),
        )
        for template, expected in cases:
            with self.subTest(template=template):
                self.assertEqual(rename_lora_stack_list(source, 2, template).stacks[2].custom_name, expected)

    def test_list_rename_rejects_invalid_indices_and_wrong_types(self):
        source = LoraStackList((self.stack,))
        for index in (1.5, True, "0"):
            with self.assertRaisesRegex(TypeError, "integer"):
                rename_lora_stack_list(source, index, "test")
        with self.assertRaisesRegex(TypeError, "LoraStackList"):
            rename_lora_stack_list(self.stack, -1, "test")
        with self.assertRaisesRegex(TypeError, "LoraStack"):
            rename_lora_stack(source, "test")

    def test_merge_preserves_names_order_and_duplicates(self):
        second = rename_lora_stack(self.stack, "第二项")
        source = LoraStackList.merge((self.stack, LoraStackList((second, self.stack))))
        self.assertEqual(source.stacks, (self.stack, second, self.stack))
        self.assertEqual(LoraStackListerNode.list_stacks(self.stack, stack_2=second)[0].stacks, (self.stack, second))

    def test_complete_combination_retains_name_and_only_subsets_reset(self):
        output = split_lora_stack(self.stack)
        self.assertEqual([stack.custom_name for stack in output.stacks], [None, None, "原组合"])
        self.assertIs(output.stacks[-1], self.stack)
        self.assertTrue(all(stack.artist_template is self.template for stack in output.stacks))
        single = replace(self.stack, items=(self.stack.items[0],))
        self.assertIs(split_lora_stack(single).stacks[0], single)

    def test_multi_entry_flatten_original_keeps_name_and_children_reset(self):
        for mode in ("inherit", "normalize", "dual"):
            for include_original in (False, True):
                with self.subTest(mode=mode, include_original=include_original):
                    output = flatten_lora_stack(self.stack, include_original, mode)
                    children = output.stacks[1:] if include_original else output.stacks
                    self.assertTrue(all(stack.custom_name is None for stack in children))
                    if include_original:
                        self.assertIs(output.stacks[0], self.stack)

    def test_single_entry_flatten_matches_named_examples(self):
        single = replace(self.stack, items=(self.stack.items[0],), custom_name="A")
        cases = {
            "inherit": [(0.8, "A")],
            "normalize": [(1.0, "A")],
            "dual": [(0.8, "A"), (1.0, None)],
        }
        for mode, expected in cases.items():
            with self.subTest(mode=mode):
                result = flatten_lora_stack(single, weight_mode=mode)
                self.assertEqual([(stack.items[0].strength, stack.custom_name) for stack in result.stacks], expected)
                self.assertTrue(all(stack.artist_template is self.template for stack in result.stacks))

    def test_single_original_merges_identical_children_without_changing_weights(self):
        for weight in (0.8, 0.0, -0.5, 1.0, 1.000000001):
            single = replace(self.stack, items=(replace(self.stack.items[0], strength=weight),))
            inherited = flatten_lora_stack(single, True, "inherit")
            self.assertEqual(inherited.stacks, (single,))
            dual = flatten_lora_stack(single, True, "dual")
            self.assertIs(dual.stacks[0], single)
            self.assertEqual(len(dual.stacks), 1 if weight == 1 else 2)
            normalized = flatten_lora_stack(single, True, "normalize")
            self.assertIs(normalized.stacks[0], single)
            self.assertEqual(len(normalized.stacks), 1 if weight == 1 else 2)
            self.assertEqual(normalized.stacks[-1].items[0].strength, 1.0)
            self.assertEqual(normalized.stacks[-1].custom_name, "原组合")

    def test_replacer_preserves_name_and_nonempty_override_is_post_processing(self):
        changed = replace_stack_artist(self.stack, "wlop", ARTIST_TAG_MODE, "@ask_(askzy)", 0.5)
        self.assertEqual(changed.custom_name, "原组合")
        output = ArtistTagReplacerNode.replace_artist(self.stack, "wlop", ARTIST_TAG_MODE, "@ask_(askzy)", 0.5)[0]
        self.assertEqual(output, changed)
        overridden = ArtistTagReplacerNode.replace_artist(
            self.stack, "wlop", ARTIST_TAG_MODE, "@ask_(askzy)", 0.5, custom_name="新风格{i}",
        )[0]
        self.assertEqual(overridden, rename_lora_stack(changed, "新风格{i}"))
        missed = ArtistTagReplacerNode.replace_artist(
            self.stack, "unknown", ARTIST_TAG_MODE, "@ask_(askzy)", 0.5, custom_name="未匹配也命名",
        )[0]
        self.assertEqual(missed.custom_name, "未匹配也命名")
        self.assertIs(missed.items, self.stack.items)
        unchanged = ArtistTagReplacerNode.replace_artist(
            self.stack, "unknown", ARTIST_TAG_MODE, "@ask_(askzy)", 0.5, custom_name="   ",
        )[0]
        self.assertIs(unchanged, self.stack)

    def test_axis_names_override_only_labels_and_preserve_source_tokens(self):
        reset = rename_lora_stack(self.stack, "")
        output = build_lora_stack_axis(LoraStackList((self.stack, reset, rename_lora_stack(self.stack, "BASE"))))
        self.assertEqual([entry.label for entry in output.entries], ["BASE", "原组合", "A-0.8+B-0.3", "BASE"])
        self.assertEqual(output.entries[0].parameters, ())
        self.assertIs(output.entries[3].parameter_map["lora_stack"].items, self.stack.items)
        self.assertEqual(output.entries[1].detail_label, self.stack.label)
        automatic = build_lora_stack_axis(LoraStackList((reset, reset, reset)))
        self.assertEqual(output.detail_blocks, automatic.detail_blocks)
        preview = format_axis_preview(output)
        self.assertIn("原组合 | lora_stack:", preview)
        self.assertIn("alpha", preview)
        self.assertIn("@wlop", preview)

    def test_automatic_labels_are_resolved_per_axis_not_stored_on_stack(self):
        reset = rename_lora_stack(self.stack, "")
        other = LoraStack((LoraStackItem("other.safetensors"),))
        first = build_lora_stack_axis(LoraStackList((reset,)), include_base=False)
        second = build_lora_stack_axis(LoraStackList((other, reset)), include_base=False)
        self.assertEqual(first.entries[0].label, "A-0.8+B-0.3")
        self.assertEqual(second.entries[1].label, "B-0.8+C-0.3")
        self.assertIsNone(reset.custom_name)

    def test_axis_nodes_and_composition_inherit_custom_names(self):
        source = LoraStackList((self.stack,))
        axis = LoraStackAxisNode.build_axis(source, False, "STYLE")[0]
        self.assertEqual(axis.entries[0].label, "原组合")
        self.assertEqual(AxisComposerNode.compose_axis("STYLE", False, source)[0], axis)
        prompt = build_prompt_axis(PromptList((PromptEntry("portrait"),)))
        merged = cross_merge_axes(axis, prompt)
        self.assertEqual(merged.entries[0].label, "原组合 × P01")
        self.assertEqual(concatenate_axes(axis, axis).entries, (axis.entries[0], axis.entries[0]))

    def test_naming_node_registration_and_outputs(self):
        self.assertIs(NODE_CLASS_MAPPINGS["LoraStackName"], LoraStackNameNode)
        self.assertIs(NODE_CLASS_MAPPINGS["LoraStackListName"], LoraStackListNameNode)
        self.assertEqual(LoraStackNameNode.RETURN_TYPES, ("LORA_STACK",))
        self.assertEqual(LoraStackListNameNode.RETURN_TYPES, ("LORA_STACK_LIST",))
        self.assertEqual(LoraStackNameNode.set_name(self.stack, "命名")[0].custom_name, "命名")
        self.assertEqual(LoraStackListNameNode.set_names(LoraStackList((self.stack,)), -1, "名-{i}")[0].stacks[0].custom_name, "名-0")

    def test_locale_docs_cover_new_nodes_and_placeholder_escaping(self):
        for language in ("en", "zh"):
            definitions = json.loads((ROOT / "locales" / language / "nodeDefs.json").read_text(encoding="utf-8"))
            for node_name in ("LoraStackName", "LoraStackListName"):
                node = NODE_CLASS_MAPPINGS[node_name]
                definition = definitions[node_name]
                self.assertTrue(definition["display_name"])
                self.assertTrue(definition["description"])
                self.assertTrue(set(node.INPUT_TYPES()["required"]).issubset(definition["inputs"]))
                self.assertIn("0", definition["outputs"])
            self.assertIn(r"\{i}", definitions["LoraStackListName"]["description"])
            for node_name in ("LoraStack", "ArtistTagTextParser", "ArtistTagReplacer"):
                self.assertIn("custom_name", definitions[node_name]["inputs"])


if __name__ == "__main__":
    unittest.main()
