from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.axis_preview import format_axis_preview
from lora_tester.nodes import (
    AxisComposerNode, GlobalPromptAppendNode, NODE_CLASS_MAPPINGS,
    PromptAxisNode, PromptListNameNode, _set_prompt_parameter,
)
from lora_tester.xy import (
    PromptEntry, PromptList, SeedList, build_prompt_axis, build_seed_axis,
    cross_merge_axes,
)


class PromptNameTests(unittest.TestCase):
    def make_prompts(self) -> PromptList:
        return PromptList(tuple(
            PromptEntry(
                f"prompt {index}", "prefix", "suffix", "@first, @second", "old name"
            )
            for index in range(4)
        ))

    def test_line_positions_preserve_blank_lines_and_restore_missing_names(self) -> None:
        original = self.make_prompts()
        result = PromptListNameNode.set_names(original, " First \n \t \n第三项")[0]
        self.assertEqual(
            [entry.custom_name for entry in result.entries],
            ["First", None, "第三项", None],
        )
        self.assertEqual(
            [entry.label for entry in build_prompt_axis(result).entries],
            ["First", "P02", "第三项", "P04"],
        )
        for source, renamed in zip(original.entries, result.entries):
            self.assertEqual(source.prompt, renamed.prompt)
            self.assertEqual(source.prefix, renamed.prefix)
            self.assertEqual(source.suffix, renamed.suffix)
            self.assertEqual(source.independent_artist_tags, renamed.independent_artist_tags)
            self.assertEqual(source.full_prompt, renamed.full_prompt)
            self.assertEqual(source.custom_name, "old name")

    def test_newline_formats_and_trailing_empty_lines(self) -> None:
        for names in ("first\n\nthird\n", "first\r\n\r\nthird\r\n", "first\r\rthird\r"):
            with self.subTest(names=names):
                result = self.make_prompts().with_names(names)
                self.assertEqual(
                    [entry.custom_name for entry in result.entries],
                    ["first", None, "third", None],
                )

    def test_extra_lines_are_ignored_and_names_are_literal(self) -> None:
        result = self.make_prompts().with_names("duplicate\nduplicate\n{i}\n{a|b}\nignored")
        self.assertEqual(
            [entry.custom_name for entry in result.entries],
            ["duplicate", "duplicate", "{i}", "{a|b}"],
        )
        self.assertEqual(len(result.entries), 4)

    def test_empty_names_reset_all_existing_names(self) -> None:
        for names in ("", " \t "):
            result = self.make_prompts().with_names(names)
            self.assertTrue(all(entry.custom_name is None for entry in result.entries))
            self.assertEqual(
                [entry.label for entry in build_prompt_axis(result).entries],
                ["P01", "P02", "P03", "P04"],
            )

    def test_global_append_preserves_names_and_artist_boundary(self) -> None:
        named = self.make_prompts().with_names("first\n\nthird")
        appended = GlobalPromptAppendNode.append_prompt(named, "@ordinary", "after", "@extra")[0]
        for original, updated in zip(named.entries, appended.entries):
            self.assertEqual(updated.custom_name, original.custom_name)
            self.assertEqual(updated.prompt, original.prompt)
            self.assertIn("@ordinary", updated.suffix)
            self.assertNotIn("@ordinary", updated.independent_artist_tags)
            self.assertIn("@extra", updated.independent_artist_tags)

    def test_both_axis_builders_preview_and_cross_merge_use_names(self) -> None:
        named = self.make_prompts().with_names("portrait\n\nlandscape")
        axis = PromptAxisNode.build_axis(named, "PROMPT")[0]
        generic = AxisComposerNode.compose_axis("PROMPT", False, source=named)[0]
        self.assertEqual(axis, generic)
        self.assertIn("portrait", format_axis_preview(axis))
        self.assertIn("landscape", format_axis_preview(axis))
        combined = cross_merge_axes(axis, build_seed_axis(SeedList((123,))))
        self.assertIn("portrait", combined.entries[0].label)
        self.assertIs(axis.entries[0].parameters[0].value, named.entries[0])

    def test_names_do_not_enter_sampling_configuration(self) -> None:
        original = self.make_prompts().entries[0]
        renamed = self.make_prompts().with_names("display only").entries[0]
        original_values, renamed_values = {}, {}
        _set_prompt_parameter(original_values, original)
        _set_prompt_parameter(renamed_values, renamed)
        self.assertEqual(original_values, renamed_values)
        self.assertNotIn("custom_name", renamed_values)

    def test_node_contract_and_invalid_source(self) -> None:
        self.assertIs(NODE_CLASS_MAPPINGS["LoraTesterPromptListName"], PromptListNameNode)
        contract = PromptListNameNode.INPUT_TYPES()["required"]
        self.assertEqual(contract["prompt_list"], ("LORA_TESTER_PROMPT_LIST",))
        self.assertEqual(PromptListNameNode.RETURN_TYPES, ("LORA_TESTER_PROMPT_LIST",))
        self.assertTrue(contract["names"][1]["multiline"])
        self.assertFalse(contract["names"][1]["dynamicPrompts"])
        with self.assertRaises(TypeError):
            PromptListNameNode.set_names("not a prompt group", "name")
