from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.artist import ARTIST_TAG_MODE, ArtistTagTemplate
from lora_tester.nodes import (
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS,
    LoraStackNode, StyleStackExtractNode, StyleStackSetNode,
)
from lora_tester.stack import (
    STACK_INSERT_MODES, LoraStack, LoraStackItem, LoraStackList,
    extract_style_stack, set_style_stack,
)
from lora_tester.xy import build_lora_stack_axis


class StyleStackAccessTests(unittest.TestCase):
    def setUp(self):
        self.template = ArtistTagTemplate("@{tag}", "(@{tag}:{weight})")
        self.stacks = tuple(
            LoraStack((LoraStackItem(ARTIST_TAG_MODE, tag, weight),), self.template, name, mixer)
            for tag, weight, name, mixer in (
                ("@first", 0.8, "First", 0.4),
                ("@second", 1.2, None, None),
                ("@third", 1.0, "Third", 2.5),
            )
        )
        self.source = LoraStackList(self.stacks)
        self.inserted = LoraStack((
            LoraStackItem("style.safetensors", "ordinary @trigger", 0.6),
            LoraStackItem(ARTIST_TAG_MODE, "@replacement", 1.4),
        ), self.template, "Replacement", 1.7)

    def test_extract_indices_and_sentinels_preserve_identity(self):
        for index, position in ((0, 0), (1, 1), (2, 2), (-1, 2), (-2, 0)):
            with self.subTest(index=index):
                self.assertIs(extract_style_stack(self.source, index), self.stacks[position])

    def test_extract_rejects_empty_and_overflow(self):
        for source, indexes in ((self.source, (3, 100)), (LoraStackList(()), (0, -1, -2, 100))):
            for index in indexes:
                with self.subTest(length=len(source.stacks), index=index):
                    with self.assertRaisesRegex(IndexError, "out of range.*length"):
                        extract_style_stack(source, index)

    def test_set_every_mode_at_every_position(self):
        for mode in STACK_INSERT_MODES:
            for index, position in ((0, 0), (1, 1), (2, 2), (-1, 2), (-2, 0)):
                with self.subTest(mode=mode, index=index):
                    expected = list(self.stacks)
                    if mode == "replace":
                        expected[position] = self.inserted
                    else:
                        expected.insert(position + (mode == "after"), self.inserted)
                    output = set_style_stack(self.source, self.inserted, index, mode)
                    self.assertEqual(output.stacks, tuple(expected))
                    self.assertTrue(all(actual is wanted for actual, wanted in zip(output.stacks, expected)))
                    self.assertIsNot(output, self.source)
                    self.assertEqual(self.source.stacks, self.stacks)

    def test_set_overflow_appends_even_for_replace(self):
        for index in (3, 4, 2147483647):
            for mode in STACK_INSERT_MODES:
                with self.subTest(index=index, mode=mode):
                    output = set_style_stack(self.source, self.inserted, index, mode)
                    self.assertEqual(output.stacks, self.stacks + (self.inserted,))

    def test_set_empty_and_singleton_lists(self):
        for mode in STACK_INSERT_MODES:
            for index in (0, -1, -2, 10):
                with self.subTest(mode=mode, index=index):
                    output = set_style_stack(LoraStackList(()), self.inserted, index, mode)
                    self.assertEqual(output.stacks, (self.inserted,))
            singleton = LoraStackList((self.stacks[0],))
            self.assertEqual(set_style_stack(singleton, self.inserted, -1, mode),
                             set_style_stack(singleton, self.inserted, -2, mode))

    def test_invalid_indices_are_not_coerced(self):
        for index in (True, False, 1.0, "1", None):
            with self.subTest(index=index):
                with self.assertRaisesRegex(TypeError, "integer"):
                    extract_style_stack(self.source, index)
                with self.assertRaisesRegex(TypeError, "integer"):
                    set_style_stack(self.source, self.inserted, index)
        for index in (-3, -100):
            with self.assertRaisesRegex(ValueError, "-2"):
                extract_style_stack(self.source, index)
            with self.assertRaisesRegex(ValueError, "-2"):
                set_style_stack(self.source, self.inserted, index)

    def test_invalid_types_and_modes(self):
        for source in (None, self.stacks, self.inserted):
            with self.assertRaises(TypeError):
                extract_style_stack(source, 0)
            with self.assertRaises(TypeError):
                set_style_stack(source, self.inserted, 0)
        for stack in (None, self.source, self.inserted.items):
            with self.assertRaises(TypeError):
                set_style_stack(self.source, stack, 0)
        for mode in ("", "前", "append", None):
            with self.assertRaisesRegex(ValueError, "Insertion mode"):
                set_style_stack(self.source, self.inserted, 0, mode)

    def test_duplicates_are_processed_by_position(self):
        duplicates = LoraStackList((self.inserted, self.inserted))
        inserted = set_style_stack(duplicates, self.inserted, 0, "before")
        self.assertEqual(len(inserted.stacks), 3)
        self.assertTrue(all(stack is self.inserted for stack in inserted.stacks))
        replaced = set_style_stack(duplicates, self.stacks[0], -1)
        self.assertIs(replaced.stacks[0], self.inserted)
        self.assertIs(replaced.stacks[1], self.stacks[0])

    def test_nodes_round_trip_metadata_into_style_axis(self):
        updated, = StyleStackSetNode.set_stack(self.source, self.inserted, -1, "after")
        output, = StyleStackExtractNode.extract_stack(updated, -1)
        self.assertIs(output, self.inserted)
        self.assertIs(output.artist_template, self.template)
        self.assertEqual(output.anima_mixer_strength, 1.7)
        self.assertEqual(output.trigger_words, ("ordinary @trigger",))
        self.assertEqual(output.artist_entries, (("replacement", 1.4),))
        axis = build_lora_stack_axis(LoraStackList((output,)), include_base=False)
        self.assertEqual(axis.entries[0].label, "Replacement")

    def test_registration_and_legacy_socket_contracts(self):
        self.assertIs(NODE_CLASS_MAPPINGS["StyleStackExtract"], StyleStackExtractNode)
        self.assertIs(NODE_CLASS_MAPPINGS["StyleStackSet"], StyleStackSetNode)
        self.assertIs(NODE_CLASS_MAPPINGS["LoraStack"], LoraStackNode)
        self.assertEqual(StyleStackExtractNode.RETURN_TYPES, ("LORA_STACK",))
        self.assertEqual(StyleStackSetNode.RETURN_TYPES, ("LORA_STACK_LIST",))
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["LoraStack"], "StyleStack")
        for node in (StyleStackExtractNode, StyleStackSetNode):
            required = node.INPUT_TYPES()["required"]
            self.assertEqual(required["lora_stack_list"][0], "LORA_STACK_LIST")
            self.assertEqual(required["index"][1]["min"], -2)
        self.assertEqual(StyleStackSetNode.INPUT_TYPES()["required"]["mode"][0], list(STACK_INSERT_MODES))

    def test_localization_documents_boundaries_and_no_old_display_names(self):
        for locale in ("en", "zh"):
            definitions = json.loads((ROOT / "locales" / locale / "nodeDefs.json").read_text(encoding="utf-8"))
            for identifier, node in (("StyleStackExtract", StyleStackExtractNode), ("StyleStackSet", StyleStackSetNode)):
                entry = definitions[identifier]
                self.assertEqual(set(entry["inputs"]), set(node.INPUT_TYPES()["required"]))
                self.assertIn("-1", entry["inputs"]["index"]["tooltip"])
                self.assertIn("-2", entry["inputs"]["index"]["tooltip"])
            for entry in definitions.values():
                self.assertNotIn("LoRA Stack", entry["display_name"])
                self.assertNotIn("Style Stack", entry["display_name"])


if __name__ == "__main__":
    unittest.main()
