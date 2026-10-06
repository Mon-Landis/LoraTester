from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.artist import ARTIST_TAG_MODE, ArtistTagTemplate
from lora_tester.nodes import (
    ArtistTagReplacerNode, ArtistTagTextParserNode, AxisComposerNode,
    LoraStackFlattenerNode, LoraStackNode, NODE_CLASS_MAPPINGS,
)
from lora_tester.stack import LoraStack, LoraStackItem, parse_artist_stack, replace_stack_artist


ARTIST_EXAMPLES = (
    (
        "@dodok_(gj77230), @ask_(askzy), @mildt, @fumihiko_(fu_mihi_ko), @henriiku_(ahemaru), @asanagi, @wlop, @nekoya_(nekodayo_22)",
        (("dodok_(gj77230)", 1.0), ("ask_(askzy)", 1.0), ("mildt", 1.0), ("fumihiko_(fu_mihi_ko)", 1.0), ("henriiku_(ahemaru)", 1.0), ("asanagi", 1.0), ("wlop", 1.0), ("nekoya_(nekodayo_22)", 1.0)),
    ),
    (
        "@vxdrq, @nekoya_(nekodayo_22), @aos, (@aoi_sakura_(seak5545):0.55), (@love_cacao:0.35), (@tab_head:0.05), @hermityy, (@laserflip:0.85), (@s16xue:1.5), (@tatsuya_(sopillars):0.35)",
        (("vxdrq", 1.0), ("nekoya_(nekodayo_22)", 1.0), ("aos", 1.0), ("aoi_sakura_(seak5545)", 0.55), ("love_cacao", 0.35), ("tab_head", 0.05), ("hermityy", 1.0), ("laserflip", 0.85), ("s16xue", 1.5), ("tatsuya_(sopillars)", 0.35)),
    ),
    (
        "(@laserflip:0.3), @yuyu_(yuyuworks), @nekometaru, @kyokucho",
        (("laserflip", 0.3), ("yuyu_(yuyuworks)", 1.0), ("nekometaru", 1.0), ("kyokucho", 1.0)),
    ),
    ("@gmkj", (("gmkj", 1.0),)),
)


class ArtistStackParserTests(unittest.TestCase):
    def test_requested_examples_preserve_exact_names_weights_and_order(self):
        for example_index, (text, expected) in enumerate(ARTIST_EXAMPLES, start=1):
            with self.subTest(example=example_index):
                outputs = ArtistTagTextParserNode.parse_text(text)
                self.assertIsInstance(outputs, tuple)
                self.assertEqual(len(outputs), 1)
                stack = outputs[0]
                self.assertEqual(stack.artist_entries, expected)
                self.assertEqual(len(stack.items), len(expected))
                self.assertTrue(all(item.name == ARTIST_TAG_MODE for item in stack.items))
                self.assertEqual(stack.trigger_words, ())
                axis = AxisComposerNode.compose_axis("ARTIST", False, source=stack)[0]
                self.assertEqual(len(axis.entries), 1)
                singles = LoraStackFlattenerNode.flatten_stack(stack)[0]
                self.assertEqual(len(singles.stacks), len(expected))
                self.assertEqual(tuple(entry.artist_entries[0] for entry in singles.stacks), expected)

    def test_parser_keeps_duplicates_template_zero_and_negative_weights(self):
        template = ArtistTagTemplate("@{tag}", "(@{tag}:{weight})")
        stack = ArtistTagTextParserNode.parse_text(" @wlop，(@wlop:0)\n(@ask_(askzy):-0.25), (@gmkj:1e-2), ", template)[0]
        self.assertEqual(stack.artist_entries, (("wlop", 1.0), ("wlop", 0.0), ("ask_(askzy)", -0.25), ("gmkj", 0.01)))
        self.assertIs(stack.artist_template, template)

    def test_parser_rejects_empty_and_malformed_text(self):
        for text in ("", " ,\n，", "@", "(@wlop:nan)", "(@wlop:inf)", "(@wlop:1e999)", "(@wlop:abc)", "(@wlop:)", "(@wlop:0.5", "@ask_(askzy"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_artist_stack(text)

    def test_parser_contract_is_registered_multiline_and_not_dynamic_prompt(self):
        self.assertIs(NODE_CLASS_MAPPINGS["ArtistTagTextParser"], ArtistTagTextParserNode)
        inputs = ArtistTagTextParserNode.INPUT_TYPES()
        self.assertTrue(inputs["required"]["artist_text"][1]["multiline"])
        self.assertFalse(inputs["required"]["artist_text"][1]["dynamicPrompts"])
        self.assertEqual(inputs["optional"]["artist_tag_template"][0], "ARTIST_TAG_TEMPLATE")
        self.assertEqual(ArtistTagTextParserNode.RETURN_TYPES, ("LORA_STACK",))


class ArtistStackReplacementTests(unittest.TestCase):
    def test_replace_and_multiply_each_matching_weight(self):
        stack = parse_artist_stack("(@laserflip:0.3), @gmkj, (@laserflip:0.85)")
        for mode, expected_weights in (("replace", (2.0, 2.0)), ("multiply", (0.6, 1.7))):
            with self.subTest(mode=mode):
                result = ArtistTagReplacerNode.replace_artist(stack, "@laserflip", ARTIST_TAG_MODE, "@wlop", 2.0, mode)[0]
                self.assertEqual(result.artist_entries, (("wlop", expected_weights[0]), ("gmkj", 1.0), ("wlop", expected_weights[1])))
                self.assertIs(result.items[1], stack.items[1])
        self.assertEqual(stack.artist_entries, (("laserflip", 0.3), ("gmkj", 1.0), ("laserflip", 0.85)))

    def test_replacement_can_be_a_lora_with_an_ordinary_at_trigger(self):
        template = ArtistTagTemplate("@{tag}", "(@{tag}:{weight})")
        stack = parse_artist_stack(ARTIST_EXAMPLES[2][0], template)
        result = replace_stack_artist(stack, "laserflip", "Style.safetensors", "@ordinary_trigger", 0.5, "multiply")
        self.assertEqual(result.items[0], LoraStackItem("Style.safetensors", "@ordinary_trigger", 0.15))
        self.assertEqual(result.trigger_words, ("@ordinary_trigger",))
        self.assertEqual(result.artist_entries, ARTIST_EXAMPLES[2][1][1:])
        self.assertIs(result.artist_template, template)

    def test_matching_is_exact_case_sensitive_and_only_artist_mode(self):
        lora = LoraStackItem("A.safetensors", "@laserflip", 0.8)
        stack = LoraStack((lora, *parse_artist_stack("@laserflip_extra, @Laserflip").items))
        result = replace_stack_artist(stack, "laserflip", ARTIST_TAG_MODE, "wlop", 1.2)
        self.assertIs(result, stack)
        self.assertIs(result.items[0], lora)

    def test_grouped_artist_entries_only_replace_the_matching_tag(self):
        template = ArtistTagTemplate("@{tag}", "(@{tag}:{weight})")
        unrelated = LoraStackItem(ARTIST_TAG_MODE, "artist_c, artist_d", 1.5)
        stack = LoraStack((LoraStackItem(ARTIST_TAG_MODE, "@artist_a, @artist_b, @artist_a", 0.4), unrelated), template)
        result = replace_stack_artist(stack, "artist_a", "B.safetensors", "trigger", 2.0, "multiply")
        self.assertEqual(result.items[:3], (LoraStackItem("B.safetensors", "trigger", 0.8), LoraStackItem(ARTIST_TAG_MODE, "artist_b", 0.4), LoraStackItem("B.safetensors", "trigger", 0.8)))
        self.assertIs(result.items[3], unrelated)
        self.assertIs(result.artist_template, template)

    def test_parenthetical_and_weighted_match_text_normalizes_exactly(self):
        stack = parse_artist_stack("(@aoi_sakura_(seak5545):0.55), @gmkj")
        for match_tag in ("aoi_sakura_(seak5545)", "@aoi_sakura_(seak5545)", "(@aoi_sakura_(seak5545):1.5)"):
            with self.subTest(match_tag=match_tag):
                result = replace_stack_artist(stack, match_tag, ARTIST_TAG_MODE, "@wlop", 1.2)
                self.assertEqual(result.artist_entries, (("wlop", 1.2), ("gmkj", 1.0)))

    def test_replacement_supports_multiple_new_artists_in_one_entry(self):
        stack = parse_artist_stack("@gmkj")
        result = replace_stack_artist(stack, "gmkj", ARTIST_TAG_MODE, "@wlop, @ask_(askzy)", 0.7)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.artist_entries, (("wlop", 0.7), ("ask_(askzy)", 0.7)))

    def test_replacement_zero_negative_and_overflow_weights(self):
        stack = parse_artist_stack("(@gmkj:-0.5)")
        self.assertEqual(replace_stack_artist(stack, "gmkj", ARTIST_TAG_MODE, "wlop", 0.0).artist_entries, (("wlop", 0.0),))
        self.assertEqual(replace_stack_artist(stack, "gmkj", ARTIST_TAG_MODE, "wlop", -2.0, "multiply").artist_entries, (("wlop", 1.0),))
        with self.assertRaisesRegex(ValueError, "finite"):
            replace_stack_artist(parse_artist_stack("(@gmkj:1e308)"), "gmkj", ARTIST_TAG_MODE, "wlop", 1e308, "multiply")

    def test_replacement_validation_rejects_invalid_configuration(self):
        stack = parse_artist_stack("@gmkj")
        for match_tag, name, trigger, strength, mode in (
            ("", ARTIST_TAG_MODE, "wlop", 1.0, "replace"),
            ("gmkj, wlop", ARTIST_TAG_MODE, "wlop", 1.0, "replace"),
            ("gmkj", "", "wlop", 1.0, "replace"),
            ("gmkj", ARTIST_TAG_MODE, "", 1.0, "replace"),
            ("gmkj", ARTIST_TAG_MODE, "wlop", float("nan"), "replace"),
            ("gmkj", ARTIST_TAG_MODE, "wlop", 1.0, "unknown"),
        ):
            with self.subTest(match_tag=match_tag, name=name, trigger=trigger, strength=strength, mode=mode):
                with self.assertRaises(ValueError):
                    replace_stack_artist(stack, match_tag, name, trigger, strength, mode)
        with self.assertRaisesRegex(TypeError, "expects a LoraStack"):
            replace_stack_artist(None, "gmkj", ARTIST_TAG_MODE, "wlop", 1.0)

    def test_replacement_contract_reuses_builder_picker_and_numeric_controls(self):
        self.assertIs(NODE_CLASS_MAPPINGS["ArtistTagReplacer"], ArtistTagReplacerNode)
        with patch("lora_tester.nodes._get_lora_names", return_value=["A.safetensors", ARTIST_TAG_MODE]):
            inputs = ArtistTagReplacerNode.INPUT_TYPES()["required"]
            builder = LoraStackNode.INPUT_TYPES()["required"]
        for name in ("lora_1_name", "lora_1_trigger", "lora_1_strength"):
            self.assertEqual(inputs[name], builder[name])
        self.assertEqual(inputs["strength_mode"][0], ["replace", "multiply"])
        self.assertEqual(ArtistTagReplacerNode.RETURN_TYPES, ("LORA_STACK",))

    def test_replacement_file_validation_and_fingerprint_use_existing_path(self):
        with patch("lora_tester.nodes._resolve_lora_path", return_value="D:/A.safetensors") as resolve:
            self.assertIs(ArtistTagReplacerNode.VALIDATE_INPUTS("A.safetensors"), True)
            resolve.assert_called_once_with("A.safetensors")
        with patch("lora_tester.nodes._resolve_lora_path", side_effect=ValueError("missing")):
            self.assertIn("unavailable", ArtistTagReplacerNode.VALIDATE_INPUTS("missing.safetensors"))
            self.assertIs(ArtistTagReplacerNode.VALIDATE_INPUTS(ARTIST_TAG_MODE), True)
        with patch("lora_tester.nodes._lora_input_fingerprint", return_value=(("A", (1, 2, 3)),)) as fingerprint:
            self.assertEqual(ArtistTagReplacerNode.IS_CHANGED(lora_1_name="A.safetensors"), (("A", (1, 2, 3)),))
            fingerprint.assert_called_once_with(("A.safetensors",))


if __name__ == "__main__":
    unittest.main()
