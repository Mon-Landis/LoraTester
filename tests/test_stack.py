from __future__ import annotations

import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.artist import ARTIST_TAG_MODE, ArtistTagTemplate
from lora_tester.stack import LoraStack, LoraStackItem, LoraStackList, flatten_lora_stack, split_lora_stack
from lora_tester.stack_compositor import (
    LoraStackMatrixCompositor,
    LoraStackMatrixSession,
    _original_lora_items,
    _prompt_axis_label,
    _stack_mix_labels,
)
from lora_tester.styles import StyleConfig


class StackModelTests(unittest.TestCase):
    def make_stack(self) -> LoraStack:
        return LoraStack.from_values(
            [
                ("A.safetensors", "alpha", 0.5),
                ("B.safetensors", "beta", 1.0),
                ("C.safetensors", "charlie", 1.5),
            ]
        )

    def test_stack_requires_a_file_and_finite_strength(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            LoraStackItem("", "", 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            LoraStackItem("A", "", float("nan"))

    def test_splitter_orders_singles_pairs_and_full_combination(self) -> None:
        result = split_lora_stack(self.make_stack())
        self.assertEqual(
            [stack.label for stack in result.stacks],
            ["A", "B", "C", "A + B", "A + C", "B + C", "A + B + C"],
        )

    def test_flattener_preserves_entry_order_and_configuration(self) -> None:
        stack = self.make_stack()
        result = flatten_lora_stack(stack)
        self.assertIsInstance(result, LoraStackList)
        self.assertEqual([entry.label for entry in result.stacks], ["A", "B", "C"])
        self.assertEqual(tuple(entry.items[0] for entry in result.stacks), stack.items)
        self.assertTrue(all(len(entry.items) == 1 for entry in result.stacks))
        self.assertEqual(len(stack.items), 3)

    def test_flattener_prepends_original_only_when_enabled(self) -> None:
        stack = self.make_stack()
        result = flatten_lora_stack(stack, include_original=True)
        self.assertEqual(len(result.stacks), 4)
        self.assertIs(result.stacks[0], stack)
        self.assertEqual([entry.label for entry in result.stacks], ["A + B + C", "A", "B", "C"])

    def test_flattener_preserves_artist_entries_and_template(self) -> None:
        template = ArtistTagTemplate("@{tag}", "(@{tag}:{weight})")
        stack = LoraStack(
            (
                LoraStackItem("A.safetensors", "@ordinary_trigger", -0.3),
                LoraStackItem(ARTIST_TAG_MODE, "artist_a, artist_b", 1.2),
            ),
            artist_template=template,
        )
        for include_original in (False, True):
            with self.subTest(include_original=include_original):
                result = flatten_lora_stack(stack, include_original)
                self.assertTrue(all(entry.artist_template is template for entry in result.stacks))
                self.assertEqual(result.stacks[-2].trigger_words, ("@ordinary_trigger",))
                self.assertEqual(result.stacks[-2].artist_entries, ())
                self.assertEqual(result.stacks[-1].artist_entries, (("artist_a", 1.2), ("artist_b", 1.2)))

    def test_flattener_preserves_duplicates_and_single_entry_original(self) -> None:
        item = LoraStackItem("A.safetensors", "alpha", 0.0)
        stack = LoraStack((item, item))
        self.assertEqual(len(flatten_lora_stack(stack).stacks), 2)
        single = LoraStack((item,))
        self.assertEqual(flatten_lora_stack(single).stacks, (single,))
        self.assertEqual(flatten_lora_stack(single, True).stacks, (single,))

    def test_flattener_rejects_non_stack_values(self) -> None:
        for value in (None, [], LoraStackList(()), self.make_stack().items):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "expects a LoraStack"):
                    flatten_lora_stack(value)

    def test_flattener_weight_modes_order_children_and_leave_original_unchanged(self) -> None:
        template = ArtistTagTemplate("@{tag}", "(@{tag}:{weight})")
        stack = LoraStack(
            (
                LoraStackItem("a.safetensors", "alpha", 1.2),
                LoraStackItem(ARTIST_TAG_MODE, "artist_b, artist_c", 0.3),
                LoraStackItem("c.safetensors", "@ordinary_trigger", 1.0),
            ),
            artist_template=template,
        )
        expected = {
            "inherit": ((0, 1.2), (1, 0.3), (2, 1.0)),
            "normalize": ((0, 1.0), (1, 1.0), (2, 1.0)),
            "dual": ((0, 1.0), (0, 1.2), (1, 1.0), (1, 0.3), (2, 1.0)),
        }
        for mode, configurations in expected.items():
            for include_original in (False, True):
                with self.subTest(mode=mode, include_original=include_original):
                    result = flatten_lora_stack(stack, include_original, mode)
                    children = result.stacks[1:] if include_original else result.stacks
                    if include_original:
                        self.assertIs(result.stacks[0], stack)
                        self.assertEqual([item.strength for item in result.stacks[0].items], [1.2, 0.3, 1.0])
                    self.assertEqual(len(children), len(configurations))
                    for child, (index, weight) in zip(children, configurations):
                        original = stack.items[index]
                        self.assertEqual(child.items, (LoraStackItem(original.name, original.trigger_word, weight),))
                        self.assertIs(child.artist_template, template)
        self.assertEqual([item.strength for item in stack.items], [1.2, 0.3, 1.0])

    def test_flattener_dual_handles_unit_zero_negative_and_near_unit_weights(self) -> None:
        weights = (1.0, 0.0, -0.5, 1.000000001)
        stack = LoraStack(tuple(LoraStackItem("A.safetensors", strength=weight) for weight in weights))
        result = flatten_lora_stack(stack, weight_mode="dual")
        self.assertEqual([entry.items[0].strength for entry in result.stacks], [1.0, 1.0, 0.0, 1.0, -0.5, 1.0, 1.000000001])

    def test_flattener_unit_weight_dual_emits_once_and_legacy_calls_inherit(self) -> None:
        stack = LoraStack((LoraStackItem("A.safetensors", strength=1.0),))
        self.assertEqual(len(flatten_lora_stack(stack, weight_mode="dual").stacks), 1)
        self.assertEqual(flatten_lora_stack(stack, True, "dual").stacks, (stack,))
        legacy = self.make_stack()
        self.assertEqual(flatten_lora_stack(legacy), flatten_lora_stack(legacy, weight_mode="inherit"))

    def test_flattener_rejects_unknown_weight_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "weight_mode"):
            flatten_lora_stack(self.make_stack(), weight_mode="unknown")

    def test_lister_merges_in_order_and_preserves_explicit_duplicates(self) -> None:
        first = self.make_stack()
        second = LoraStack((LoraStackItem("D.safetensors", strength=0.25),))
        result = LoraStackList.merge((first, second, first))
        self.assertEqual([stack.label for stack in result.stacks], ["A + B + C", "D", "A + B + C"])


class StackCompositorTests(unittest.TestCase):
    def test_matrix_uses_mix_state_headers_and_a_narrow_prompt_axis(self) -> None:
        a = LoraStack((LoraStackItem("Lora_A.safetensors"),))
        b = LoraStack((LoraStackItem("Lora_B.safetensors"),))
        combined = LoraStack(a.items + b.items)
        self.assertEqual(_stack_mix_labels((a, b, combined)), ("A", "B", "A+B"))
        self.assertEqual((_prompt_axis_label(0), _prompt_axis_label(1)), ("Prompt 1", "Prompt 2"))
        self.assertEqual([item.display_name for item in _original_lora_items((a, b, combined))], ["Lora_A", "Lora_B"])
        compositor = LoraStackMatrixCompositor(
            (a, b),
            ("a long prompt label",),
            16,
            10,
            style=StyleConfig.black(decorator="none"),
            max_canvas_pixels=None,
        )
        self.assertLessEqual(compositor.geometry.cell(0, 0)[0], 100)

    def test_same_file_with_different_configuration_gets_distinct_output_identity(self) -> None:
        low = LoraStack((LoraStackItem("Shared.safetensors", "style_low", 0.5),))
        high = LoraStack((LoraStackItem("Shared.safetensors", "style_high", 1.0),))
        combined = LoraStack(low.items + high.items)

        self.assertEqual(_stack_mix_labels((low, high, combined)), ("A", "B", "A+B"))
        original_items = _original_lora_items((low, high, combined))
        self.assertEqual(len(original_items), 2)
        self.assertEqual(
            [(item.display_name, item.trigger_word, item.strength) for item in original_items],
            [
                ("Shared", "style_low", 0.5),
                ("Shared", "style_high", 1.0),
            ],
        )

        compositor = LoraStackMatrixCompositor(
            (low, high, combined),
            ("portrait",),
            32,
            24,
            style=StyleConfig.black(decorator="none"),
            max_canvas_pixels=None,
        )
        self.assertEqual(compositor.column_count, 4)
        self.assertEqual(compositor.original_lora_items, original_items)

    def test_xy_matrix_keeps_base_column_and_control_gap(self) -> None:
        a = LoraStack((LoraStackItem("A.safetensors"),))
        b = LoraStack((LoraStackItem("B.safetensors"),))
        compositor = LoraStackMatrixCompositor(
            (a, b),
            ("portrait", "landscape"),
            16,
            10,
            style=StyleConfig.black(decorator="none", show_axis_labels=True),
            max_canvas_pixels=None,
        )
        images = [
            Image.new("RGB", (16, 10), color)
            for color in ("red", "green", "blue", "yellow", "purple", "cyan")
        ]
        output = compositor.compose(images)
        self.assertEqual(output.size, compositor.geometry.canvas_size)
        base = compositor.geometry.cell(0, 0)
        self.assertEqual(
            output.getpixel(((base[0] + base[2]) // 2, (base[1] + base[3]) // 2)),
            (255, 0, 0),
        )
        first_stack = compositor.geometry.cell(0, 1)
        self.assertGreater(first_stack[0] - base[2], compositor.style.cell_gap)
        self.assertEqual(
            output.getpixel(((first_stack[0] + first_stack[2]) // 2, (first_stack[1] + first_stack[3]) // 2)),
            (0, 128, 0),
        )
        second_row = compositor.geometry.cell(1, 2)
        self.assertEqual(
            output.getpixel(((second_row[0] + second_row[2]) // 2, (second_row[1] + second_row[3]) // 2)),
            (0, 255, 255),
        )

    def test_default_control_gap_scales_with_image_width(self) -> None:
        stack = LoraStack((LoraStackItem("A.safetensors"),))
        for width, expected_gap in ((320, 40), (640, 80)):
            with self.subTest(width=width):
                compositor = LoraStackMatrixCompositor(
                    (stack,),
                    ("portrait",),
                    width,
                    100,
                    style=StyleConfig.black(decorator="none"),
                    max_canvas_pixels=None,
                )
                base = compositor.geometry.cell(0, 0)
                effect = compositor.geometry.cell(0, 1)
                self.assertEqual(compositor.geometry.control_gap, expected_gap)
                self.assertEqual(effect[0] - base[2], expected_gap)

        overridden = LoraStackMatrixCompositor(
            (stack,),
            ("portrait",),
            640,
            100,
            style=StyleConfig.black(decorator="none"),
            control_gap=48,
            max_canvas_pixels=None,
        )
        self.assertEqual(overridden.geometry.control_gap, 48)

    def test_technical_style_can_render_with_stack_geometry(self) -> None:
        stack = LoraStack((LoraStackItem("A.safetensors"),))
        compositor = LoraStackMatrixCompositor(
            (stack,),
            ("portrait",),
            32,
            24,
            style=StyleConfig.black(decorator="technical"),
            max_canvas_pixels=None,
        )
        output = compositor.compose([Image.new("RGB", (32, 24), "red")] * 2)
        self.assertEqual(output.size, compositor.geometry.canvas_size)

    def test_artist_mixer_annotation_is_drawn_only_for_marked_cell(self) -> None:
        stacks = (
            LoraStack((LoraStackItem("A.safetensors"),)),
            LoraStack((LoraStackItem("B.safetensors"),)),
        )
        compositor = LoraStackMatrixCompositor(
            stacks,
            ("portrait", "landscape"),
            64,
            48,
            style=StyleConfig.black(decorator="none"),
            show_stack_details=False,
            reserve_artist_mixer_labels=True,
            max_canvas_pixels=None,
        )
        calls: list[str] = []
        original = LoraStackMatrixSession._draw_fitted_text

        def capture(session, text, rect, font_size, color, **kwargs):
            calls.append(str(text))
            return original(session, text, rect, font_size, color, **kwargs)

        with patch.object(LoraStackMatrixSession, "_draw_fitted_text", new=capture):
            session = compositor.start()
            session.submit(Image.new("RGB", (64, 48), "red"), coordinate=(0, 0))
            self.assertNotIn("Anima Artist Mixer", calls)
            session.submit(Image.new("RGB", (64, 48), "blue"), coordinate=(1, 0), artist_mixer=True)
            output = session.finalize(strict=False)

        self.assertIn("Anima Artist Mixer", calls)
        label = compositor.geometry.artist_mixer_label(1, 0)
        self.assertIsNotNone(label)
        self.assertEqual(
            compositor.geometry.cell(1, 0)[1] - compositor.geometry.cell(0, 0)[3],
            compositor.geometry.artist_mixer_label_height + compositor.style.cell_gap,
        )
        self.assertTrue(
            np.any(np.all(np.asarray(output.crop(label)) == compositor.style.text_color, axis=2))
        )


if __name__ == "__main__":
    unittest.main()
