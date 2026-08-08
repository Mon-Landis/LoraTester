from __future__ import annotations

import unittest
from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.stack import LoraStack, LoraStackItem, LoraStackList, split_lora_stack
from lora_tester.stack_compositor import (
    LoraStackMatrixCompositor,
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


if __name__ == "__main__":
    unittest.main()
