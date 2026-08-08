from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.layout import LoraSpec, build_layout


class LayoutPlanTests(unittest.TestCase):
    def test_mode_counts(self) -> None:
        expected = {
            1: (5, 5, 0, 0),
            2: (25, 25, 0, 0),
            3: (69, 73, 8, 4),
        }
        for count, values in expected.items():
            with self.subTest(count=count):
                plan = build_layout(tuple(LoraSpec(chr(65 + index)) for index in range(count)))
                self.assertEqual(
                    (
                        plan.unique_task_count,
                        plan.occupied_cell_count,
                        plan.blank_cell_count,
                        plan.duplicate_placement_count,
                    ),
                    values,
                )

    def test_three_lora_coordinate_weights(self) -> None:
        plan = build_layout(
            (LoraSpec("A.safetensors", 0.8), LoraSpec("B.safetensors", 1.0), LoraSpec("C.safetensors", 2.0))
        )
        cases = {
            (-4, 4): (0.8, 1.0, 0.0),
            (2, 3): (0.6, 0.0, 1.0),
            (0, 4): (0.8, 0.0, 0.0),
            (1, -3): (0.0, 0.75, 0.5),
            (-2, -2): (0.4, 1.0, 1.0),
            (-4, -2): (0.8, 1.0, 1.0),
        }
        for coordinate, weights in cases.items():
            with self.subTest(coordinate=coordinate):
                cell = plan.cell_at(coordinate)
                self.assertIsNotNone(cell.task_id)
                task = plan.task(cell.task_id or "")
                for actual, expected in zip(task.weights, weights):
                    self.assertAlmostEqual(actual, expected)
        self.assertIsNone(plan.cell_at((-4, -3)).task_id)

    def test_b_axis_task_is_reused(self) -> None:
        plan = build_layout((LoraSpec("A"), LoraSpec("B"), LoraSpec("C")))
        self.assertEqual(plan.placements_for("B075"), ((-3, 0), (0, -3)))
        self.assertEqual(plan.cell_at((-3, 1)).task_id, "A025+B075")

    def test_three_lora_layout_declares_table_style_axes(self) -> None:
        plan = build_layout((LoraSpec("LoraX"), LoraSpec("LoraY"), LoraSpec("LoraZZZ")))
        self.assertEqual(
            {(axis.key, axis.slot, axis.side) for axis in plan.axes},
            {
                ("B_TOP", "B", "top"),
                ("C_TOP", "C", "top"),
                ("A_LEFT", "A", "left"),
                ("A_RIGHT", "A", "right"),
                ("B_RIGHT", "B", "right"),
                ("C_BOTTOM", "C", "bottom"),
            },
        )

    def test_prompt_additions_follow_active_slots(self) -> None:
        plan = build_layout(
            (
                LoraSpec("A", trigger_word="alpha"),
                LoraSpec("B", trigger_word=""),
                LoraSpec("C", trigger_word="charlie"),
            )
        )
        task = plan.task("A075+C050")
        self.assertEqual(task.active_slots, ("A", "C"))
        self.assertEqual(task.prompt_additions, ("alpha", "charlie"))

    def test_min_weights_apply_at_the_center_and_to_prompt_additions(self) -> None:
        plan = build_layout(
            (
                LoraSpec("A", 0.9, "alpha", 0.2),
                LoraSpec("B", 3.0, "beta", 0.75),
                LoraSpec("C", 2.0, "charlie", 0.5),
            )
        )
        base = plan.task("base")
        self.assertEqual(base.weights, (0.2, 0.75, 0.5))
        self.assertEqual(base.active_slots, ("A", "B", "C"))
        self.assertEqual(base.prompt_additions, ("alpha", "beta", "charlie"))
        self.assertEqual(base.caption, "BASE / A 0.2 + B 0.75 + C 0.5")

        a_quarter = plan.task("A025")
        self.assertAlmostEqual(a_quarter.weights[0], 0.375)
        self.assertEqual(a_quarter.weights[1:], (0.75, 0.5))

    def test_min_weight_must_be_lower_than_max_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "lower than max_weight"):
            LoraSpec("A", 0.2, min_weight=0.2)
        with self.assertRaisesRegex(ValueError, "lower than max_weight"):
            LoraSpec("A", 0.2, min_weight=0.3)

    def test_rejects_invalid_lora_count_and_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "between one and three"):
            build_layout(())
        with self.assertRaisesRegex(ValueError, "between one and three"):
            build_layout((LoraSpec("A"), LoraSpec("B"), LoraSpec("C"), LoraSpec("D")))
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            LoraSpec(" ")
        with self.assertRaisesRegex(ValueError, "finite"):
            LoraSpec("A", float("nan"))


if __name__ == "__main__":
    unittest.main()
