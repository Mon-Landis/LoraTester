from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.compositor import CompositionSession, LoraComparisonCompositor, image_to_pil
from lora_tester.layout import LoraSpec
from lora_tester.styles import StyleConfig, register_style_decorator


class CompositorTests(unittest.TestCase):
    def make_three(self, **kwargs) -> LoraComparisonCompositor:
        return LoraComparisonCompositor(
            (
                LoraSpec("folder/角色画风_A.safetensors", 0.8),
                LoraSpec("B.safetensors", 1.0),
                LoraSpec("C.safetensors", 2.0),
            ),
            32,
            24,
            style=StyleConfig.black(decorator="none", caption_height=16, outer_margin=8, region_gap=8),
            max_canvas_pixels=None,
            **kwargs,
        )

    def test_template_has_expected_mode_and_size(self) -> None:
        compositor = self.make_three()
        image = compositor.render_template()
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, compositor.geometry.canvas_size)
        self.assertEqual(compositor.plan.unique_task_count, 69)

    def test_default_labels_regions_with_axes_instead_of_every_cell(self) -> None:
        compositor = LoraComparisonCompositor.from_values(
            ["LoraX", "LoraY", "LoraZZZ"],
            [0.9, 3.0, 2.0],
            640,
            800,
            style=StyleConfig.black(decorator="none"),
            max_canvas_pixels=None,
        )
        self.assertEqual(compositor.geometry.caption_height, 0)
        self.assertTrue(all(cell.caption_rect is None for cell in compositor.geometry.cells.values()))
        self.assertGreater(compositor.geometry.axis_font_size, 80)
        self.assertGreater(compositor.geometry.axis_value_font_size, 50)
        self.assertEqual(len(compositor.geometry.axis_rects), 6)
        axes = {axis.key: axis for axis in compositor.plan.axes}
        self.assertEqual(
            compositor._axis_value_labels(axes["A_LEFT"]),
            ("0.9", "0.675", "0.45", "0.225"),
        )
        self.assertEqual(
            compositor._axis_value_labels(axes["B_TOP"]),
            ("3", "2.25", "1.5", "0.75"),
        )
        self.assertEqual(
            compositor._axis_value_labels(axes["C_BOTTOM"]),
            ("0.5", "1", "1.5", "2"),
        )
        for geometry in compositor.geometry.cells.values():
            self.assertEqual(
                (geometry.image_rect[2] - geometry.image_rect[0], geometry.image_rect[3] - geometry.image_rect[1]),
                (640, 800),
            )

    def test_min_weights_are_reflected_in_axis_labels_and_specs(self) -> None:
        compositor = LoraComparisonCompositor.from_values(
            ["A", "B", "C"],
            [0.9, 3.0, 2.0],
            64,
            80,
            lora_min_weights=[0.2, 0.75, 0.5],
            style=StyleConfig.black(decorator="none"),
            max_canvas_pixels=None,
        )
        self.assertEqual(tuple(lora.min_weight for lora in compositor.plan.loras), (0.2, 0.75, 0.5))
        axes = {axis.key: axis for axis in compositor.plan.axes}
        self.assertEqual(
            compositor._axis_value_labels(axes["A_LEFT"]),
            ("0.9", "0.725", "0.55", "0.375"),
        )
        self.assertEqual(
            compositor._axis_value_labels(axes["B_TOP"]),
            ("3", "2.4375", "1.875", "1.3125"),
        )
        self.assertEqual(
            compositor._axis_value_labels(axes["C_BOTTOM"]),
            ("0.875", "1.25", "1.625", "2"),
        )

    def test_from_values_rejects_min_weight_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "lora_min_weights"):
            LoraComparisonCompositor.from_values(
                ["A", "B"],
                [1.0, 1.0],
                16,
                16,
                lora_min_weights=[0.1],
                max_canvas_pixels=None,
            )

    def test_type_scale_tracks_the_full_matrix_size(self) -> None:
        small = LoraComparisonCompositor.from_values(
            ["A", "B", "C"],
            [1.0, 1.0, 1.0],
            64,
            80,
            style=StyleConfig.black(decorator="none"),
            max_canvas_pixels=None,
        )
        large = LoraComparisonCompositor.from_values(
            ["A", "B", "C"],
            [1.0, 1.0, 1.0],
            640,
            800,
            style=StyleConfig.black(decorator="none"),
            max_canvas_pixels=None,
        )
        self.assertGreater(large.geometry.axis_font_size, small.geometry.axis_font_size * 5)
        self.assertGreater(large.geometry.axis_value_font_size, small.geometry.axis_value_font_size * 4)

    def test_side_axis_titles_and_values_are_all_rotated(self) -> None:
        compositor = self.make_three(show_lora_details=False)
        calls: list[str] = []
        original = CompositionSession._draw_vertical_text

        def capture(session, text, rect, **kwargs):
            calls.append(text)
            return original(session, text, rect, **kwargs)

        with patch.object(CompositionSession, "_draw_vertical_text", new=capture):
            compositor.render_template()

        self.assertEqual(len(calls), 15)
        self.assertIn("A / 角色画风_A", calls)
        self.assertIn("0.75", calls)

    def test_only_triple_cells_receive_labels_below_the_image(self) -> None:
        compositor = LoraComparisonCompositor.from_values(
            ["LoraX", "LoraY", "LoraZZZ"],
            [0.9, 3.0, 2.0],
            640,
            800,
            style=StyleConfig.black(decorator="none"),
            max_canvas_pixels=None,
        )
        labeled = [
            geometry
            for geometry in compositor.geometry.cells.values()
            if geometry.special_label_rect is not None
        ]
        self.assertEqual(len(labeled), 8)
        self.assertIsNone(compositor.geometry.cell((0, 0)).special_label_rect)
        self.assertTrue(
            all(
                geometry.cell_rect[3] - geometry.cell_rect[1] == 800
                for geometry in compositor.geometry.cells.values()
            )
        )
        for coordinate in ((-3, -1), (-4, -2), (-3, -3)):
            geometry = compositor.geometry.cell(coordinate)
            below = compositor.geometry.cell((coordinate[0], coordinate[1] - 1)).image_rect
            self.assertEqual(geometry.special_label_rect[1], geometry.image_rect[3])
            self.assertGreater(geometry.special_label_rect[3], below[1])
            self.assertLessEqual(geometry.special_label_rect[3], below[3])
        for coordinate in ((-4, -4), (-2, -4)):
            geometry = compositor.geometry.cell(coordinate)
            label_rect = geometry.special_label_rect
            self.assertEqual(label_rect[1], geometry.image_rect[3])
            self.assertLess(label_rect[3], compositor.geometry.footer_rect[1])

        session = compositor.start()
        red = Image.new("RGB", (640, 800), "red")
        task = session.submit(red, task_id="A100+B050+C050")
        coordinate = compositor.plan.placements_for(task.task_id)[0]
        image_rect = compositor.geometry.cell(coordinate).image_rect
        output = session.finalize(strict=False)
        self.assertEqual(output.getpixel((image_rect[0] + 20, image_rect[1] + 20)), (255, 0, 0))

    def test_artist_mixer_annotation_is_drawn_only_for_marked_submission(self) -> None:
        compositor = LoraComparisonCompositor.from_values(
            ["LoraX", "LoraY"],
            [1.0, 1.0],
            64,
            48,
            style=StyleConfig.black(decorator="none", show_axis_labels=False),
            show_lora_details=False,
            reserve_artist_mixer_labels=True,
            max_canvas_pixels=None,
        )
        calls: list[str] = []
        original = CompositionSession._draw_fitted_text

        def capture(session, text, rect, **kwargs):
            calls.append(str(text))
            return original(session, text, rect, **kwargs)

        with patch.object(CompositionSession, "_draw_fitted_text", new=capture):
            session = compositor.start()
            session.submit(Image.new("RGB", (64, 48), "red"), task_id="base")
            self.assertNotIn("Anima Artist Mixer", calls)
            session.submit(Image.new("RGB", (64, 48), "blue"), task_id="A100", artist_mixer=True)
            output = session.finalize(strict=False)

        self.assertIn("Anima Artist Mixer", calls)
        first_x = min(cell.image_rect[0] for cell in compositor.geometry.cells.values())
        first_column = sorted(
            (
                cell
                for cell in compositor.geometry.cells.values()
                if cell.artist_mixer_label_rect and cell.image_rect[0] == first_x
            ),
            key=lambda cell: cell.image_rect[1],
        )
        label = first_column[0].artist_mixer_label_rect
        self.assertIsNotNone(label)
        self.assertLessEqual(label[3], first_column[1].image_rect[1])
        mixer_label = compositor.geometry.cell(
            compositor.plan.placements_for("A100")[0]
        ).artist_mixer_label_rect
        self.assertTrue(
            np.any(np.all(np.asarray(output.crop(mixer_label)) == compositor.style.text_color, axis=2))
        )

    def test_footer_spacing_scales_with_large_composites(self) -> None:
        compositor = LoraComparisonCompositor.from_values(
            ["LoraX", "LoraY", "LoraZZZ"],
            [0.9, 3.0, 2.0],
            640,
            800,
            style=StyleConfig.black(decorator="none"),
            max_canvas_pixels=None,
        )
        self.assertGreater(compositor.geometry.footer_gap, compositor.style.footer_gap)
        self.assertGreater(compositor.geometry.footer_title_gap, 5)
        self.assertGreater(compositor.geometry.footer_name_line_gap, 0)

    def test_major_region_gaps_scale_with_the_image_size(self) -> None:
        compositor = LoraComparisonCompositor.from_values(
            ["LoraX", "LoraY", "LoraZZZ"],
            [0.9, 3.0, 2.0],
            640,
            800,
            style=StyleConfig.black(decorator="none"),
            max_canvas_pixels=None,
        )
        horizontal_gap = (
            compositor.geometry.cell((0, 0)).image_rect[0]
            - compositor.geometry.cell((-1, 0)).image_rect[2]
        )
        vertical_gap = (
            compositor.geometry.cell((0, 0)).image_rect[1]
            - compositor.geometry.cell((0, 1)).image_rect[3]
        )
        self.assertEqual(horizontal_gap, 64)
        self.assertEqual(vertical_gap, 64)

    def test_triple_images_have_individual_accent_frames_without_an_outer_frame(self) -> None:
        style = StyleConfig.black(decorator="none", frame_width=2, cell_frame_width=1)
        compositor = LoraComparisonCompositor.from_values(
            ["LoraX", "LoraY", "LoraZZZ"],
            [0.9, 3.0, 2.0],
            64,
            80,
            style=style,
            max_canvas_pixels=None,
        )
        session = compositor.start()
        coordinate = (-3, -1)
        task = compositor.plan.task(compositor.plan.cell_at(coordinate).task_id or "")
        expected = session._task_accent(task)
        region_rect = compositor.geometry.region_rects["ABC"]
        template = session.finalize(strict=False)
        self.assertEqual(template.getpixel((region_rect[0], region_rect[1])), style.background_color)

        session.submit(Image.new("RGB", (64, 80), "red"), task_id=task.task_id)
        output = session.finalize(strict=False)
        image_rect = compositor.geometry.cell(coordinate).image_rect
        self.assertEqual(output.getpixel((image_rect[0], image_rect[1])), expected)
        self.assertEqual(output.getpixel((image_rect[0] + 3, image_rect[1] + 3)), (255, 0, 0))

    def test_footer_toggle_removes_footer_and_space(self) -> None:
        with_footer = self.make_three(show_lora_details=True)
        without_footer = self.make_three(show_lora_details=False)
        self.assertIsNotNone(with_footer.geometry.footer_rect)
        self.assertIsNone(without_footer.geometry.footer_rect)
        self.assertLess(without_footer.geometry.canvas_size[1], with_footer.geometry.canvas_size[1])

    def test_footer_grows_to_preserve_full_original_name(self) -> None:
        style = StyleConfig.black(decorator="none", outer_margin=8)
        short = LoraComparisonCompositor(
            (LoraSpec("A.safetensors"),), 24, 20, style=style, max_canvas_pixels=None
        )
        long = LoraComparisonCompositor(
            (LoraSpec("very/long/角色画风_" + "segment_" * 24 + ".safetensors"),),
            24,
            20,
            style=style,
            max_canvas_pixels=None,
        )
        self.assertGreater(long.geometry.footer_rect[3] - long.geometry.footer_rect[1], short.geometry.footer_rect[3] - short.geometry.footer_rect[1])

    def test_single_b_submission_populates_both_axes(self) -> None:
        compositor = self.make_three(show_lora_details=False)
        session = compositor.start()
        red = Image.new("RGB", (32, 24), (255, 0, 0))
        session.submit(red, task_id="B075")
        output = session.finalize(strict=False)
        for coordinate in ((-3, 0), (0, -3)):
            rect = compositor.geometry.cell(coordinate).image_rect
            center = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
            self.assertEqual(output.getpixel(center), (255, 0, 0))

    def test_session_accepts_out_of_order_tasks_and_rejects_duplicates(self) -> None:
        compositor = self.make_three(show_lora_details=False)
        session = compositor.start()
        image = Image.new("RGB", (32, 24), "blue")
        session.submit(image, task_id="C100")
        session.submit(image, task_id="base")
        self.assertEqual(session.submitted_count, 2)
        with self.assertRaisesRegex(ValueError, "already been submitted"):
            session.submit(image, task_id="C100")
        with self.assertRaisesRegex(ValueError, "missing 67"):
            session.finalize(strict=True)

    def test_sequence_count_is_strict(self) -> None:
        compositor = LoraComparisonCompositor(
            (LoraSpec("A"),),
            12,
            10,
            style=StyleConfig.white(decorator="none", show_cell_captions=False),
            show_lora_details=False,
            max_canvas_pixels=None,
        )
        images = [Image.new("RGB", (12, 10), (index * 30, 0, 0)) for index in range(5)]
        self.assertEqual(compositor.compose(images).size, compositor.geometry.canvas_size)
        with self.assertRaisesRegex(ValueError, "more than 5"):
            compositor.compose(images + [images[0]])
        with self.assertRaisesRegex(ValueError, "missing 1"):
            compositor.compose(images[:-1])

    def test_image_fit_modes(self) -> None:
        strict = self.make_three(show_lora_details=False)
        with self.assertRaisesRegex(ValueError, "Expected image size"):
            strict.start().submit(Image.new("RGB", (8, 8)), task_id="base")
        contain = LoraComparisonCompositor(
            (LoraSpec("A"),),
            20,
            10,
            image_fit="contain",
            show_lora_details=False,
            style=StyleConfig.black(decorator="none"),
            max_canvas_pixels=None,
        )
        contain.start().submit(Image.new("RGB", (8, 8)), task_id="base")

    def test_numpy_and_tensor_shaped_images_convert(self) -> None:
        array = np.ones((1, 4, 5, 3), dtype=np.float32) * 0.5
        image = image_to_pil(array)
        self.assertEqual(image.size, (5, 4))
        self.assertEqual(image.getpixel((2, 2)), (127, 127, 127))
        with self.assertRaisesRegex(ValueError, "cannot contain a batch"):
            image_to_pil(np.zeros((2, 4, 5, 3), dtype=np.float32))

    def test_custom_style_and_decorator_extension_point(self) -> None:
        name = f"test-{uuid.uuid4()}"

        class TestDecorator:
            def draw_background(self, context) -> None:
                context.draw.point((0, 0), fill=(3, 4, 5))

            def draw_foreground(self, context) -> None:
                context.draw.point((1, 0), fill=(6, 7, 8))

        register_style_decorator(name, TestDecorator())
        style = StyleConfig.custom(
            background_color="#123456",
            panel_color="#202122",
            placeholder_color="#303132",
            text_color="#FAFAFA",
            muted_text_color="#AAAAAA",
            frame_color="#BBBBBB",
            accent_colors=("#FF0000", "#00FF00", "#0000FF"),
            decorator=name,
        )
        compositor = LoraComparisonCompositor(
            (LoraSpec("A"),), 8, 8, style=style, show_lora_details=False, max_canvas_pixels=None
        )
        image = compositor.render_template()
        self.assertEqual(image.getpixel((0, 0)), (3, 4, 5))
        self.assertEqual(image.getpixel((1, 0)), (6, 7, 8))

    def test_background_image_is_supported(self) -> None:
        background = Image.new("RGB", (2, 2), (12, 34, 56))
        style = StyleConfig.custom(background_image=background, background_fit="tile", decorator="none")
        compositor = LoraComparisonCompositor(
            (LoraSpec("A"),), 8, 8, style=style, show_lora_details=False, max_canvas_pixels=None
        )
        image = compositor.render_template()
        self.assertEqual(image.getpixel((0, 0)), (12, 34, 56))

    def test_comfy_shaped_background_is_supported_without_temp_file(self) -> None:
        background = np.zeros((1, 2, 2, 3), dtype=np.float32)
        background[..., 1] = 1.0
        style = StyleConfig.custom(background_image=background, background_fit="tile", decorator="none")
        compositor = LoraComparisonCompositor(
            (LoraSpec("A"),), 8, 8, style=style, show_lora_details=False, max_canvas_pixels=None
        )
        image = compositor.render_template()
        self.assertEqual(image.getpixel((0, 0)), (0, 255, 0))

    def test_canvas_guard_reports_large_allocations(self) -> None:
        with self.assertRaisesRegex(MemoryError, "Composite canvas"):
            LoraComparisonCompositor(
                (LoraSpec("A"), LoraSpec("B"), LoraSpec("C")),
                128,
                128,
                max_canvas_pixels=100,
            )


if __name__ == "__main__":
    unittest.main()
