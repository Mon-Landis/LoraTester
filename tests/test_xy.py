from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.artist import ARTIST_TAG_MODE
from lora_tester.nodes import (
    AxisComposerNode,
    GlobalPromptAppendNode,
    LoraStackAxisNode,
    MultiPromptInputNode,
    SeedListNode,
    XYTestSampler,
)
from lora_tester.stack import LoraStack, LoraStackItem, LoraStackList
from lora_tester.styles import StyleConfig, mix_color
from lora_tester.xy import (
    AxisEntry,
    AxisParameter,
    DetailBlock,
    PromptEntry,
    PromptList,
    SeedList,
    XYAxis,
    build_lora_stack_axis,
    build_prompt_axis,
    build_seed_axis,
    concatenate_axes,
    cross_merge_axes,
    merge_axis_parameters,
)
from lora_tester.xy_compositor import XYMatrixCompositor


class _Clip:
    def __init__(self, stack=()):
        self.stack = tuple(stack)

    def tokenize(self, text):
        return {"text": text, "stack": self.stack}

    def encode_from_tokens_scheduled(self, tokens):
        return tokens


class _Vae:
    def decode(self, _latent):
        return torch.zeros((1, 8, 10, 3))


class _Progress:
    def update_absolute(self, *_args, **_kwargs):
        return None


class XYModelTests(unittest.TestCase):
    def test_long_prompt_parser_and_global_append_preserve_artist_boundary(self) -> None:
        prompt_list = PromptList.parse(
            "@ordinary_prompt, portrait\n\nlandscape",
            separator_mode="blank_lines",
        )
        appended = prompt_list.append_global(
            "masterpiece",
            position="before",
            independent_artist_tags="@fkey, (@ciloranko:0.5)",
        )
        self.assertEqual(
            [entry.full_prompt for entry in appended.entries],
            ["masterpiece, @ordinary_prompt, portrait", "masterpiece, landscape"],
        )
        self.assertEqual(
            appended.entries[0].independent_artist_tags,
            "@fkey, (@ciloranko:0.5)",
        )

    def test_prompt_nodes_expose_the_requested_chain(self) -> None:
        prompt_list = MultiPromptInputNode.build_prompts(
            prompt_count=2,
            positive_prompt_1="portrait",
            positive_prompt_2="landscape",
        )[0]
        result = GlobalPromptAppendNode.append_prompt(
            prompt_list,
            "masterpiece",
            "after",
            "@artist",
        )[0]
        self.assertEqual(
            [entry.full_prompt for entry in result.entries],
            ["portrait, masterpiece", "landscape, masterpiece"],
        )
        self.assertEqual(result.entries[1].independent_artist_tags, "@artist")

    def test_multi_prompt_input_uses_separate_rows(self) -> None:
        inputs = MultiPromptInputNode.INPUT_TYPES()["required"]
        self.assertIn("prompt_count", inputs)
        self.assertEqual(inputs["prompt_count"][0], "INT")
        self.assertEqual(
            {name for name in inputs if name.startswith("positive_prompt_")},
            {f"positive_prompt_{index}" for index in range(1, 17)},
        )
        with self.assertRaisesRegex(ValueError, "row 2"):
            MultiPromptInputNode.build_prompts(
                prompt_count=2,
                positive_prompt_1="portrait",
                positive_prompt_2="",
            )

    def test_axis_composer_accepts_all_raw_axis_sources(self) -> None:
        prompt_axis = AxisComposerNode.compose_axis(
            "PROMPTS",
            True,
            PromptList((PromptEntry("portrait"),)),
        )[0]
        style = LoraStack((LoraStackItem("A.safetensors", "alpha", 0.8),))
        style_axis = AxisComposerNode.compose_axis(
            "STYLE",
            True,
            LoraStackList((style,)),
        )[0]
        seed_axis = AxisComposerNode.compose_axis("SEEDS", True, SeedList((11, 22)))[0]

        self.assertEqual(prompt_axis.title, "PROMPTS")
        self.assertEqual(style_axis.entries[0].label, "BASE")
        self.assertEqual(seed_axis.entries[1].parameter_map["seed"], 22)
        with self.assertRaisesRegex(TypeError, "expects a PromptList"):
            AxisComposerNode.compose_axis("AXIS", True, object())

    def test_axis_operations_are_whole_axis_and_preserve_group_boundaries(self) -> None:
        prompts = build_prompt_axis(PromptList((PromptEntry("portrait"),)))
        seeds = build_seed_axis(SeedList((3, 4)))
        combined = cross_merge_axes(prompts, seeds, title="COMBINED")
        self.assertEqual(combined.title, "COMBINED")
        self.assertEqual(len(combined.entries), 2)
        self.assertEqual(combined.entries[0].parameter_map["seed"], 3)
        concatenated = concatenate_axes(prompts, seeds, title="CONCAT")
        self.assertEqual(concatenated.title, "CONCAT")
        self.assertEqual(len(concatenated.groups), 2)

    def test_axis_groups_are_two_dimensional_and_conflicts_are_rejected(self) -> None:
        x = AxisEntry("X", (AxisParameter("seed", 1),))
        y = AxisEntry("Y", (AxisParameter("prompt", PromptEntry("portrait")),))
        self.assertEqual(set(merge_axis_parameters(x, y)), {"seed", "prompt"})
        with self.assertRaisesRegex(ValueError, "Both axes assign"):
            merge_axis_parameters(x, AxisEntry("Y", (AxisParameter("seed", 2),)))
        axis = XYAxis("GROUPED", ((x,), (AxisEntry("X2", ()),)))
        self.assertEqual(axis.group_breaks, (1,))
        self.assertEqual(axis.data, (((AxisParameter("seed", 1),),), ((),)))

    def test_same_lora_at_different_weights_reuses_one_source_code(self) -> None:
        low = LoraStack((LoraStackItem("Shared.safetensors", "low", 0.5),))
        high = LoraStack((LoraStackItem("Shared.safetensors", "high", 1.0),))
        axis = build_lora_stack_axis(LoraStackList((low, high)), include_base=True, show_single_style_name=False)
        self.assertEqual([entry.label for entry in axis.entries], ["BASE", "A-0.5", "A-1"])
        self.assertEqual(axis.group_breaks, (1,))
        sources = next(block for block in axis.detail_blocks if block.title == "STYLE SOURCES")
        self.assertEqual(sources.headers, ("CODE", "TYPE", "SOURCE", "INFO"))
        self.assertEqual(sources.rows, (("A", "LORA", "Shared", "low; high"),))
        self.assertFalse(any(block.title == "STYLE CONFIGURATIONS" for block in axis.detail_blocks))

    def test_prompt_axis_does_not_repeat_prompt_text_in_footer(self) -> None:
        axis = build_prompt_axis(PromptList((PromptEntry("portrait"), PromptEntry("landscape"))))
        self.assertEqual(axis.detail_blocks, ())

    def test_artist_tags_also_share_source_codes_across_weights(self) -> None:
        low = LoraStack((LoraStackItem(ARTIST_TAG_MODE, "@fkey", 0.5),))
        high = LoraStack((LoraStackItem(ARTIST_TAG_MODE, "@fkey", 1.0),))
        axis = build_lora_stack_axis(LoraStackList((low, high)), include_base=False, show_single_style_name=False)
        self.assertEqual([entry.label for entry in axis.entries], ["A-0.5", "A-1"])
        sources = next(block for block in axis.detail_blocks if block.title == "STYLE SOURCES")
        self.assertEqual(sources.rows, (("A", "ARTIST", "fkey", "artist tag"),))

    def test_single_artist_names_preserve_full_artist_name(self) -> None:
        stacks = LoraStackList((
            LoraStack((LoraStackItem(ARTIST_TAG_MODE, "@wlop", 0.5),)),
            LoraStack((LoraStackItem(ARTIST_TAG_MODE, "@nekoya_(nekodayo_22)", 1.2),)),
            LoraStack((LoraStackItem(ARTIST_TAG_MODE, "@wlop", 1.0),)),
        ))
        axis = build_lora_stack_axis(stacks)
        self.assertEqual(
            [entry.label for entry in axis.entries],
            ["BASE", "A-wlop-0.5", "B-nekoya_(nekodayo_22)-1.2", "A-wlop-1"],
        )
        self.assertEqual(axis.group_breaks, (1,))
        for entry, stack in zip(axis.entries[1:], stacks.stacks):
            self.assertIs(entry.parameter_map["lora_stack"], stack)
            self.assertEqual(entry.detail_label, stack.label)
        legacy = build_lora_stack_axis(stacks, show_single_style_name=False)
        self.assertEqual(axis.detail_blocks, legacy.detail_blocks)

    def test_single_lora_names_use_basename_prefix(self) -> None:
        examples = (
            ("styles/foo_bar.safetensors", "foo"),
            (r"styles\foo bar.safetensors", "foo"),
            ("styles/plain.safetensors", "plain"),
            ("styles/version.v2.safetensors", "version.v2"),
            ("styles/foo_bar baz.safetensors", "foo"),
            ("styles/foo bar_baz.safetensors", "foo"),
            ("styles/画风_版本.safetensors", "画风"),
            ("styles/foo\tbar.safetensors", "foo"),
            ("styles/_prefix.safetensors", "_prefix"),
        )
        for filename, expected in examples:
            with self.subTest(filename=filename):
                stack = LoraStack((LoraStackItem(filename, "trigger", 0.8),))
                axis = build_lora_stack_axis(LoraStackList((stack,)), include_base=False)
                self.assertEqual(axis.entries[0].label, f"A-{expected}-0.8")
                self.assertEqual(axis.entries[0].parameter_map["lora_stack"].items[0].strength, 0.8)

    def test_single_style_labels_preserve_zero_and_negative_weights(self) -> None:
        for item in (LoraStackItem("foo_bar.safetensors"), LoraStackItem(ARTIST_TAG_MODE, "@wlop")):
            for strength in (0.0, -0.5, 1.25):
                with self.subTest(item=item, strength=strength):
                    stack = LoraStack((LoraStackItem(item.name, item.trigger_word, strength),))
                    axis = build_lora_stack_axis(LoraStackList((stack,)), include_base=False)
                    name = "wlop" if item.is_artist_tag else "foo"
                    self.assertEqual(axis.entries[0].label, f"A-{name}-{strength:g}")

    def test_multisource_styles_keep_code_weight_labels(self) -> None:
        artists = LoraStack((LoraStackItem(ARTIST_TAG_MODE, "@wlop, @aos", 0.5),))
        mixed = LoraStack((
            LoraStackItem(ARTIST_TAG_MODE, "@wlop", 0.8),
            LoraStackItem("foo_bar.safetensors", "foo", 1.2),
        ))
        repeated = LoraStack((
            LoraStackItem("foo_bar.safetensors", "foo", 0.3),
            LoraStackItem("foo_bar.safetensors", "foo", 0.4),
        ))
        axis = build_lora_stack_axis(LoraStackList((artists, mixed, repeated)), include_base=False)
        self.assertEqual([entry.label for entry in axis.entries], ["A-0.5+B-0.5", "A-0.8+C-1.2", "C-0.3+C-0.4"])

    def test_custom_single_style_names_override_both_modes(self) -> None:
        for item in (LoraStackItem("foo_bar.safetensors"), LoraStackItem(ARTIST_TAG_MODE, "@wlop")):
            for mode in (True, False):
                with self.subTest(item=item, mode=mode):
                    stack = LoraStack((item,), custom_name="自定义风格")
                    axis = build_lora_stack_axis(LoraStackList((stack,)), include_base=False, show_single_style_name=mode)
                    self.assertEqual(axis.entries[0].label, "自定义风格")
                    self.assertIs(axis.entries[0].parameter_map["lora_stack"], stack)

    def test_single_style_labels_keep_spreadsheet_source_numbering(self) -> None:
        stacks = LoraStackList(tuple(
            LoraStack((LoraStackItem(f"style{index}_v2.safetensors"),))
            for index in range(28)
        ))
        axis = build_lora_stack_axis(stacks, include_base=False)
        self.assertEqual([entry.label for entry in axis.entries[-3:]], ["Z-style25-1", "AA-style26-1", "AB-style27-1"])
        self.assertEqual([row[0] for row in axis.detail_blocks[0].rows[-3:]], ["Z", "AA", "AB"])

    def test_single_style_switch_is_default_on_and_appended_to_widgets(self) -> None:
        source = LoraStackList((LoraStack((LoraStackItem("foo_bar.safetensors", "", 0.8),)),))
        for node, old_widgets in (
            (LoraStackAxisNode, ["lorastacks", "include_base", "axis_title"]),
            (AxisComposerNode, ["axis_title", "include_base"]),
        ):
            with self.subTest(node=node.__name__):
                required = node.INPUT_TYPES()["required"]
                self.assertEqual(list(required), old_widgets + ["show_single_style_name"])
                self.assertEqual(required["show_single_style_name"][0], "BOOLEAN")
                self.assertIs(required["show_single_style_name"][1]["default"], True)
        direct = LoraStackAxisNode.build_axis(source, False, "STYLE")[0]
        self.assertEqual(direct.entries[0].label, "A-foo-0.8")
        self.assertEqual(AxisComposerNode.compose_axis("STYLE", False, source)[0], direct)
        self.assertEqual(AxisComposerNode.compose_axis("STYLE", False, source.stacks[0])[0], direct)
        legacy = LoraStackAxisNode.build_axis(source, False, "STYLE", False)[0]
        self.assertEqual(legacy.entries[0].label, "A-0.8")
        self.assertEqual(AxisComposerNode.compose_axis("STYLE", False, source, False)[0], legacy)

    def test_single_style_switch_does_not_relabel_existing_axes_or_other_sources(self) -> None:
        stack = LoraStack((LoraStackItem("foo_bar.safetensors", "", 0.8),))
        legacy = build_lora_stack_axis(LoraStackList((stack,)), show_single_style_name=False)
        self.assertEqual(AxisComposerNode.compose_axis("STYLE", False, legacy, True)[0], legacy)
        for source in (PromptList((PromptEntry("portrait"),)), SeedList((42,))):
            with self.subTest(source=source):
                self.assertEqual(
                    AxisComposerNode.compose_axis("AXIS", False, source, True),
                    AxisComposerNode.compose_axis("AXIS", False, source, False),
                )

    def test_single_style_switch_preserves_empty_base_only_axis(self) -> None:
        axis = build_lora_stack_axis(LoraStackList(()))
        self.assertEqual([entry.label for entry in axis.entries], ["BASE"])
        self.assertEqual(axis.detail_blocks, ())

    def test_seed_list_accepts_explicit_and_deterministic_random_values(self) -> None:
        self.assertEqual(SeedList.parse("1, 2\n3").seeds, (1, 2, 3))
        first = SeedList.random(5, 1234)
        second = SeedList.random(5, 1234)
        self.assertEqual(first, second)
        self.assertEqual(
            SeedListNode.build_seeds("random", "ignored", 5, 1234)[0],
            first,
        )
        axis = build_seed_axis(first)
        self.assertEqual(len(axis.entries), 5)
        self.assertEqual(axis.entries[0].parameter_map["seed"], first.seeds[0])

    def test_random_generator_seed_sentinel_uses_a_fresh_source_seed(self) -> None:
        with patch("lora_tester.xy.random.SystemRandom") as system_random:
            system_random.return_value.randrange.return_value = 987654321
            generated = SeedListNode.build_seeds("random", "ignored", 4, -1)[0]
        self.assertEqual(generated, SeedList.random(4, 987654321))
        system_random.return_value.randrange.assert_called_once_with(2**64)

    def test_seed_list_node_exposes_random_generator_seed_sentinel(self) -> None:
        options = SeedListNode.INPUT_TYPES()["required"]["random_source_seed"][1]
        self.assertEqual(options["min"], -1)
        self.assertIn("-1", options["tooltip"])
        with self.assertRaisesRegex(ValueError, "Generator seed must be -1"):
            SeedList.random(1, -2)


class XYCompositorTests(unittest.TestCase):
    def test_detail_tables_have_subtle_alternating_full_width_rows(self) -> None:
        styles = (
            StyleConfig.black(decorator="none"),
            StyleConfig.white(decorator="none"),
            StyleConfig.custom(panel_color="#24384C", text_color="#E8F0FF"),
        )
        block = DetailBlock("STYLE SOURCES", "table", ("CODE", "TYPE", "SOURCE", "INFO"), tuple(
            (f"CODE{index}", "LORA", f"long_source_{index}", f"long trigger row {index}")
            for index in range(4)
        ))
        x = XYAxis("STYLE", ((AxisEntry("A", ()),),), (block,))
        y = build_seed_axis(SeedList((123, 456, 789)))
        for style in styles:
            with self.subTest(mode=style.mode):
                compositor = XYMatrixCompositor(x, y, 96, 64, style=style, max_canvas_pixels=None)
                images = [Image.new("RGB", (96, 64), (110, 90, 70)) for _ in y.entries]
                rendered = compositor.compose(images)
                try:
                    for detail in compositor.geometry.detail_blocks:
                        rect = detail.content_rect
                        row_count = len(detail.block.rows) + 1
                        row_height = (rect[3] - rect[1]) // row_count
                        column_width = (rect[2] - rect[0]) / len(detail.block.headers)
                        for row_index in range(row_count):
                            top = rect[1] + row_index * row_height
                            bottom = rect[3] if row_index == row_count - 1 else top + row_height
                            expected = mix_color(style.panel_color, style.text_color, 0.08) if row_index % 2 else style.panel_color
                            for column_index in range(len(detail.block.headers)):
                                left = round(rect[0] + column_index * column_width)
                                self.assertEqual(rendered.getpixel((left + 2, (top + bottom) // 2)), expected)
                            if row_index:
                                self.assertEqual(rendered.getpixel((rect[0] + 2, top)), style.frame_color)
                            self.assertEqual(rendered.getpixel((round(rect[0] + column_width), top + 2)), style.frame_color)
                    cell = compositor.geometry.cell(0, 0)
                    self.assertEqual(rendered.getpixel(((cell[0] + cell[2]) // 2, (cell[1] + cell[3]) // 2)), (110, 90, 70))
                finally:
                    rendered.close()
                    for image in images:
                        image.close()

    def test_header_only_detail_table_keeps_original_background(self) -> None:
        style = StyleConfig.black(decorator="none")
        block = DetailBlock("EMPTY", "table", ("CODE",), ())
        axis = XYAxis("STYLE", ((AxisEntry("BASE", ()),),), (block,))
        compositor = XYMatrixCompositor(axis, build_seed_axis(SeedList((1,))), 96, 64, style=style)
        rendered = compositor.render_template()
        try:
            rect = compositor.geometry.detail_blocks[0].content_rect
            self.assertEqual(rendered.getpixel((rect[0] + 2, (rect[1] + rect[3]) // 2)), style.panel_color)
        finally:
            rendered.close()

    def make_axes(self) -> tuple[XYAxis, XYAxis]:
        x = XYAxis(
            "SEED",
            (
                (AxisEntry("BASE", (AxisParameter("seed", 1),)),),
                (
                    AxisEntry("2", (AxisParameter("seed", 2),)),
                    AxisEntry("3", (AxisParameter("seed", 3),)),
                ),
            ),
            (
                DetailBlock(
                    "SEEDS",
                    "table",
                    headers=("INDEX", "SEED"),
                    rows=(("S01", "1"), ("S02", "2"), ("S03", "3")),
                ),
            ),
        )
        y = build_prompt_axis(PromptList((PromptEntry("portrait"), PromptEntry("landscape"))))
        return x, y

    def test_group_breaks_and_multiple_detail_modes_affect_layout(self) -> None:
        x, y = self.make_axes()
        compositor = XYMatrixCompositor(
            x,
            y,
            32,
            24,
            style=StyleConfig.black(decorator="none"),
            max_canvas_pixels=None,
        )
        first = compositor.geometry.cell(0, 0)
        second = compositor.geometry.cell(0, 1)
        third = compositor.geometry.cell(0, 2)
        self.assertGreater(second[0] - first[2], third[0] - second[2])
        self.assertEqual(
            [item.block.mode for item in compositor.geometry.detail_blocks],
            ["table"],
        )

    def test_session_pastes_in_place_and_finalizes_without_a_canvas_copy(self) -> None:
        x, y = self.make_axes()
        compositor = XYMatrixCompositor(
            x,
            y,
            8,
            6,
            style=StyleConfig.black(decorator="none"),
            max_canvas_pixels=None,
        )
        session = compositor.start()
        for row in range(compositor.row_count):
            for column in range(compositor.column_count):
                session.submit(Image.new("RGB", (8, 6), (row * 50, column * 50, 10)), coordinate=(row, column))
        result = session.finalize()
        self.assertIs(result, session._canvas)
        self.assertEqual(result.size, compositor.geometry.canvas_size)

    def test_extra_footer_text_renders_even_when_axis_details_are_hidden(self) -> None:
        x, y = self.make_axes()
        compositor = XYMatrixCompositor(
            x,
            y,
            8,
            6,
            style=StyleConfig.white(decorator="none"),
            show_details=False,
            extra_detail_text="workflow revision 12",
            max_canvas_pixels=None,
        )
        self.assertEqual(
            [(item.block.title, item.block.text) for item in compositor.geometry.detail_blocks],
            [("NOTES", ("workflow revision 12",))],
        )

    def test_explicit_x_group_gap_keeps_legacy_control_separation(self) -> None:
        x, y = self.make_axes()
        compositor = XYMatrixCompositor(
            x,
            y,
            80,
            40,
            style=StyleConfig.black(decorator="none"),
            x_group_gap=0,
            max_canvas_pixels=None,
        )
        first = compositor.geometry.cell(0, 0)
        second = compositor.geometry.cell(0, 1)
        self.assertGreaterEqual(second[0] - first[2], 10)

    def test_row_labels_are_rotated_vertically_in_the_side_rail(self) -> None:
        x, y = self.make_axes()
        style = StyleConfig.black(decorator="none")
        compositor = XYMatrixCompositor(
            x,
            y,
            80,
            64,
            style=style,
            max_canvas_pixels=None,
        )
        rendered = compositor.render_template()
        try:
            rect = compositor.geometry.row_label_rects[0]
            crop = rendered.crop(rect)
            background = style.background_color
            pixels = crop.load()
            points = [
                (column, row)
                for row in range(crop.height)
                for column in range(crop.width)
                if pixels[column, row] != background
            ]
            self.assertTrue(points)
            min_x = min(point[0] for point in points)
            max_x = max(point[0] for point in points)
            min_y = min(point[1] for point in points)
            max_y = max(point[1] for point in points)
            self.assertGreater(max_y - min_y, max_x - min_x)
        finally:
            rendered.close()


class XYSamplerTests(unittest.TestCase):
    def test_sampler_rejects_same_parameter_on_both_axes_before_sampling(self) -> None:
        x_axis = build_seed_axis(SeedList((1, 2)))
        y_axis = build_seed_axis(SeedList((3, 4)))
        with patch("lora_tester.nodes._common_ksampler") as sample:
            with self.assertRaisesRegex(ValueError, "cannot both modify.*seed"):
                XYTestSampler().sample(
                    model=(),
                    clip=_Clip(),
                    vae=_Vae(),
                    latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                    x_axis=x_axis,
                    y_axis=y_axis,
                    positive_prompt="portrait",
                    negative_prompt="",
                    seed=0,
                    steps=1,
                    cfg=1.0,
                    sampler_name="sampler",
                    scheduler="scheduler",
                    denoise=1.0,
                    color_mode="black",
                    show_axis_details=False,
                    log_test_details=False,
                    max_canvas_megapixels=10.0,
                )
        sample.assert_not_called()

    def test_prompt_y_seed_x_uses_cell_seeds_and_returns_row_major_raw_batch(self) -> None:
        x_axis = build_seed_axis(SeedList((11, 22, 33)))
        y_axis = build_prompt_axis(
            PromptList((PromptEntry("portrait"), PromptEntry("landscape")))
        )
        calls = []

        def sample(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise, **_kwargs):
            calls.append((seed, positive["text"]))
            return latent

        with (
            patch("lora_tester.nodes._common_ksampler", side_effect=sample),
            patch("lora_tester.nodes._make_progress_bar", return_value=_Progress()),
            patch("lora_tester.nodes._throw_if_interrupted"),
        ):
            sheet, raw = XYTestSampler().sample(
                model=(),
                clip=_Clip(),
                vae=_Vae(),
                latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                x_axis=x_axis,
                y_axis=y_axis,
                positive_prompt="fallback",
                negative_prompt="",
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="sampler",
                scheduler="scheduler",
                denoise=1.0,
                color_mode="black",
                show_axis_details=True,
                log_test_details=False,
                max_canvas_megapixels=10.0,
            )
        self.assertEqual(
            calls,
            [
                (11, "portrait"),
                (22, "portrait"),
                (33, "portrait"),
                (11, "landscape"),
                (22, "landscape"),
                (33, "landscape"),
            ],
        )
        self.assertEqual(tuple(raw.shape), (6, 8, 10, 3))
        self.assertEqual(int(sheet.shape[0]), 1)

    def test_seed_can_be_the_y_axis_and_prompt_the_x_axis(self) -> None:
        x_axis = build_prompt_axis(PromptList((PromptEntry("a"), PromptEntry("b"))))
        y_axis = build_seed_axis(SeedList((7, 8)))
        calls = []

        def sample(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise, **_kwargs):
            calls.append((seed, positive["text"]))
            return latent

        with (
            patch("lora_tester.nodes._common_ksampler", side_effect=sample),
            patch("lora_tester.nodes._make_progress_bar", return_value=_Progress()),
            patch("lora_tester.nodes._throw_if_interrupted"),
        ):
            XYTestSampler().sample(
                model=(),
                clip=_Clip(),
                vae=_Vae(),
                latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                x_axis=x_axis,
                y_axis=y_axis,
                positive_prompt="",
                negative_prompt="",
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="sampler",
                scheduler="scheduler",
                denoise=1.0,
                color_mode="white",
                show_axis_details=False,
                log_test_details=False,
                max_canvas_megapixels=10.0,
            )
        self.assertEqual(calls, [(7, "a"), (7, "b"), (8, "a"), (8, "b")])


if __name__ == "__main__":
    unittest.main()
