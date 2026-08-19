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
    GlobalPromptAppendNode,
    MultiPromptInputNode,
    SeedListNode,
    XYTestSampler,
)
from lora_tester.stack import LoraStack, LoraStackItem, LoraStackList
from lora_tester.styles import StyleConfig
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
            "portrait\nlandscape",
            "lines",
            "---",
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
        axis = build_lora_stack_axis(LoraStackList((low, high)), include_base=True)
        self.assertEqual([entry.label for entry in axis.entries], ["BASE", "A-0.5", "A-1"])
        self.assertEqual(axis.group_breaks, (1,))
        sources = next(block for block in axis.detail_blocks if block.title == "STYLE SOURCES")
        self.assertEqual(sources.rows, (("A", "LORA", "Shared"),))
        triggers = next(block for block in axis.detail_blocks if block.title == "LORA TRIGGERS")
        self.assertEqual(len(triggers.rows), 2)

    def test_artist_tags_also_share_source_codes_across_weights(self) -> None:
        low = LoraStack((LoraStackItem(ARTIST_TAG_MODE, "@fkey", 0.5),))
        high = LoraStack((LoraStackItem(ARTIST_TAG_MODE, "@fkey", 1.0),))
        axis = build_lora_stack_axis(LoraStackList((low, high)), include_base=False)
        self.assertEqual([entry.label for entry in axis.entries], ["A-0.5", "A-1"])

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


class XYCompositorTests(unittest.TestCase):
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
            ["table", "text"],
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
