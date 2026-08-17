from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.nodes import (
    LoraStackListerNode,
    LoraStackNode,
    LoraStackSplitterNode,
    MultiPromptSampleNode,
)
from lora_tester.artist import ARTIST_TAG_MODE, ArtistTagTemplate
from lora_tester.stack import LoraStack, LoraStackItem, LoraStackList, split_lora_stack
from lora_tester.styles import StyleConfig


class _Clip:
    def __init__(self, stack=()):
        self.stack = tuple(stack)

    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return {"text": tokens, "stack": self.stack}


class _Vae:
    def decode(self, _latent):
        return torch.zeros((1, 8, 10, 3), dtype=torch.float32)


class _Progress:
    def __init__(self):
        self.updates = []

    def update_absolute(self, value, total=None, preview=None):
        self.updates.append((value, total))


class StackNodeTests(unittest.TestCase):
    def test_multi_prompt_input_contract_contains_advanced_log_toggle(self):
        inputs = MultiPromptSampleNode.INPUT_TYPES()
        options = inputs["required"]["log_test_details"][1]
        self.assertTrue(options["default"])
        self.assertTrue(options["advanced"])
        mixer_options = inputs["required"]["use_anima_artist_mixer"][1]
        self.assertTrue(mixer_options["default"])
        self.assertTrue(mixer_options["advanced"])

    def test_stack_node_builds_requested_entries_and_rejects_blank_active_file(self):
        node = LoraStackNode()
        values = {
            "lora_1_name": "A.safetensors",
            "lora_1_trigger": "alpha",
            "lora_1_strength": 0.75,
            "lora_2_name": "B.safetensors",
            "lora_2_trigger": "beta",
            "lora_2_strength": 1.25,
        }
        stack = node.build_stack(2, **values)[0]
        self.assertEqual(stack.label, "A + B")
        self.assertEqual(stack.trigger_words, ("alpha", "beta"))
        self.assertEqual(tuple(item.strength for item in stack.items), (0.75, 1.25))
        with self.assertRaisesRegex(ValueError, "entry 2"):
            node.build_stack(2, **{**values, "lora_2_name": ""})

    def test_stack_validator_ignores_missing_hidden_entries(self):
        def resolve(name):
            if name in {"A.safetensors", "B.safetensors"}:
                return name
            raise FileNotFoundError(name)

        with patch("lora_tester.nodes._resolve_lora_path", side_effect=resolve):
            self.assertTrue(
                LoraStackNode.VALIDATE_INPUTS(
                    lora_count=2,
                    lora_1_name="A.safetensors",
                    lora_2_name="B.safetensors",
                    lora_3_name="missing.safetensors",
                )
            )
            self.assertIn(
                "active slot 2",
                LoraStackNode.VALIDATE_INPUTS(
                    lora_count=2,
                    lora_1_name="A.safetensors",
                    lora_2_name="missing.safetensors",
                    lora_3_name="missing.safetensors",
                ),
            )

    def test_stack_artist_mode_and_template_survive_splitter(self):
        template = ArtistTagTemplate("artist:{tag}", "(artist:{tag}:{weight})")
        stack = LoraStackNode().build_stack(
            2,
            artist_tag_template=template,
            lora_1_name=ARTIST_TAG_MODE,
            lora_1_trigger="@first, second",
            lora_1_strength=1.25,
            lora_2_name="A.safetensors",
            lora_2_trigger="alpha",
            lora_2_strength=0.8,
        )[0]
        self.assertEqual(stack.label, "first, second + A")
        self.assertEqual(stack.trigger_words, ("alpha",))
        self.assertEqual(stack.artist_entries, (("first", 1.25), ("second", 1.25)))
        split = split_lora_stack(stack)
        self.assertTrue(all(item.artist_template is template for item in split.stacks))

    def test_stack_is_changed_tracks_active_lora_file_signature(self):
        with (
            patch(
                "lora_tester.nodes._resolve_lora_path",
                side_effect=lambda name: str(ROOT / name),
            ),
            patch(
                "lora_tester.nodes._lora_file_signature",
                return_value=(100, 200, 300),
            ),
        ):
            fingerprint = LoraStackNode.IS_CHANGED(
                lora_count=2,
                lora_1_name=ARTIST_TAG_MODE,
                lora_2_name="A.safetensors",
                lora_3_name="ignored.safetensors",
            )

        self.assertEqual(len(fingerprint), 1)
        self.assertTrue(fingerprint[0][0].lower().endswith("a.safetensors"))
        self.assertEqual(fingerprint[0][1], (100, 200, 300))

    def test_splitter_and_lister_nodes_use_custom_types(self):
        stack = LoraStack((LoraStackItem("A.safetensors"), LoraStackItem("B.safetensors")))
        split = LoraStackSplitterNode.split_stack(stack)[0]
        result = LoraStackListerNode.list_stacks(split.stacks[0], stack_2=split.stacks[1])[0]
        self.assertIsInstance(result, LoraStackList)
        self.assertEqual([item.label for item in result.stacks], ["A", "B"])

    def test_multi_prompt_sample_uses_base_column_and_shared_seed(self):
        stack = LoraStack((LoraStackItem("A.safetensors", "alpha", 0.8),))
        progress = _Progress()
        calls = []
        apply_calls = []

        def apply(model, clip, state, weight, metadata):
            apply_calls.append((state, weight))
            return model, _Clip(clip.stack + ((state, weight),))

        def sample(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise, **kwargs):
            calls.append((model, seed, positive, negative, latent))
            return latent

        values = {
            "positive_prompt_1": "portrait",
            "positive_prompt_2": "landscape",
        }
        with (
            patch("lora_tester.nodes._resolve_lora_path", return_value=str(ROOT / "A.safetensors")),
            patch("lora_tester.nodes._load_lora_file", return_value=("state:A", {"source": "A"})),
            patch("lora_tester.nodes._apply_lora_to_models", side_effect=apply),
            patch("lora_tester.nodes._common_ksampler", side_effect=sample),
            patch("lora_tester.nodes._decode_vae", return_value=torch.zeros((1, 8, 10, 3))),
            patch("lora_tester.nodes._make_progress_bar", return_value=progress),
            patch("lora_tester.nodes._throw_if_interrupted"),
        ):
            result = MultiPromptSampleNode().sample(
                model=(),
                clip=_Clip(),
                vae=_Vae(),
                latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                lorastacks=LoraStackList((stack,)),
                prompt_count=2,
                prompt_prefix="shared",
                negative_prompt="blur",
                seed=123,
                steps=4,
                cfg=7.0,
                sampler_name="sampler",
                scheduler="scheduler",
                denoise=1.0,
                color_mode="black",
                show_lora_details=False,
                max_canvas_megapixels=10.0,
                custom_style=None,
                unique_id="test",
                **values,
            )[0]

        self.assertEqual(len(calls), 4)
        self.assertEqual(len(apply_calls), 1)
        self.assertEqual([call[1] for call in calls], [123, 123, 123, 123])
        self.assertEqual([call[2]["text"] for call in calls], [
            "shared, portrait",
            "shared, landscape",
            "shared, alpha, portrait",
            "shared, alpha, landscape",
        ])
        self.assertEqual(progress.updates[-1], (4, 4))
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[-1], 3)
        self.assertGreater(result.shape[1], 16)
        self.assertGreater(result.shape[2], 30)

    def test_multi_prompt_sample_keeps_same_file_at_different_strengths(self):
        stacks = LoraStackList(
            (
                LoraStack((LoraStackItem("Shared.safetensors", "low", 0.5),)),
                LoraStack((LoraStackItem("Shared.safetensors", "high", 1.0),)),
            )
        )
        apply_calls = []
        sample_calls = []

        def apply(model, clip, state, weight, metadata):
            apply_calls.append((state, weight))
            return model, _Clip(clip.stack + ((state, weight),))

        def sample(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise, **kwargs):
            sample_calls.append(positive)
            return latent

        with (
            patch("lora_tester.nodes._resolve_lora_path", return_value=str(ROOT / "Shared.safetensors")),
            patch("lora_tester.nodes._load_lora_file", return_value=("state:shared", None)) as load,
            patch("lora_tester.nodes._apply_lora_to_models", side_effect=apply),
            patch("lora_tester.nodes._common_ksampler", side_effect=sample),
            patch("lora_tester.nodes._decode_vae", return_value=torch.zeros((1, 8, 10, 3))),
            patch("lora_tester.nodes._make_progress_bar", return_value=_Progress()),
            patch("lora_tester.nodes._throw_if_interrupted"),
        ):
            MultiPromptSampleNode().sample(
                model=(),
                clip=_Clip(),
                vae=_Vae(),
                latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                lorastacks=stacks,
                prompt_count=1,
                prompt_prefix="",
                negative_prompt="",
                seed=1,
                steps=1,
                cfg=1.0,
                sampler_name="sampler",
                scheduler="scheduler",
                denoise=1.0,
                color_mode="black",
                show_lora_details=True,
                max_canvas_megapixels=10.0,
                positive_prompt_1="portrait",
            )

        load.assert_called_once()
        self.assertEqual(apply_calls, [("state:shared", 0.5), ("state:shared", 1.0)])
        self.assertEqual(
            [(call["text"], call["stack"]) for call in sample_calls],
            [
                ("portrait", ()),
                ("low, portrait", (("state:shared", 0.5),)),
                ("high, portrait", (("state:shared", 1.0),)),
            ],
        )

    def test_multi_prompt_mixed_stack_skips_artist_lora_and_clears_run_cache(self):
        stack = LoraStack(
            (
                LoraStackItem(ARTIST_TAG_MODE, "@fkey", 1.2),
                LoraStackItem("A.safetensors", "alpha", 0.8),
            )
        )
        sample_calls = []

        def apply(model, clip, state, weight, metadata):
            return model, _Clip(clip.stack + ((state, weight),))

        def sample(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise, **kwargs):
            sample_calls.append(positive)
            return latent

        node = MultiPromptSampleNode()
        with (
            patch("lora_tester.nodes._resolve_lora_path", return_value=str(ROOT / "A.safetensors")),
            patch("lora_tester.nodes._load_lora_file", return_value=("state:A", None)) as load,
            patch("lora_tester.nodes._apply_lora_to_models", side_effect=apply),
            patch("lora_tester.nodes._common_ksampler", side_effect=sample),
            patch("lora_tester.nodes._decode_vae", return_value=torch.zeros((1, 8, 10, 3))),
            patch("lora_tester.nodes._make_progress_bar", return_value=_Progress()),
            patch("lora_tester.nodes._throw_if_interrupted"),
        ):
            node.sample(
                model=(),
                clip=_Clip(),
                vae=_Vae(),
                latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                lorastacks=LoraStackList((stack,)),
                prompt_count=1,
                prompt_prefix="shared",
                negative_prompt="",
                seed=1,
                steps=1,
                cfg=1.0,
                sampler_name="sampler",
                scheduler="scheduler",
                denoise=1.0,
                color_mode="black",
                show_lora_details=True,
                max_canvas_megapixels=10.0,
                positive_prompt_1="portrait",
            )

        load.assert_called_once()
        self.assertEqual(len(node._lora_cache), 0)
        self.assertEqual(
            [(call["text"], call["stack"]) for call in sample_calls],
            [
                ("shared, portrait", ()),
                ("shared, (fkey:1.2), alpha, portrait", (("state:A", 0.8),)),
            ],
        )

    def test_multi_prompt_anima_prompt_artist_stays_in_base_prompt(self):
        stack = LoraStack(
            (
                LoraStackItem(ARTIST_TAG_MODE, "@test_artist", 0.5),
                LoraStackItem(ARTIST_TAG_MODE, "@second", 1.0),
            )
        )
        pack_calls = []
        mixer_calls = []
        sample_calls = []

        class Pack:
            def pack(self, **kwargs):
                pack_calls.append(kwargs)
                return ({"chain": kwargs["artist_chain"]},)

        class Mixer:
            def patch(self, **kwargs):
                mixer_calls.append(kwargs)
                return (
                    ("mixed", kwargs["artist_pack"]["chain"]),
                    {"text": kwargs["artist_pack"]["chain"], "stack": ()},
                )

        model = SimpleNamespace(
            model=SimpleNamespace(
                model_config=SimpleNamespace(unet_config={"image_model": "anima"})
            )
        )

        def sample(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise, **kwargs):
            sample_calls.append((model, positive))
            return latent

        with (
            patch(
                "lora_tester.artist._resolve_anima_mixer_nodes",
                return_value=(Pack, Mixer),
            ),
            patch("lora_tester.nodes._common_ksampler", side_effect=sample),
            patch("lora_tester.nodes._decode_vae", return_value=torch.zeros((1, 8, 10, 3))),
            patch("lora_tester.nodes._make_progress_bar", return_value=_Progress()),
            patch("lora_tester.nodes._throw_if_interrupted"),
        ):
            MultiPromptSampleNode().sample(
                model=model,
                clip=_Clip(),
                vae=_Vae(),
                latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                lorastacks=LoraStackList((stack,)),
                prompt_count=1,
                prompt_prefix="masterpiece",
                negative_prompt="",
                seed=1,
                steps=1,
                cfg=1.0,
                sampler_name="sampler",
                scheduler="scheduler",
                denoise=1.0,
                color_mode="black",
                show_lora_details=True,
                max_canvas_megapixels=10.0,
                positive_prompt_1="@prompt_artist, portrait",
            )

        self.assertEqual(len(pack_calls), 1)
        self.assertEqual(
            pack_calls[0]["artist_chain"],
            "(@test_artist:0.5)\n@second",
        )
        self.assertEqual(
            pack_calls[0]["base_prompt"], "masterpiece, @prompt_artist, portrait"
        )
        self.assertEqual(len(mixer_calls), 1)
        self.assertIs(sample_calls[0][0], model)
        self.assertEqual(sample_calls[0][1]["text"], "masterpiece, @prompt_artist, portrait")
        self.assertEqual(
            sample_calls[1][0],
            ("mixed", "(@test_artist:0.5)\n@second"),
        )

    def test_multi_prompt_at_lora_trigger_stays_outside_artist_chain(self):
        stack = LoraStack(
            (
                LoraStackItem("A.safetensors", "@lora_trigger", 0.8),
                LoraStackItem(ARTIST_TAG_MODE, "@artist_one", 1.0),
                LoraStackItem(ARTIST_TAG_MODE, "@artist_two", 1.0),
            )
        )
        pack_calls = []

        class Pack:
            def pack(self, **kwargs):
                pack_calls.append(kwargs)
                return ({"chain": kwargs["artist_chain"]},)

        class Mixer:
            def patch(self, **kwargs):
                return (
                    kwargs["model"],
                    {"text": kwargs["artist_pack"]["chain"], "stack": ()},
                )

        model = SimpleNamespace(
            model=SimpleNamespace(
                model_config=SimpleNamespace(unet_config={"image_model": "anima"})
            )
        )

        with (
            patch("lora_tester.artist._resolve_anima_mixer_nodes", return_value=(Pack, Mixer)),
            patch("lora_tester.nodes._resolve_lora_path", return_value=str(ROOT / "A.safetensors")),
            patch("lora_tester.nodes._load_lora_file", return_value=("state:A", None)),
            patch("lora_tester.nodes._apply_lora_to_models", side_effect=lambda model, clip, state, weight, metadata: (model, clip)),
            patch("lora_tester.nodes._common_ksampler", side_effect=lambda *args, **kwargs: args[8]),
            patch("lora_tester.nodes._decode_vae", return_value=torch.zeros((1, 8, 10, 3))),
            patch("lora_tester.nodes._make_progress_bar", return_value=_Progress()),
            patch("lora_tester.nodes._throw_if_interrupted"),
        ):
            MultiPromptSampleNode().sample(
                model=model,
                clip=_Clip(),
                vae=_Vae(),
                latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                lorastacks=LoraStackList((stack,)),
                prompt_count=1,
                prompt_prefix="",
                negative_prompt="",
                seed=1,
                steps=1,
                cfg=1.0,
                sampler_name="sampler",
                scheduler="scheduler",
                denoise=1.0,
                color_mode="black",
                show_lora_details=False,
                log_test_details=False,
                max_canvas_megapixels=10.0,
                positive_prompt_1="portrait",
            )

        self.assertEqual(pack_calls[0]["artist_chain"], "@artist_one\n@artist_two")
        self.assertEqual(pack_calls[0]["base_prompt"], "@lora_trigger, portrait")

    def test_multi_prompt_disabled_mixer_uses_shared_native_routing(self):
        stack = LoraStack(
            (
                LoraStackItem(ARTIST_TAG_MODE, "@artist_one", 1.0),
                LoraStackItem(ARTIST_TAG_MODE, "@artist_two", 0.5),
            )
        )
        sample_calls = []

        def sample(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise, **kwargs):
            sample_calls.append(positive)
            return latent

        model = SimpleNamespace(
            model=SimpleNamespace(
                model_config=SimpleNamespace(unet_config={"image_model": "anima"})
            )
        )
        with (
            patch("lora_tester.nodes.anima_artist_mixer_available", return_value=True),
            patch(
                "lora_tester.artist._resolve_anima_mixer_nodes",
                side_effect=AssertionError("Disabled Mixer must not be queried"),
            ),
            patch("lora_tester.nodes._common_ksampler", side_effect=sample),
            patch("lora_tester.nodes._decode_vae", return_value=torch.zeros((1, 8, 10, 3))),
            patch("lora_tester.nodes._make_progress_bar", return_value=_Progress()),
            patch("lora_tester.nodes._throw_if_interrupted"),
        ):
            MultiPromptSampleNode().sample(
                model=model,
                clip=_Clip(),
                vae=_Vae(),
                latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                lorastacks=LoraStackList((stack,)),
                prompt_count=1,
                prompt_prefix="masterpiece",
                negative_prompt="",
                seed=1,
                steps=1,
                cfg=1.0,
                sampler_name="sampler",
                scheduler="scheduler",
                denoise=1.0,
                color_mode="black",
                show_lora_details=False,
                log_test_details=False,
                use_anima_artist_mixer=False,
                max_canvas_megapixels=10.0,
                positive_prompt_1="portrait",
            )

        self.assertEqual(sample_calls[0]["text"], "masterpiece, portrait")
        self.assertEqual(
            sample_calls[1]["text"],
            "masterpiece, @artist_one, (@artist_two:0.5), portrait",
        )

    def test_multi_prompt_interrupt_clears_run_local_lora_cache(self):
        stack = LoraStack((LoraStackItem("A.safetensors", "alpha", 0.8),))
        node = MultiPromptSampleNode()

        with (
            patch("lora_tester.nodes._resolve_lora_path", return_value=str(ROOT / "A.safetensors")),
            patch("lora_tester.nodes._load_lora_file", return_value=("state:A", None)) as load,
            patch(
                "lora_tester.nodes._apply_lora_to_models",
                side_effect=lambda model, clip, state, weight, metadata: (model, clip),
            ),
            patch("lora_tester.nodes._common_ksampler", side_effect=lambda *args, **kwargs: args[8]),
            patch("lora_tester.nodes._decode_vae", return_value=torch.zeros((1, 8, 10, 3))),
            patch("lora_tester.nodes._make_progress_bar", return_value=_Progress()),
            patch(
                "lora_tester.nodes._throw_if_interrupted",
                side_effect=[None, RuntimeError("interrupted")],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                node.sample(
                    model=(),
                    clip=_Clip(),
                    vae=_Vae(),
                    latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                    lorastacks=LoraStackList((stack,)),
                    prompt_count=1,
                    prompt_prefix="",
                    negative_prompt="",
                    seed=1,
                    steps=1,
                    cfg=1.0,
                    sampler_name="sampler",
                    scheduler="scheduler",
                    denoise=1.0,
                    color_mode="black",
                    show_lora_details=True,
                    max_canvas_megapixels=10.0,
                    positive_prompt_1="portrait",
                )

        load.assert_called_once()
        self.assertEqual(len(node._lora_cache), 0)

    def test_multi_prompt_log_lists_preflight_weights_and_column_reuse(self):
        stack = LoraStack((LoraStackItem("A.safetensors", "alpha", 0.8),))

        with (
            patch("lora_tester.nodes._resolve_lora_path", return_value=str(ROOT / "A.safetensors")),
            patch("lora_tester.nodes._load_lora_file", return_value=("state:A", None)),
            patch("lora_tester.nodes._apply_lora_to_models", side_effect=lambda model, clip, state, weight, metadata: (model, clip)),
            patch("lora_tester.nodes._common_ksampler", side_effect=lambda *args, **kwargs: args[8]),
            patch("lora_tester.nodes._decode_vae", return_value=torch.zeros((1, 8, 10, 3))),
            patch("lora_tester.nodes._make_progress_bar", return_value=_Progress()),
            patch("lora_tester.nodes._throw_if_interrupted"),
        ):
            with self.assertLogs("lora_tester.nodes", level="INFO") as captured:
                MultiPromptSampleNode().sample(
                    model=(),
                    clip=_Clip(),
                    vae=_Vae(),
                    latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                    lorastacks=LoraStackList((stack,)),
                    prompt_count=2,
                    prompt_prefix="shared",
                    negative_prompt="",
                    seed=1,
                    steps=1,
                    cfg=1.0,
                    sampler_name="sampler",
                    scheduler="scheduler",
                    denoise=1.0,
                    color_mode="black",
                    show_lora_details=False,
                    log_test_details=True,
                    max_canvas_megapixels=10.0,
                    positive_prompt_1="portrait",
                    positive_prompt_2="landscape",
                )

        output = "\n".join(captured.output)
        self.assertIn("Combination test preflight", output)
        self.assertIn("Model family: danbooru", output)
        self.assertIn("Anima Artist Mixer switch: on", output)
        self.assertIn('"A.safetensors" | weight=0.8', output)
        self.assertIn("cache=run-local:miss", output)
        self.assertIn("cache=run-local:hit", output)
        self.assertIn("Patched model: column reuse", output)
        self.assertIn("Combination image", output)


if __name__ == "__main__":
    unittest.main()
