from __future__ import annotations

import sys
import unittest
from pathlib import Path
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
from lora_tester.stack import LoraStack, LoraStackItem, LoraStackList
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
        self.assertEqual(len(apply_calls), 2)
        self.assertEqual([call[1] for call in calls], [123, 123, 123, 123])
        self.assertEqual([call[2]["text"] for call in calls], [
            "shared, portrait",
            "shared, alpha, portrait",
            "shared, landscape",
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


if __name__ == "__main__":
    unittest.main()
