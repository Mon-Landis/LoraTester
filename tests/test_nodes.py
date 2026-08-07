from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from lora_tester.compositor import CompositionSession
from lora_tester.layout import LoraSpec, build_layout
from lora_tester.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    LoraTesterSampler,
    LoraTesterStyleNode,
    compose_positive_prompt,
)
from lora_tester.styles import StyleConfig


class FakeClip:
    def __init__(self, stack=()):
        self.stack = tuple(stack)

    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return {"text": tokens, "stack": self.stack}


class FakeVAE:
    def __init__(self, events):
        self.events = events
        self.decode_count = 0

    def decode(self, latent):
        self.decode_count += 1
        self.events.append("decode")
        value = (self.decode_count % 10) / 10.0
        return torch.full((1, 8, 10, 3), value, dtype=torch.float32)


class FakeProgress:
    def __init__(self):
        self.updates = []

    def update_absolute(self, value, total=None, preview=None):
        self.updates.append((value, total))


class NodeTests(unittest.TestCase):
    def _sample_arguments(self, count):
        return {
            "model": (),
            "clip": FakeClip(),
            "vae": None,
            "latent_image": {"samples": torch.zeros((1, 4, 2, 2), dtype=torch.float32)},
            "positive_prompt": "portrait",
            "negative_prompt": "blur",
            "seed": 123456,
            "steps": 12,
            "cfg": 6.5,
            "sampler_name": "fake_sampler",
            "scheduler": "fake_scheduler",
            "denoise": 0.8,
            "lora_count": count,
            "lora_a_name": "A.safetensors",
            "lora_a_trigger": "alpha",
            "lora_a_max_strength": 0.8,
            "lora_b_name": "B.safetensors",
            "lora_b_trigger": "beta",
            "lora_b_max_strength": 1.0,
            "lora_c_name": "C.safetensors",
            "lora_c_trigger": "charlie",
            "lora_c_max_strength": 2.0,
            "color_mode": "black",
            "show_lora_details": True,
            "max_canvas_megapixels": 10.0,
            "unique_id": "test-node",
        }

    def _run_fake_sample(self, count):
        events = []
        sampler_calls = []
        load_calls = []
        apply_calls = []
        progress = FakeProgress()
        arguments = self._sample_arguments(count)
        arguments["vae"] = FakeVAE(events)

        def resolve(name):
            return str(ROOT / "fake_loras" / name)

        def load(path):
            load_calls.append(path)
            return f"state:{Path(path).name}", {"path": path}

        def apply(model, clip, state_dict, weight, metadata):
            apply_calls.append((model, clip.stack, state_dict, weight, metadata))
            patch_value = (state_dict, weight)
            return model + (patch_value,), FakeClip(clip.stack + (patch_value,))

        def sample(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise,
        ):
            events.append("sample")
            sampler_calls.append(
                {
                    "model": model,
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "positive": positive,
                    "negative": negative,
                    "latent": latent,
                    "denoise": denoise,
                }
            )
            return {"samples": latent["samples"]}

        original_submit = CompositionSession.submit

        def submit(session, image, **kwargs):
            events.append("submit")
            return original_submit(session, image, **kwargs)

        node = LoraTesterSampler()
        with (
            patch("lora_tester.nodes._resolve_lora_path", side_effect=resolve),
            patch("lora_tester.nodes._load_lora_file", side_effect=load),
            patch("lora_tester.nodes._apply_lora_to_models", side_effect=apply),
            patch("lora_tester.nodes._common_ksampler", side_effect=sample),
            patch("lora_tester.nodes._throw_if_interrupted"),
            patch("lora_tester.nodes._make_progress_bar", return_value=progress),
            patch.object(CompositionSession, "submit", new=submit),
        ):
            result = node.sample(**arguments)[0]

        return {
            "arguments": arguments,
            "events": events,
            "sampler_calls": sampler_calls,
            "load_calls": load_calls,
            "apply_calls": apply_calls,
            "progress": progress,
            "result": result,
        }

    def test_sampler_call_counts_and_streaming_order(self):
        for count, expected_count in ((1, 5), (2, 25), (3, 69)):
            with self.subTest(lora_count=count):
                run = self._run_fake_sample(count)
                self.assertEqual(len(run["sampler_calls"]), expected_count)
                self.assertEqual(len(run["load_calls"]), count)
                self.assertEqual(
                    run["events"],
                    [event for _ in range(expected_count) for event in ("sample", "decode", "submit")],
                )
                self.assertEqual(
                    run["progress"].updates,
                    [(index, expected_count) for index in range(1, expected_count + 1)],
                )
                self.assertEqual(run["result"].shape[0], 1)
                self.assertEqual(run["result"].shape[-1], 3)

    def test_every_task_reuses_seed_latent_and_base_model(self):
        run = self._run_fake_sample(3)
        arguments = run["arguments"]
        specs = (
            LoraSpec("A.safetensors", 0.8, "alpha"),
            LoraSpec("B.safetensors", 1.0, "beta"),
            LoraSpec("C.safetensors", 2.0, "charlie"),
        )
        plan = build_layout(specs)

        for task, call in zip(plan.tasks, run["sampler_calls"]):
            expected_stack = tuple(
                (f"state:{spec.name}", weight)
                for spec, weight in zip(specs, task.weights)
                if weight != 0.0
            )
            self.assertEqual(call["model"], expected_stack)
            self.assertEqual(call["positive"]["stack"], expected_stack)
            self.assertEqual(call["negative"]["stack"], expected_stack)
            self.assertEqual(call["seed"], 123456)
            self.assertIs(call["latent"], arguments["latent_image"])

    def test_trigger_words_are_prepended_in_slot_order(self):
        run = self._run_fake_sample(3)
        plan = build_layout(
            (
                LoraSpec("A.safetensors", 0.8, "alpha"),
                LoraSpec("B.safetensors", 1.0, "beta"),
                LoraSpec("C.safetensors", 2.0, "charlie"),
            )
        )
        calls = {
            task.task_id: call for task, call in zip(plan.tasks, run["sampler_calls"])
        }
        self.assertEqual(calls["base"]["positive"]["text"], "portrait")
        self.assertEqual(calls["A025+C050"]["positive"]["text"], "alpha, charlie, portrait")
        self.assertEqual(
            calls["A100+B100+C100"]["positive"]["text"],
            "alpha, beta, charlie, portrait",
        )
        self.assertTrue(all(call["negative"]["text"] == "blur" for call in calls.values()))

    def test_prompt_join_omits_empty_values(self):
        self.assertEqual(compose_positive_prompt(" base ", ("alpha", "", " charlie ")), "alpha, charlie, base")
        self.assertEqual(compose_positive_prompt("", ()), "")

    def test_rejects_latent_batches_before_loading_or_sampling(self):
        arguments = self._sample_arguments(1)
        arguments["latent_image"] = {"samples": torch.zeros((2, 4, 2, 2))}
        node = LoraTesterSampler()
        with self.assertRaisesRegex(ValueError, "batch size of 1"):
            node.sample(**arguments)

    def test_lora_cache_is_lru_bounded_to_three_files(self):
        node = LoraTesterSampler()
        loads = []

        def load(path):
            loads.append(path)
            return path, None

        with (
            patch("lora_tester.nodes._resolve_lora_path", side_effect=lambda name: str(ROOT / name)),
            patch("lora_tester.nodes._load_lora_file", side_effect=load),
        ):
            for name in ("A", "B", "C", "D"):
                node._load_lora(name)
            self.assertEqual(len(node._lora_cache), 3)
            node._load_lora("B")
            self.assertEqual(len(loads), 4)
            node._load_lora("A")
            self.assertEqual(len(loads), 5)

    def test_main_input_contract_contains_display_toggle_and_custom_style(self):
        with (
            patch("lora_tester.nodes._get_lora_names", return_value=["A.safetensors"]),
            patch("lora_tester.nodes._get_sampler_names", return_value=("sampler",)),
            patch("lora_tester.nodes._get_scheduler_names", return_value=("scheduler",)),
        ):
            inputs = LoraTesterSampler.INPUT_TYPES()
        self.assertIn("show_lora_details", inputs["required"])
        self.assertTrue(inputs["required"]["show_lora_details"][1]["default"])
        self.assertEqual(inputs["required"]["lora_count"][1]["default"], 1)
        self.assertEqual(inputs["required"]["color_mode"][0], ["black", "white", "custom"])
        self.assertEqual(inputs["optional"]["custom_style"][0], "LORA_TESTER_STYLE")

    def test_style_node_builds_custom_style_and_rejects_background_batches(self):
        required = LoraTesterStyleNode.INPUT_TYPES()["required"]
        values = {}
        for name, specification in required.items():
            input_type, options = specification
            if isinstance(input_type, (list, tuple)):
                values[name] = options.get("default", input_type[0])
            else:
                values[name] = options["default"]
        values["background_color"] = "#123456"
        values["region_gap"] = 72

        style = LoraTesterStyleNode().build_style(**values)[0]
        self.assertIsInstance(style, StyleConfig)
        self.assertEqual(style.mode, "custom")
        self.assertEqual(style.background_color, (0x12, 0x34, 0x56))
        self.assertEqual(style.region_gap, 72)
        self.assertIsNone(style.font_size)

        with self.assertRaisesRegex(ValueError, "one background image"):
            LoraTesterStyleNode().build_style(
                **values,
                background_image=torch.zeros((2, 4, 4, 3)),
            )

    def test_nodes_are_registered(self):
        self.assertIs(NODE_CLASS_MAPPINGS["LoraTesterSampler"], LoraTesterSampler)
        self.assertIs(NODE_CLASS_MAPPINGS["LoraTesterStyle"], LoraTesterStyleNode)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["LoraTesterSampler"],
            "LoRA Tester (KSampler)",
        )


if __name__ == "__main__":
    unittest.main()
