from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from lora_tester.compositor import CompositionSession
from lora_tester.artist import ARTIST_TAG_MODE
from lora_tester.layout import LoraSpec, build_layout
from lora_tester.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    AnimaArtistMixerConfigNode,
    ArtistTagTemplateNode,
    LoraTesterSampler,
    LoraTesterStyleNode,
    _common_ksampler,
    _sampling_progress_value,
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
            "independent_artist_tags": "",
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
            "lora_a_min_strength": 0.0,
            "lora_a_max_strength": 0.8,
            "lora_b_name": "B.safetensors",
            "lora_b_trigger": "beta",
            "lora_b_min_strength": 0.0,
            "lora_b_max_strength": 1.0,
            "lora_c_name": "C.safetensors",
            "lora_c_trigger": "charlie",
            "lora_c_min_strength": 0.0,
            "lora_c_max_strength": 2.0,
            "color_mode": "black",
            "show_lora_details": True,
            "max_canvas_megapixels": 10.0,
            "use_anima_artist_mixer": True,
            "unique_id": "test-node",
        }

    def _run_fake_sample(self, count, overrides=None):
        events = []
        sampler_calls = []
        load_calls = []
        apply_calls = []
        progress = FakeProgress()
        arguments = self._sample_arguments(count)
        arguments.update(overrides or {})
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
            *,
            progress,
            completed_tasks,
            total_tasks,
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
                    "progress": progress,
                    "completed_tasks": completed_tasks,
                    "total_tasks": total_tasks,
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
            "node": node,
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
                self.assertTrue(
                    all(
                        call["completed_tasks"] == index
                        and call["total_tasks"] == expected_count
                        and call["progress"] is run["progress"]
                        for index, call in enumerate(run["sampler_calls"])
                    )
                )

    def test_sampling_progress_reserves_postprocess_boundary(self):
        self.assertAlmostEqual(_sampling_progress_value(0, 0, 20), 1 / 21)
        self.assertAlmostEqual(_sampling_progress_value(0, 19, 20), 20 / 21)
        self.assertAlmostEqual(_sampling_progress_value(7, 19, 20), 7 + 20 / 21)
        self.assertLess(_sampling_progress_value(68, 19, 20), 69)

    def test_custom_sampler_reports_overall_progress_without_competing_node_progress(self):
        samples = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
        sampled = torch.ones_like(samples)
        captured = {}

        fake_sample = ModuleType("comfy.sample")
        fake_utils = ModuleType("comfy.utils")
        fake_comfy = ModuleType("comfy")
        fake_comfy.sample = fake_sample
        fake_comfy.utils = fake_utils
        fake_preview = ModuleType("latent_preview")

        def fix_empty(_model, latent, _spatial, _temporal):
            return latent

        def prepare_noise(latent, seed, batch_indices):
            captured["noise"] = (latent, seed, batch_indices)
            return torch.zeros_like(latent)

        def sample_impl(*args, **kwargs):
            captured["sample_args"] = args
            captured["sample_kwargs"] = kwargs
            return sampled

        fake_sample.fix_empty_latent_channels = fix_empty
        fake_sample.prepare_noise = prepare_noise
        fake_sample.sample = sample_impl
        fake_utils.PROGRESS_BAR_ENABLED = True
        fake_preview.get_previewer = lambda _device, _format: None

        model = type(
            "FakeModel",
            (),
            {
                "load_device": "cpu",
                "model": type("InnerModel", (), {"latent_format": object()})(),
            },
        )()
        progress = FakeProgress()
        latent = {
            "samples": samples,
            "batch_index": [3],
            "noise_mask": "mask",
            "downscale_ratio_spacial": 8,
            "downscale_ratio_temporal": 4,
        }

        with patch.dict(
            sys.modules,
            {
                "comfy": fake_comfy,
                "comfy.sample": fake_sample,
                "comfy.utils": fake_utils,
                "latent_preview": fake_preview,
            },
        ):
            result = _common_ksampler(
                model,
                123,
                20,
                7.0,
                "sampler",
                "scheduler",
                "positive",
                "negative",
                latent,
                0.8,
                progress=progress,
                completed_tasks=2,
                total_tasks=5,
            )

        kwargs = captured["sample_kwargs"]
        self.assertFalse(kwargs["disable_pbar"])
        self.assertEqual(kwargs["noise_mask"], "mask")
        self.assertEqual(kwargs["seed"], 123)
        kwargs["callback"](19, torch.zeros_like(samples), samples, 20)
        self.assertEqual(progress.updates[-1][1], 5)
        self.assertAlmostEqual(progress.updates[-1][0], 2 + 20 / 21)
        self.assertIs(result["samples"], sampled)
        self.assertNotIn("downscale_ratio_spacial", result)
        self.assertNotIn("downscale_ratio_temporal", result)

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

    def test_artist_mode_skips_lora_loading_and_uses_tag_weight(self):
        run = self._run_fake_sample(
            2,
            {
                "lora_a_name": ARTIST_TAG_MODE,
                "lora_a_trigger": "@fkey",
                "lora_a_min_strength": 0.0,
                "lora_a_max_strength": 0.8,
            },
        )
        self.assertEqual([Path(path).name for path in run["load_calls"]], ["B.safetensors"])
        self.assertTrue(
            all(call[2] == "state:B.safetensors" for call in run["apply_calls"])
        )
        plan = build_layout(
            (
                LoraSpec(ARTIST_TAG_MODE, 0.8, "@fkey"),
                LoraSpec("B.safetensors", 1.0, "beta"),
            )
        )
        calls = {
            task.task_id: call for task, call in zip(plan.tasks, run["sampler_calls"])
        }
        self.assertEqual(calls["base"]["positive"]["text"], "portrait")
        self.assertEqual(calls["A025"]["positive"]["text"], "(fkey:0.2), portrait")
        self.assertEqual(
            calls["A100+B100"]["positive"]["text"],
            "(fkey:0.8), beta, portrait",
        )
        self.assertEqual(len(run["node"]._lora_cache), 0)

    def test_low_weight_artist_lora_cell_keeps_independent_lora_and_trigger(self):
        """A low mixed cell is still a real LoRA+artist conditioning sample."""
        run = self._run_fake_sample(
            2,
            {
                "lora_a_name": ARTIST_TAG_MODE,
                "lora_a_trigger": "@fkey",
                "lora_a_max_strength": 0.8,
                "lora_b_name": "B.safetensors",
                "lora_b_trigger": "beta",
                "lora_b_max_strength": 1.0,
            },
        )
        calls = {
            task.task_id: call
            for task, call in zip(
                build_layout(
                    (
                        LoraSpec(ARTIST_TAG_MODE, 0.8, "@fkey"),
                        LoraSpec("B.safetensors", 1.0, "beta"),
                    )
                ).tasks,
                run["sampler_calls"],
            )
        }
        low_mix = calls["A025+B025"]
        self.assertEqual(low_mix["model"], (("state:B.safetensors", 0.25),))
        self.assertEqual(
            low_mix["positive"]["stack"],
            (("state:B.safetensors", 0.25),),
        )
        self.assertEqual(
            low_mix["positive"]["text"],
            "(fkey:0.2), beta, portrait",
        )

    def test_anima_independent_artist_joins_test_artist_in_external_mixer(self):
        pack_calls = []
        mixer_calls = []

        class Pack:
            def pack(self, **kwargs):
                pack_calls.append(kwargs)
                return ({"artist_chain": kwargs["artist_chain"]},)

        class Mixer:
            def patch(self, **kwargs):
                mixer_calls.append(kwargs)
                return (
                    ("mixed", kwargs["artist_pack"]["artist_chain"]),
                    {"text": kwargs["artist_pack"]["artist_chain"], "stack": ()},
                )

        class AnimaModel:
            model = SimpleNamespace(
                model_config=SimpleNamespace(unet_config={"image_model": "anima"})
            )

            def __add__(self, _value):
                return self

        anima_model = AnimaModel()
        with patch(
            "lora_tester.artist._resolve_anima_mixer_nodes",
            return_value=(Pack, Mixer),
        ):
            run = self._run_fake_sample(
                1,
                {
                    "model": anima_model,
                    "positive_prompt": "@prompt_artist, portrait",
                    "independent_artist_tags": "@independent_artist",
                    "lora_a_name": ARTIST_TAG_MODE,
                    "lora_a_trigger": "@test_artist",
                    "lora_a_min_strength": 0.0,
                    "lora_a_max_strength": 1.0,
                },
            )

        self.assertEqual(len(pack_calls), 4)
        self.assertEqual(
            pack_calls[0]["artist_chain"],
            "(@test_artist:0.25)\n@independent_artist",
        )
        self.assertEqual(
            pack_calls[0]["base_prompt"], "@prompt_artist, portrait"
        )
        self.assertEqual(mixer_calls[0]["strength"], 1.6)
        self.assertEqual(
            run["sampler_calls"][1]["model"],
            ("mixed", "(@test_artist:0.25)\n@independent_artist"),
        )

    def test_direct_template_and_mixer_config_inputs_reach_external_nodes(self):
        pack_calls = []
        mixer_calls = []

        class Pack:
            def pack(self, **kwargs):
                pack_calls.append(kwargs)
                return (kwargs,)

        class Mixer:
            def patch(self, **kwargs):
                mixer_calls.append(kwargs)
                return kwargs["model"], {"text": "mixed"}

        template = ArtistTagTemplateNode.build_template(
            "artist::{tag}",
            "artist::{tag}::{weight:.2f}",
        )[0]
        config = AnimaArtistMixerConfigNode.build_config(
            2.0,
            False,
            "shared_base_ids",
            True,
            True,
            0.25,
        )[0]
        anima_model = SimpleNamespace(
            model=SimpleNamespace(
                model_config=SimpleNamespace(unet_config={"image_model": "anima"})
            )
        )
        with patch(
            "lora_tester.artist._resolve_anima_mixer_nodes",
            return_value=(Pack, Mixer),
        ):
            self._run_fake_sample(
                1,
                {
                    "model": anima_model,
                    "log_test_details": False,
                    "independent_artist_tags": "@independent",
                    "lora_a_name": ARTIST_TAG_MODE,
                    "lora_a_trigger": "@test_artist",
                    "artist_tag_template": template,
                    "anima_mixer_config": config,
                },
            )

        self.assertTrue(pack_calls)
        self.assertEqual(
            pack_calls[0]["artist_chain"],
            "artist::test_artist::0.20\nartist::independent",
        )
        self.assertEqual(mixer_calls[0]["strength"], 2.0)
        self.assertFalse(mixer_calls[0]["normalize_weights"])
        self.assertEqual(mixer_calls[0]["alignment_mode"], "shared_base_ids")
        self.assertTrue(mixer_calls[0]["apply_to_uncond"])
        self.assertEqual(mixer_calls[0]["uncond_strength"], 0.25)

    def test_at_lora_trigger_is_never_added_to_direct_artist_chain(self):
        pack_calls = []

        class Pack:
            def pack(self, **kwargs):
                pack_calls.append(kwargs)
                return ({"artist_chain": kwargs["artist_chain"]},)

        class Mixer:
            def patch(self, **kwargs):
                return (
                    kwargs["model"],
                    {"text": kwargs["artist_pack"]["artist_chain"], "stack": ()},
                )

        class AnimaModel(tuple):
            def __new__(cls, values=()):
                return tuple.__new__(cls, values)

            def __init__(self, values=()):
                self.model = SimpleNamespace(
                    model_config=SimpleNamespace(unet_config={"image_model": "anima"})
                )

            def __add__(self, value):
                return AnimaModel(tuple(self) + tuple(value))

        anima_model = AnimaModel()
        with patch(
            "lora_tester.artist._resolve_anima_mixer_nodes",
            return_value=(Pack, Mixer),
        ):
            self._run_fake_sample(
                3,
                {
                    "model": anima_model,
                    "lora_a_name": "A.safetensors",
                    "lora_a_trigger": "@lora_trigger",
                    "lora_b_name": ARTIST_TAG_MODE,
                    "lora_b_trigger": "@artist_one",
                    "lora_c_name": ARTIST_TAG_MODE,
                    "lora_c_trigger": "@artist_two",
                },
            )

        self.assertTrue(pack_calls)
        self.assertTrue(
            all("@lora_trigger" not in call["artist_chain"] for call in pack_calls)
        )
        self.assertTrue(
            any("@lora_trigger" in call["base_prompt"] for call in pack_calls)
        )

    def test_two_axis_single_artist_and_at_lora_never_use_mixer(self):
        class AnimaModel(tuple):
            def __new__(cls, values=()):
                return tuple.__new__(cls, values)

            def __init__(self, values=()):
                self.model = SimpleNamespace(
                    model_config=SimpleNamespace(unet_config={"image_model": "anima"})
                )

            def __add__(self, value):
                return AnimaModel(tuple(self) + tuple(value))

        with patch(
            "lora_tester.artist._resolve_anima_mixer_nodes",
            side_effect=AssertionError("A zero/single-artist cell must not use the mixer"),
        ):
            run = self._run_fake_sample(
                2,
                {
                    "model": AnimaModel(),
                    "log_test_details": False,
                    "lora_a_name": "A.safetensors",
                    "lora_a_trigger": "@lora_trigger",
                    "lora_b_name": ARTIST_TAG_MODE,
                    "lora_b_trigger": "@artist_one",
                },
            )

        encoded_prompts = {
            call["positive"]["text"] for call in run["sampler_calls"]
        }
        self.assertIn("@lora_trigger, portrait", encoded_prompts)
        self.assertIn("@artist_one, portrait", encoded_prompts)
        self.assertIn("@lora_trigger, @artist_one, portrait", encoded_prompts)

    def test_direct_sampler_mixer_matrix_for_extra_artist_tags(self):
        class AnimaModel(tuple):
            def __new__(cls, values=()):
                return tuple.__new__(cls, values)

            def __init__(self, values=()):
                self.model = SimpleNamespace(
                    model_config=SimpleNamespace(unet_config={"image_model": "anima"})
                )

            def __add__(self, value):
                return AnimaModel(tuple(self) + tuple(value))

        mixer_calls = []

        class Pack:
            def pack(self, **kwargs):
                return (kwargs,)

        class Mixer:
            def patch(self, **kwargs):
                mixer_calls.append(kwargs)
                return kwargs["model"], {"text": "mixed"}

        cases = (
            ("0 extra / artist+LoRA", ("", True), 0),
            ("0 extra / 2 LoRA", ("", False), 0),
            ("1 extra / artist+LoRA", ("@extra", True), 20),
            ("1 extra / 2 LoRA", ("@extra", False), 0),
            ("2 extra / artist+LoRA", ("@extra_one, @extra_two", True), 25),
            ("2 extra / 2 LoRA", ("@extra_one, @extra_two", False), 25),
        )
        observed = []
        for label, (independent, has_artist_entry), expected in cases:
            mixer_calls.clear()
            overrides = {
                "model": AnimaModel(),
                "log_test_details": False,
                "independent_artist_tags": independent,
                "lora_a_name": ARTIST_TAG_MODE if has_artist_entry else "A.safetensors",
                "lora_a_trigger": "@artist_one" if has_artist_entry else "alpha",
                "lora_b_name": "B.safetensors",
                "lora_b_trigger": "beta",
            }
            with patch(
                "lora_tester.artist._resolve_anima_mixer_nodes",
                return_value=(Pack, Mixer),
            ):
                self._run_fake_sample(2, overrides)
            actual = len(mixer_calls)
            observed.append((label, actual))
            self.assertEqual(actual, expected, label)
        self.assertEqual(
            observed,
            [
                ("0 extra / artist+LoRA", 0),
                ("0 extra / 2 LoRA", 0),
                ("1 extra / artist+LoRA", 20),
                ("1 extra / 2 LoRA", 0),
                ("2 extra / artist+LoRA", 25),
                ("2 extra / 2 LoRA", 25),
            ],
        )

    def test_min_strengths_reach_tasks_and_activate_center_triggers(self):
        arguments = self._sample_arguments(3)
        arguments.update(
            lora_a_min_strength=0.2,
            lora_b_min_strength=0.3,
            lora_c_min_strength=0.4,
        )
        events = []
        sampler_calls = []
        progress = FakeProgress()
        arguments["vae"] = FakeVAE(events)

        def apply(model, clip, state_dict, weight, metadata):
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
            *,
            progress,
            completed_tasks,
            total_tasks,
        ):
            sampler_calls.append({"model": model, "positive": positive})
            return {"samples": latent["samples"]}

        with (
            patch("lora_tester.nodes._resolve_lora_path", side_effect=lambda name: str(ROOT / name)),
            patch("lora_tester.nodes._load_lora_file", side_effect=lambda path: (f"state:{Path(path).name}", None)),
            patch("lora_tester.nodes._apply_lora_to_models", side_effect=apply),
            patch("lora_tester.nodes._common_ksampler", side_effect=sample),
            patch("lora_tester.nodes._throw_if_interrupted"),
            patch("lora_tester.nodes._make_progress_bar", return_value=progress),
        ):
            LoraTesterSampler().sample(**arguments)

        specs = (
            LoraSpec("A.safetensors", 0.8, "alpha", 0.2),
            LoraSpec("B.safetensors", 1.0, "beta", 0.3),
            LoraSpec("C.safetensors", 2.0, "charlie", 0.4),
        )
        plan = build_layout(specs)
        base_call = sampler_calls[0]
        self.assertEqual(
            base_call["model"],
            (
                ("state:A.safetensors", 0.2),
                ("state:B.safetensors", 0.3),
                ("state:C.safetensors", 0.4),
            ),
        )
        self.assertEqual(base_call["positive"]["text"], "alpha, beta, charlie, portrait")
        a025_call = sampler_calls[next(i for i, task in enumerate(plan.tasks) if task.task_id == "A025")]
        self.assertEqual(len(a025_call["model"]), 3)
        self.assertAlmostEqual(a025_call["model"][0][1], 0.35)
        self.assertEqual(a025_call["model"][1:], (("state:B.safetensors", 0.3), ("state:C.safetensors", 0.4)))

    def test_min_strength_must_be_lower_than_maximum(self):
        with self.assertRaisesRegex(ValueError, "min_weight must be lower"):
            LoraTesterSampler._make_specs(1, (("A.safetensors", 0.2, "", 0.2),))

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

    def test_lora_cache_reloads_when_file_is_overwritten(self):
        node = LoraTesterSampler()
        with (
            patch("lora_tester.nodes._resolve_lora_path", return_value="A.safetensors"),
            patch(
                "lora_tester.nodes._lora_file_signature",
                side_effect=[(10, 1, 1), (10, 1, 1), (10, 2, 2), (10, 2, 2)],
            ),
            patch(
                "lora_tester.nodes._load_lora_file",
                side_effect=[("state:first", None), ("state:second", None)],
            ) as load,
        ):
            first = node._load_lora("A.safetensors")
            second = node._load_lora("A.safetensors")

        self.assertEqual(first.state_dict, "state:first")
        self.assertEqual(second.state_dict, "state:second")
        self.assertEqual(load.call_count, 2)

    def test_is_changed_fingerprints_only_active_lora_files(self):
        with (
            patch(
                "lora_tester.nodes._resolve_lora_path",
                side_effect=lambda name: str(ROOT / name),
            ),
            patch(
                "lora_tester.nodes._lora_file_signature",
                side_effect=lambda path: (len(path), 20, 30),
            ),
        ):
            fingerprint = LoraTesterSampler.IS_CHANGED(
                lora_count=2,
                lora_a_name="A.safetensors",
                lora_b_name=ARTIST_TAG_MODE,
                lora_c_name="ignored.safetensors",
            )

        self.assertEqual(len(fingerprint), 1)
        self.assertTrue(fingerprint[0][0].lower().endswith("a.safetensors"))
        self.assertEqual(fingerprint[0][1], (len(str(ROOT / "A.safetensors")), 20, 30))

    def test_main_input_contract_contains_display_toggle_and_custom_style(self):
        with (
            patch("lora_tester.nodes._get_lora_names", return_value=["A.safetensors"]),
            patch("lora_tester.nodes._get_sampler_names", return_value=("sampler",)),
            patch("lora_tester.nodes._get_scheduler_names", return_value=("scheduler",)),
        ):
            inputs = LoraTesterSampler.INPUT_TYPES()
        self.assertIn("show_lora_details", inputs["required"])
        self.assertTrue(inputs["required"]["show_lora_details"][1]["default"])
        self.assertIn("log_test_details", inputs["required"])
        self.assertTrue(inputs["required"]["log_test_details"][1]["default"])
        self.assertTrue(inputs["required"]["log_test_details"][1]["advanced"])
        self.assertIn("use_anima_artist_mixer", inputs["required"])
        self.assertTrue(inputs["required"]["use_anima_artist_mixer"][1]["default"])
        self.assertTrue(inputs["required"]["use_anima_artist_mixer"][1]["advanced"])
        self.assertEqual(inputs["required"]["lora_count"][1]["default"], 1)
        for field in ("lora_a_min_strength", "lora_b_min_strength", "lora_c_min_strength"):
            self.assertEqual(inputs["required"][field][1]["default"], 0.0)
        self.assertEqual(inputs["required"]["color_mode"][0], ["black", "white", "custom"])
        self.assertEqual(inputs["optional"]["custom_style"][0], "LORA_TESTER_STYLE")

    def test_direct_validator_ignores_missing_hidden_lora(self):
        def resolve(name):
            if name in {"A.safetensors", "B.safetensors"}:
                return name
            raise FileNotFoundError(name)

        with patch("lora_tester.nodes._resolve_lora_path", side_effect=resolve):
            self.assertTrue(
                LoraTesterSampler.VALIDATE_INPUTS(
                    lora_count=2,
                    lora_a_name="A.safetensors",
                    lora_b_name="B.safetensors",
                    lora_c_name="missing.safetensors",
                )
            )
            self.assertTrue(
                LoraTesterSampler.VALIDATE_INPUTS(
                    lora_count=1,
                    lora_a_name="A.safetensors",
                    lora_b_name="missing.safetensors",
                    lora_c_name="missing.safetensors",
                )
            )
            self.assertIn(
                "active slot 2",
                LoraTesterSampler.VALIDATE_INPUTS(
                    lora_count=2,
                    lora_a_name="A.safetensors",
                    lora_b_name="missing.safetensors",
                    lora_c_name="missing.safetensors",
                ),
            )

    def test_direct_log_lists_weights_artists_and_observable_cache_state(self):
        with self.assertLogs("lora_tester.nodes", level="INFO") as captured:
            self._run_fake_sample(
                2,
                {
                    "lora_b_name": ARTIST_TAG_MODE,
                    "lora_b_trigger": "@painter",
                    "lora_b_max_strength": 1.25,
                    "independent_artist_tags": "(indie:0.7)",
                },
            )
        output = "\n".join(captured.output)
        self.assertIn('"A.safetensors" | weight=', output)
        self.assertIn("cache=run-local:miss", output)
        self.assertIn('"painter" | weight=', output)
        self.assertIn('"indie" | weight=0.7', output)
        self.assertIn("cache=none", output)
        self.assertIn("Route: native_prompt", output)

    def test_direct_log_toggle_suppresses_detail_records(self):
        with patch("lora_tester.nodes.logger.info") as log_info:
            self._run_fake_sample(1, {"log_test_details": False})
        self.assertFalse(log_info.called)

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
