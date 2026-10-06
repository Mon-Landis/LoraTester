from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.anima_flow_adapter import (
    FLOW_CONTROLS, PROJECT_URL, AnimaFlowAdapter, AnimaFlowDependencyError,
    anima_flow_status, missing_control_schema, validate_control,
)
from lora_tester.artist import ARTIST_TAG_MODE
from lora_tester.nodes import (
    AnimaFlowParameterAxisNode, AnimaFlowXYTestSampler, NODE_CLASS_MAPPINGS,
    _CachedLora,
)
from lora_tester.stack import LoraStack, LoraStackItem, LoraStackList
from lora_tester.xy import (
    AxisEntry, AxisParameter, PromptEntry, PromptList, SeedList, XYAxis,
    build_lora_stack_axis, build_prompt_axis, build_seed_axis, cross_merge_axes,
)


class FakeFlowSampler:
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    calls = []
    failure = None

    @classmethod
    def INPUT_TYPES(cls):
        controls = missing_control_schema()
        controls["flow_solver"] = (["flow_euler", "flow_new_solver"], {"default": "flow_euler"})
        controls["flow_schedule"] = (["flow_linear", "flow_new_schedule"], {"default": "flow_linear"})
        controls["cfg_mode"] = (["const", "ramp cfg"], {"default": "const"})
        return {
            "required": {"model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "latent_image": ("LATENT",), **controls},
            "optional": {"flow_settings": ("ANIMA_FLOW_SETTINGS",), "vae": ("VAE",)},
        }

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               cfg_mode, flow_solver, flow_schedule, flow_shift, denoise,
               add_noise, flow_settings=None, vae=None):
        FakeFlowSampler.calls.append(dict(
            model=model, positive=positive, negative=negative, seed=seed, steps=steps,
            cfg=cfg, cfg_mode=cfg_mode, flow_solver=flow_solver,
            flow_schedule=flow_schedule, flow_shift=flow_shift, denoise=denoise,
            add_noise=add_noise, flow_settings=deepcopy(flow_settings), vae=vae,
        ))
        if FakeFlowSampler.failure:
            raise FakeFlowSampler.failure
        if isinstance(flow_settings, dict):
            flow_settings["nested"] = "changed"
        latent_image["temporary"] = True
        output = {"samples": torch.full_like(latent_image["samples"], float(seed % 10))}
        image = vae.decode(output["samples"]) if vae is not None else None
        return output, image, "flow log"


class FakeFlowSettings:
    RETURN_TYPES = ("ANIMA_FLOW_SETTINGS", "STRING")
    FUNCTION = "build"
    calls = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "flow_unipc_order": ("INT", {"default": 2, "min": 1, "max": 6}),
            "final_clean_pass": ("BOOLEAN", {"default": False}),
            "flow_unipc_solver_type": (["bh1", "bh2"], {"default": "bh2"}),
        }}

    def build(self, flow_unipc_order, final_clean_pass, flow_unipc_solver_type):
        values = dict(flow_unipc_order=flow_unipc_order, final_clean_pass=final_clean_pass, flow_unipc_solver_type=flow_unipc_solver_type)
        FakeFlowSettings.calls.append(values)
        return {**values, "upstream_default": True}, "settings log"


class Clip:
    def tokenize(self, text):
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens):
        return tokens


class Vae:
    def decode(self, latent):
        return torch.full((1, 8, 10, 3), float(latent.flatten()[0]) / 10)


class Progress:
    def update_absolute(self, *args):
        pass


class AnimaFlowTests(unittest.TestCase):
    def setUp(self):
        FakeFlowSampler.calls = []
        FakeFlowSampler.failure = None
        FakeFlowSettings.calls = []
        self.registry = SimpleNamespace(NODE_CLASS_MAPPINGS={
            "AnimaFlowCorrectiveSampler": FakeFlowSampler,
            "AnimaFlowSettings": FakeFlowSettings,
        })
        self.registry_patch = patch.dict(sys.modules, {"nodes": self.registry})
        self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)

    def controls(self):
        return dict(seed=1, steps=30, cfg=5.0, cfg_mode="const", flow_solver="flow_euler",
                    flow_schedule="flow_linear", flow_shift=3.0, denoise=1.0, add_noise=True,
                    flow_settings=None)

    def sample_arguments(self):
        return dict(
            model=(), clip=Clip(), vae=Vae(),
            latent_image={"samples": torch.zeros((1, 4, 2, 2))},
            x_axis=build_seed_axis(SeedList((1, 2))),
            y_axis=build_prompt_axis(PromptList((PromptEntry("a"), PromptEntry("b")))),
            positive_prompt="base", negative_prompt="negative", color_mode="white",
            show_axis_details=False, log_test_details=False,
            use_anima_artist_mixer=False, max_canvas_megapixels=10.0, **self.controls(),
        )

    def run_sampler(self, node=None, **changes):
        arguments = {**self.sample_arguments(), **changes}
        with (
            patch("lora_tester.nodes._make_progress_bar", return_value=Progress()),
            patch("lora_tester.nodes._throw_if_interrupted"),
            patch("lora_tester.nodes._common_ksampler", side_effect=AssertionError("native sampler used")),
            patch("lora_tester.nodes._decode_vae", side_effect=AssertionError("native VAE decoder used")),
        ):
            return (node or AnimaFlowXYTestSampler()).sample(**arguments)

    def test_missing_dependency_keeps_node_registration_and_actionable_errors(self):
        self.registry.NODE_CLASS_MAPPINGS.clear()
        self.assertIs(NODE_CLASS_MAPPINGS["LoraTesterAnimaFlowXYSampler"], AnimaFlowXYTestSampler)
        self.assertEqual(AnimaFlowXYTestSampler.INPUT_TYPES()["optional"]["flow_settings"][0], "ANIMA_FLOW_SETTINGS")
        self.assertIn(PROJECT_URL, AnimaFlowXYTestSampler.VALIDATE_INPUTS())
        self.assertFalse(anima_flow_status()["available"])
        with self.assertRaisesRegex(AnimaFlowDependencyError, "requires"):
            self.run_sampler()

    def test_external_schema_is_copied_and_not_frozen_or_mutated(self):
        schema = AnimaFlowXYTestSampler.INPUT_TYPES()
        required = schema["required"]
        self.assertIn("flow_new_solver", required["flow_solver"][0])
        self.assertIn("flow_new_schedule", required["flow_schedule"][0])
        self.assertNotIn("sampler_name", required)
        self.assertNotIn("scheduler", required)
        self.assertTrue(required["seed"][1]["control_after_generate"])
        required["flow_solver"][0].append("mutated")
        self.assertNotIn("mutated", FakeFlowSampler.INPUT_TYPES()["required"]["flow_solver"][0])
        self.assertTrue(AnimaFlowXYTestSampler.VALIDATE_INPUTS())

    def test_incompatible_contract_gives_dependency_link(self):
        with patch.object(FakeFlowSampler, "RETURN_TYPES", ("IMAGE",)):
            status = anima_flow_status()
            self.assertFalse(status["available"])
            self.assertIn("Incompatible", status["error"])
            self.assertIn(PROJECT_URL, status["error"])

    def test_parameter_axis_parses_enums_booleans_and_unsigned_seed_exactly(self):
        builder = AnimaFlowParameterAxisNode()
        axis = builder.build_axis("flow_solver", "flow_euler, flow_new_solver", "Solvers")[0]
        self.assertEqual([entry.parameter_map["flow_solver"] for entry in axis.entries], ["flow_euler", "flow_new_solver"])
        axis = builder.build_axis("add_noise", "true\nfalse\n1\n0", "")[0]
        self.assertEqual([entry.parameter_map["add_noise"] for entry in axis.entries], [True, False, True, False])
        axis = builder.build_axis("seed", str(0xFFFFFFFFFFFFFFFF), "Seeds")[0]
        self.assertEqual(axis.entries[0].parameter_map["seed"], 0xFFFFFFFFFFFFFFFF)

    def test_parameter_axis_rejects_unknown_removed_and_out_of_range_values(self):
        builder = AnimaFlowParameterAxisNode()
        for parameter, text in [("sampler_name", "euler"), ("flow_solver", "removed"), ("steps", "0"), ("denoise", "0"), ("cfg", "nan"), ("add_noise", "yes"), ("flow_solver", "")]:
            with self.subTest(parameter=parameter, text=text), self.assertRaises(ValueError):
                builder.build_axis(parameter, text, "test")

    def test_cfg_uses_external_range_not_native_cfg_100_cap(self):
        axis = AnimaFlowParameterAxisNode().build_axis("cfg", "120", "CFG")[0]
        self.run_sampler(x_axis=axis)
        self.assertEqual([call["cfg"] for call in FakeFlowSampler.calls], [120.0, 120.0])

    def test_prompt_and_seed_axes_use_external_sampler_and_row_major_outputs(self):
        sheet, raw = self.run_sampler()
        self.assertEqual([(call["seed"], call["positive"]["text"]) for call in FakeFlowSampler.calls], [(1, "a"), (2, "a"), (1, "b"), (2, "b")])
        self.assertEqual(tuple(raw.shape), (4, 8, 10, 3))
        self.assertTrue(torch.allclose(raw[:, 0, 0, 0], torch.tensor([0.1, 0.2, 0.1, 0.2])))
        self.assertEqual(int(sheet.shape[0]), 1)
        self.assertTrue(all(isinstance(call["vae"], Vae) for call in FakeFlowSampler.calls))

    def test_solver_axis_and_cross_merge_preserve_parameters(self):
        solvers = AnimaFlowParameterAxisNode().build_axis("flow_solver", "flow_euler\nflow_new_solver", "solver")[0]
        steps = AnimaFlowParameterAxisNode().build_axis("steps", "10\n20", "steps")[0]
        self.run_sampler(x_axis=cross_merge_axes(solvers, steps))
        self.assertEqual([(call["flow_solver"], call["steps"]) for call in FakeFlowSampler.calls[:4]], [("flow_euler", 10), ("flow_euler", 20), ("flow_new_solver", 10), ("flow_new_solver", 20)])

    def test_advanced_axis_builds_settings_through_external_public_node(self):
        settings = {"flow_unipc_order": 4, "final_clean_pass": False, "flow_unipc_solver_type": "bh1", "cfg_interval_start": 0.18}
        axis = AnimaFlowParameterAxisNode().build_axis("final_clean_pass", "true\nfalse", "clean")[0]
        self.run_sampler(x_axis=axis, flow_settings=settings)
        self.assertEqual(settings, {"flow_unipc_order": 4, "final_clean_pass": False, "flow_unipc_solver_type": "bh1", "cfg_interval_start": 0.18})
        self.assertEqual([call["flow_settings"]["final_clean_pass"] for call in FakeFlowSampler.calls], [True, False, True, False])
        self.assertTrue(all(call["flow_settings"]["flow_unipc_order"] == 4 for call in FakeFlowSampler.calls))
        self.assertTrue(all(call["flow_settings"]["upstream_default"] for call in FakeFlowSampler.calls))
        self.assertTrue(all(call["flow_settings"]["cfg_interval_start"] == 0.18 for call in FakeFlowSampler.calls))

    def test_passthrough_settings_and_latent_metadata_are_not_mutated(self):
        settings = {"nested": {"value": 2}}
        latent = {"samples": torch.zeros((1, 4, 2, 2)), "batch_index": [0]}
        self.run_sampler(flow_settings=settings, latent_image=latent)
        self.assertEqual(settings, {"nested": {"value": 2}})
        self.assertNotIn("temporary", latent)
        self.assertEqual(latent["batch_index"], [0])

    def test_style_artist_and_lora_axes_share_existing_routing(self):
        stack = LoraStack((LoraStackItem("test.safetensors", "@ordinary_trigger", 0.5), LoraStackItem(ARTIST_TAG_MODE, "@artist", 0.7)))
        axis = build_lora_stack_axis(LoraStackList((stack,)), include_base=True)
        with (
            patch.object(AnimaFlowXYTestSampler, "_load_lora_with_status", return_value=(_CachedLora("path", None, {}, {}), False)),
            patch("lora_tester.nodes._apply_lora_to_models", side_effect=lambda model, clip, *args: (model, clip)),
        ):
            self.run_sampler(x_axis=axis)
        self.assertEqual(len(FakeFlowSampler.calls), 4)
        styled = [call for call in FakeFlowSampler.calls if "@ordinary_trigger" in call["positive"]["text"]]
        self.assertEqual(len(styled), 2)
        self.assertTrue(all("artist" in call["positive"]["text"] for call in styled))

    def test_axis_conflicts_and_native_options_fail_before_sampling(self):
        axis = AnimaFlowParameterAxisNode().build_axis("flow_solver", "flow_euler", "solver")[0]
        with self.assertRaisesRegex(ValueError, "same sampling parameter"):
            self.run_sampler(x_axis=axis, y_axis=axis)
        axis = XYAxis("native", ((AxisEntry("euler", (AxisParameter("sampler_name", "euler"),)),),))
        with self.assertRaisesRegex(ValueError, "cannot be mapped"):
            self.run_sampler(x_axis=axis)
        self.assertEqual(FakeFlowSampler.calls, [])

    def test_sampling_failure_does_not_fallback_and_clears_run_cache(self):
        node = AnimaFlowXYTestSampler()
        FakeFlowSampler.failure = RuntimeError("upstream failed")
        with self.assertRaisesRegex(RuntimeError, "upstream failed"):
            self.run_sampler(node=node)
        self.assertEqual(node._lora_cache, {})

    def test_full_settings_axis_replaces_optional_settings(self):
        replacement = {"final_clean_pass": True}
        axis = XYAxis("settings", ((AxisEntry("custom", (AxisParameter("flow_settings", replacement),)),),))
        self.run_sampler(x_axis=axis, flow_settings={"final_clean_pass": False})
        self.assertTrue(all(call["flow_settings"]["final_clean_pass"] for call in FakeFlowSampler.calls))

    def test_invalid_result_is_reported_as_incompatible_dependency(self):
        adapter = AnimaFlowAdapter()
        with patch.object(FakeFlowSampler, "sample", return_value=({}, None, "log")):
            with self.assertRaisesRegex(AnimaFlowDependencyError, "LATENT result"):
                adapter.sample_cell(model=None, positive=None, negative=None, latent={}, values=self.controls())

    def test_missing_settings_node_keeps_basic_sampler_available(self):
        self.registry.NODE_CLASS_MAPPINGS.pop("AnimaFlowSettings")
        self.assertTrue(anima_flow_status()["available"])
        self.assertNotIn("final_clean_pass", AnimaFlowParameterAxisNode.INPUT_TYPES()["required"]["parameter"][0])

    def test_numeric_validation_does_not_accept_fractional_integer_or_nonfinite(self):
        with self.assertRaises(ValueError):
            validate_control("steps", 1.5, ("INT", {}))
        with self.assertRaises(ValueError):
            validate_control("cfg", float("inf"), ("FLOAT", {}))


if __name__ == "__main__":
    unittest.main()
