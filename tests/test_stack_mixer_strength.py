from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lora_tester.artist import ARTIST_TAG_MODE, AnimaArtistMixerConfig
from lora_tester.axis_preview import format_axis_preview
from lora_tester.nodes import (
    AnimaFlowXYTestSampler, LoraStackMixerStrengthNode, NODE_CLASS_MAPPINGS,
    XYTestSampler, _combination_preflight, _stack_mixer_config,
    _xy_mixer_labels_possible, _xy_stack_key,
)
from lora_tester.stack import (
    LoraStack, LoraStackItem, LoraStackList, flatten_lora_stack,
    rename_lora_stack, rename_lora_stack_list, replace_stack_artist, split_lora_stack,
)
from lora_tester.xy import PromptEntry, PromptList, build_lora_stack_axis, build_prompt_axis
from tests.test_anima_flow import (
    Clip, Vae, Progress, FakeFlowSampler, FakeFlowSettings, FlowArtistPack, FlowArtistMixer,
)


class StackMixerStrengthTests(unittest.TestCase):
    def setUp(self):
        self.stack = LoraStack((
            LoraStackItem(ARTIST_TAG_MODE, "@first", 0.8),
            LoraStackItem(ARTIST_TAG_MODE, "@second", 0.3),
        ), custom_name="named style")
        self.model = SimpleNamespace(model=SimpleNamespace(model_config=SimpleNamespace(unet_config={"image_model": "anima"})))
        self.registry = SimpleNamespace(NODE_CLASS_MAPPINGS={
            "AnimaFlowCorrectiveSampler": FakeFlowSampler,
            "AnimaFlowSettings": FakeFlowSettings,
            "AnimaArtistPack": FlowArtistPack,
            "AnimaArtistAdapterMixer": FlowArtistMixer,
        })
        self.registry_patch = patch.dict(sys.modules, {"nodes": self.registry})
        self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)
        FakeFlowSampler.calls = []
        FakeFlowSampler.failure = None
        FlowArtistPack.calls = []
        FlowArtistMixer.calls = []

    def configured(self, strength=2.25):
        return LoraStackMixerStrengthNode.set_strength(self.stack, strength)[0]

    def sample(self, stacks, *, flow=False, **changes):
        arguments = dict(
            model=self.model, clip=Clip(), vae=Vae(),
            latent_image={"samples": torch.zeros((1, 4, 2, 2))},
            x_axis=build_lora_stack_axis(LoraStackList(tuple(stacks)), include_base=False),
            y_axis=build_prompt_axis(PromptList((PromptEntry("portrait"),))),
            positive_prompt="", negative_prompt="negative", color_mode="white",
            show_axis_details=False, log_test_details=False, use_anima_artist_mixer=True,
            max_canvas_megapixels=10.0, seed=1, steps=2, cfg=5.0, denoise=1.0,
        )
        if flow:
            arguments.update(cfg_mode="const", flow_solver="flow_euler", flow_schedule="flow_linear", flow_shift=3.0, add_noise=True)
        else:
            arguments.update(sampler_name="euler", scheduler="normal")
        arguments.update(changes)
        with (
            patch("lora_tester.nodes._make_progress_bar", return_value=Progress()),
            patch("lora_tester.nodes._throw_if_interrupted"),
            patch("lora_tester.nodes._common_ksampler", side_effect=lambda *args, **kwargs: args[8]),
        ):
            return (AnimaFlowXYTestSampler() if flow else XYTestSampler()).sample(**arguments)

    def test_node_contract_and_immutable_metadata(self):
        self.assertIs(NODE_CLASS_MAPPINGS["LoraStackMixerStrength"], LoraStackMixerStrengthNode)
        inputs = LoraStackMixerStrengthNode.INPUT_TYPES()["required"]
        self.assertEqual(inputs["lora_stack"], ("LORA_STACK",))
        self.assertEqual(inputs["strength"][1]["default"], 1.0)
        self.assertEqual(LoraStackMixerStrengthNode.RETURN_TYPES, ("LORA_STACK",))
        result = self.configured()
        self.assertEqual(result.anima_mixer_strength, 2.25)
        self.assertIs(result.items, self.stack.items)
        self.assertEqual(result.custom_name, self.stack.custom_name)
        self.assertIsNone(self.stack.anima_mixer_strength)
        self.assertEqual(LoraStackMixerStrengthNode.set_strength(result, 0.5)[0].anima_mixer_strength, 0.5)

    def test_invalid_values_fail_without_loading_models(self):
        for strength in (-0.1, 4.1, math.inf, -math.inf, math.nan):
            with self.subTest(strength=strength), self.assertRaisesRegex(ValueError, "between 0 and 4"):
                self.configured(strength)
        with self.assertRaises(TypeError):
            LoraStackMixerStrengthNode.set_strength("bad", 1.0)

    def test_missing_dependency_warns_and_preserves_output(self):
        self.registry.NODE_CLASS_MAPPINGS.pop("AnimaArtistAdapterMixer")
        with self.assertLogs("lora_tester.nodes", level="WARNING") as captured:
            output = self.configured()
        self.assertEqual(output.anima_mixer_strength, 2.25)
        self.assertIn("was not found", captured.output[0])

    def test_stack_operations_preserve_strength_and_name_rules(self):
        configured = self.configured()
        outputs = [rename_lora_stack(configured, "new"), replace_stack_artist(configured, "first", ARTIST_TAG_MODE, "@other", 0.6)]
        outputs.extend(rename_lora_stack_list(LoraStackList((configured,)), -1, "name-{i}").stacks)
        outputs.extend(split_lora_stack(configured).stacks)
        for mode in ("inherit", "normalize", "dual"):
            outputs.extend(flatten_lora_stack(configured, True, mode).stacks)
            single = replace(configured, items=(configured.items[0],))
            outputs.extend(flatten_lora_stack(single, False, mode).stacks)
        for output in outputs:
            self.assertEqual(output.anima_mixer_strength, 2.25)
        self.assertEqual(split_lora_stack(configured).stacks[-1].custom_name, "named style")
        self.assertIsNone(split_lora_stack(configured).stacks[0].custom_name)
        self.assertIs(LoraStackList.merge((configured,)).stacks[0], configured)

    def test_strength_not_in_image_labels_or_source_table_but_in_preview(self):
        for stack in (self.stack, rename_lora_stack(self.stack, ""), replace(self.stack, items=(self.stack.items[0],), custom_name=None)):
            changed = LoraStackMixerStrengthNode.set_strength(stack, 2.25)[0]
            first = build_lora_stack_axis(LoraStackList((stack,)))
            second = build_lora_stack_axis(LoraStackList((changed,)))
            self.assertEqual([entry.label for entry in first.entries], [entry.label for entry in second.entries])
            self.assertEqual(first.detail_blocks, second.detail_blocks)
            self.assertIn("Stack Mixer strength: 2.25", format_axis_preview(second, "en"))

    def test_override_only_changes_strength_and_partitions_sampling_groups(self):
        advanced = {"artist_anchor_q": True, "anchor_seed_list": "123"}
        config = AnimaArtistMixerConfig(strength=0.0, normalize_weights=False, alignment_mode="shared_base_ids", apply_to_uncond=True, uncond_strength=0.3, advanced_options=advanced)
        output = _stack_mixer_config(self.configured(), config)
        self.assertEqual(output, replace(config, strength=2.25))
        self.assertEqual(config.strength, 0.0)
        self.assertIs(_stack_mixer_config(self.stack, config), config)
        self.assertIsNone(_stack_mixer_config(self.stack, None))
        self.assertNotEqual(_xy_stack_key(self.stack), _xy_stack_key(self.configured()))

    def test_native_and_flow_override_default_and_explicit_global_values(self):
        for flow in (False, True):
            for config in (None, AnimaArtistMixerConfig(strength=0.0, normalize_weights=False, advanced_options={"artist_anchor_q": True})):
                with self.subTest(flow=flow, config=config):
                    FlowArtistMixer.calls = []
                    self.sample((self.configured(),), flow=flow, anima_mixer_config=config)
                    self.assertEqual(FlowArtistMixer.calls[0]["strength"], 2.25)
                    if config is not None:
                        self.assertFalse(FlowArtistMixer.calls[0]["normalize_weights"])
                        self.assertEqual(FlowArtistMixer.calls[0]["advanced_options"], config.advanced_options)

    def test_identical_styles_with_different_strengths_are_not_reused_in_sampling(self):
        stacks = (self.configured(0.5), self.configured(2.25), self.stack)
        for flow in (False, True):
            with self.subTest(flow=flow):
                FlowArtistMixer.calls = []
                result = self.sample(stacks, flow=flow)
                self.assertEqual(result[1].shape[0], 3)
                self.assertEqual([call["strength"] for call in FlowArtistMixer.calls], [0.5, 2.25, 1.0 if flow else 1.6])

    def test_override_respects_existing_activation_conditions(self):
        cases = (
            {"model": ()},
            {"use_anima_artist_mixer": False},
            {"anima_mixer_config": AnimaArtistMixerConfig(enabled=False)},
        )
        for flow in (False, True):
            for changes in cases:
                with self.subTest(flow=flow, changes=changes):
                    FlowArtistMixer.calls = []
                    self.sample((self.configured(),), flow=flow, **changes)
                    self.assertEqual(FlowArtistMixer.calls, [])
            FlowArtistMixer.calls = []
            self.sample((self.configured(0.0),), flow=flow)
            self.assertEqual(FlowArtistMixer.calls, [])
            FlowArtistMixer.calls = []
            single = replace(self.configured(), items=(self.stack.items[0],))
            self.sample((single,), flow=flow)
            self.assertEqual(FlowArtistMixer.calls, [])
        self.registry.NODE_CLASS_MAPPINGS.pop("AnimaArtistAdapterMixer")
        with self.assertLogs("lora_tester.artist", level="WARNING"):
            self.sample((replace(self.stack, anima_mixer_strength=2.25),))
        self.assertEqual(FlowArtistMixer.calls, [])

    def test_single_stack_plus_independent_artist_uses_override(self):
        single = replace(self.configured(), items=(self.stack.items[0],))
        prompts = build_prompt_axis(PromptList((PromptEntry("portrait", independent_artist_tags="@third"),)))
        self.sample((single,), y_axis=prompts)
        self.assertEqual(FlowArtistMixer.calls[0]["strength"], 2.25)

    def test_preflight_and_mixer_layout_use_effective_strength(self):
        configured = self.configured()
        config = AnimaArtistMixerConfig(strength=0.0)
        tasks = [(0, 0, {"lora_stack": configured})]
        self.assertTrue(_xy_mixer_labels_possible(self.model, tasks, config, True))
        self.assertFalse(_xy_mixer_labels_possible(self.model, [(0, 0, {"lora_stack": self.configured(0.0)})], AnimaArtistMixerConfig(), True))
        preflight = _combination_preflight(self.model, (configured,), None, config, True, "")
        self.assertTrue(preflight.mixer_active)
        preflight = _combination_preflight(self.model, (self.configured(0.0),), None, AnimaArtistMixerConfig(), True, "")
        self.assertFalse(preflight.mixer_active)


if __name__ == "__main__":
    unittest.main()
