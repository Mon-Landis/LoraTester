from __future__ import annotations

import importlib
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lora_tester.artist import ARTIST_TAG_MODE, AnimaArtistMixerConfig
from lora_tester.nodes import AnimaFlowParameterAxisNode, AnimaFlowXYTestSampler
from lora_tester.stack import LoraStack, LoraStackItem, LoraStackList
from lora_tester.xy import PromptEntry, PromptList, SeedList, build_lora_stack_axis, build_prompt_axis, build_seed_axis


FLOW_ROOT = os.environ.get("LORA_TESTER_ANIMA_FLOW_ROOT")
MIXER_ROOT = os.environ.get("LORA_TESTER_ANIMA_MIXER_ROOT")


class AuditClip:
    def tokenize(self, text):
        return str(text)

    def encode_from_tokens_scheduled(self, text):
        values = [ord(character) for character in text] or [0]
        ids = torch.tensor(values, dtype=torch.long)
        raw = torch.stack((ids.float() / 1000, ids.float() / 2000), dim=-1).unsqueeze(0)
        return [[raw, {"t5xxl_ids": ids}]]


class AuditDiffusionModel:
    def preprocess_text_embeds(self, raw, ids, t5xxl_weights=None):
        return raw


class AuditModel:
    def __init__(self):
        self.diffusion_model = AuditDiffusionModel()
        self.model = SimpleNamespace(model_config=SimpleNamespace(unet_config={"image_model": "anima"}))
        self.model_options = {}
        self.object_patches = {}
        self.callbacks = {}

    def get_model_object(self, name):
        if name != "diffusion_model":
            raise KeyError(name)
        return self.diffusion_model

    def clone(self):
        cloned = AuditModel()
        cloned.diffusion_model = self.diffusion_model
        cloned.model = self.model
        cloned.model_options = dict(self.model_options)
        cloned.object_patches = dict(self.object_patches)
        cloned.callbacks = deepcopy(self.callbacks)
        for callback in self.get_all_callbacks("on_clone"):
            callback(self, cloned)
        return cloned

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper

    def get_callbacks(self, call_type, key):
        return self.callbacks.get(call_type, {}).get(key, [])

    def get_all_callbacks(self, call_type):
        return [callback for callbacks in self.callbacks.get(call_type, {}).values() for callback in callbacks]

    def add_callback_with_key(self, call_type, key, callback):
        self.callbacks.setdefault(call_type, {}).setdefault(key, []).append(callback)

    def remove_callbacks_with_key(self, call_type, key):
        self.callbacks.setdefault(call_type, {}).pop(key, None)


class AuditVae:
    def decode(self, samples):
        return samples[:, :3].permute(0, 2, 3, 1)


class AuditProgress:
    def update_absolute(self, *arguments):
        pass


@unittest.skipUnless(FLOW_ROOT and MIXER_ROOT, "Set external Flow/Mixer roots to run real-upstream CPU boundary tests")
class AnimaFlowUpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module_patch = patch.dict(sys.modules)
        cls.module_patch.start()
        cls.addClassCleanup(cls.module_patch.stop)
        cls.path_patch = patch.object(sys, "path", [str(Path(FLOW_ROOT)), str(Path(MIXER_ROOT)), *sys.path])
        cls.path_patch.start()
        cls.addClassCleanup(cls.path_patch.stop)
        cls.sampler_module = importlib.import_module("anima_sampler.corrective_sampler_node")
        cls.settings_module = importlib.import_module("anima_sampler.settings_node")
        cls.pack_module = importlib.import_module("anima_mixer.nodes_core")
        cls.mixer_module = importlib.import_module("anima_mixer.nodes_embedding")

    def setUp(self):
        registry = SimpleNamespace(NODE_CLASS_MAPPINGS={
            "AnimaFlowCorrectiveSampler": self.sampler_module.AnimaFlowCorrectiveSampler,
            "AnimaFlowSettings": self.settings_module.AnimaFlowSettings,
            "AnimaArtistPack": self.pack_module.AnimaArtistPack,
            "AnimaArtistAdapterMixer": self.mixer_module.AnimaArtistAdapterMixer,
        })
        registry_patch = patch.dict(sys.modules, {"nodes": registry})
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        self.calls = []
        self.backend_patch = patch.object(self.sampler_module, "_run_sampler_with_params", side_effect=self.capture_backend)
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)

    def capture_backend(self, **values):
        self.calls.append(values)
        output = dict(values["latent_image"])
        output["samples"] = torch.full_like(output["samples"], float(values["params"]["seed"] % 17) / 17)
        return output, "CPU boundary parity audit"

    def controls(self, mode="const", settings=None):
        return dict(
            seed=0xFFFFFFFFFFFFFFFF, steps=20, cfg=6.0, cfg_mode=mode,
            flow_solver="flow_euler", flow_schedule="flow_cosmos_rho7", flow_shift=3.0,
            denoise=1.0, add_noise=True, flow_settings=settings,
        )

    def run_xy(self, model, clip, vae, latent, controls, x_axis=None, y_axis=None, config=None):
        with (
            patch("lora_tester.nodes._make_progress_bar", return_value=AuditProgress()),
            patch("lora_tester.nodes._throw_if_interrupted"),
        ):
            return AnimaFlowXYTestSampler().sample(
                model=model, clip=clip, vae=vae, latent_image=latent,
                x_axis=x_axis or build_seed_axis(SeedList((controls["seed"],))),
                y_axis=y_axis or build_prompt_axis(PromptList((PromptEntry("portrait"),))),
                positive_prompt="unused", negative_prompt="negative", color_mode="white",
                show_axis_details=False, log_test_details=False, use_anima_artist_mixer=True,
                anima_mixer_config=config, **controls,
            )

    def assert_conditioning_equal(self, first, second):
        self.assertEqual(len(first), len(second))
        for first_entry, second_entry in zip(first, second):
            self.assertTrue(torch.equal(first_entry[0], second_entry[0]))
            self.assertTrue(torch.equal(first_entry[1]["t5xxl_ids"], second_entry[1]["t5xxl_ids"]))

    def assert_backend_equal(self, direct, xy):
        self.assertEqual(direct["params"], xy["params"])
        for name in ("denoise", "add_noise", "disable_pbar"):
            self.assertEqual(direct[name], xy[name])
        self.assert_conditioning_equal(direct["positive"], xy["positive"])
        self.assert_conditioning_equal(direct["negative"], xy["negative"])
        self.assertTrue(torch.equal(direct["latent_image"]["samples"], xy["latent_image"]["samples"]))
        self.assertEqual(direct["latent_image"]["batch_index"], xy["latent_image"]["batch_index"])
        self.assertTrue(torch.equal(direct["latent_image"]["noise_mask"], xy["latent_image"]["noise_mask"]))

    def default_settings(self):
        schema = self.settings_module.AnimaFlowSettings.INPUT_TYPES()["required"]
        values = {name: specification[1]["default"] for name, specification in schema.items()}
        return self.settings_module.AnimaFlowSettings().build(**values)[0]

    def test_actual_external_sampler_normalizes_direct_and_xy_identically(self):
        for mode in ("const", "ramp cfg", "bump cfg"):
            for settings in (None, {}, self.default_settings()):
                with self.subTest(mode=mode, settings_connected=settings is not None):
                    clip, model, vae = AuditClip(), AuditModel(), AuditVae()
                    latent = {"samples": torch.zeros((1, 4, 4, 4)), "batch_index": [7], "noise_mask": torch.ones((1, 4, 4))}
                    controls = self.controls(mode, settings)
                    direct = self.sampler_module.AnimaFlowCorrectiveSampler().sample(
                        model=model, positive=clip.encode_from_tokens_scheduled("portrait"),
                        negative=clip.encode_from_tokens_scheduled("negative"), latent_image=latent, vae=vae, **controls,
                    )
                    xy = self.run_xy(model, clip, vae, latent, controls)
                    self.assert_backend_equal(self.calls[-2], self.calls[-1])
                    self.assertTrue(torch.equal(direct[1], xy[1]))

    def test_actual_external_pack_and_mixer_match_direct_production(self):
        clip, model, vae = AuditClip(), AuditModel(), AuditVae()
        latent = {"samples": torch.zeros((1, 4, 4, 4)), "batch_index": [7], "noise_mask": torch.ones((1, 4, 4))}
        artist_text = r"@first_\(alias\), (@second:0.4)"
        stack = LoraStack((LoraStackItem(ARTIST_TAG_MODE, r"@first_\(alias\)", 1.0), LoraStackItem(ARTIST_TAG_MODE, "@second", 0.4)))
        axis = build_lora_stack_axis(LoraStackList((stack,)), include_base=False)
        schema = self.mixer_module.AnimaArtistAdapterMixer.INPUT_TYPES()["required"]
        defaults = {name: specification[1]["default"] for name, specification in schema.items() if name not in ("model", "artist_pack")}
        pack = self.pack_module.AnimaArtistPack().pack(clip=clip, artist_chain=artist_text, base_prompt="portrait")[0]
        direct_model, direct_positive = self.mixer_module.AnimaArtistAdapterMixer().patch(model=model, artist_pack=pack, **defaults)
        controls = self.controls(settings=self.default_settings())
        self.sampler_module.AnimaFlowCorrectiveSampler().sample(
            model=direct_model, positive=direct_positive, negative=clip.encode_from_tokens_scheduled("negative"),
            latent_image=latent, vae=vae, **controls,
        )
        self.run_xy(model, clip, vae, latent, controls, x_axis=axis)
        direct, xy = self.calls[-2:]
        self.assert_backend_equal(direct, xy)
        direct_state = direct["model"].model_options["model_function_wrapper"]._anima_adapter_mixer_state
        xy_state = xy["model"].model_options["model_function_wrapper"]._anima_adapter_mixer_state
        for name in ("labels", "strength", "alignment_mode", "normalize_weights", "user_weights", "artist_anchor_q"):
            self.assertEqual(direct_state[name], xy_state[name])
        for name in ("raws", "ids_list"):
            self.assertEqual(len(direct_state[name]), len(xy_state[name]))
            for direct_tensor, xy_tensor in zip(direct_state[name], xy_state[name]):
                self.assertTrue(torch.equal(direct_tensor, xy_tensor))

    def test_actual_advanced_axis_matches_settings_node_output(self):
        clip, model, vae = AuditClip(), AuditModel(), AuditVae()
        latent = {"samples": torch.zeros((1, 4, 4, 4)), "batch_index": [7], "noise_mask": torch.ones((1, 4, 4))}
        settings = self.default_settings()
        direct_controls = self.controls(settings={**settings, "final_clean_pass": True})
        self.sampler_module.AnimaFlowCorrectiveSampler().sample(
            model=model, positive=clip.encode_from_tokens_scheduled("portrait"), negative=clip.encode_from_tokens_scheduled("negative"),
            latent_image=latent, vae=vae, **direct_controls,
        )
        axis = AnimaFlowParameterAxisNode().build_axis("final_clean_pass", "true", "CLEAN")[0]
        self.run_xy(model, clip, vae, latent, self.controls(settings=settings), x_axis=axis)
        self.assert_backend_equal(self.calls[-2], self.calls[-1])


if __name__ == "__main__":
    unittest.main()
