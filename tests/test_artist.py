from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.artist import (
    ANIMA_ARTIST_TEMPLATE,
    DANBOORU_ARTIST_TEMPLATE,
    ArtistTagTemplate,
    AnimaArtistMixerConfig,
    detect_model_family,
    extract_anima_artist_tags,
    route_artist_prompt,
    split_artist_tags,
)


class _Clip:
    def __init__(self):
        self.encoded = []

    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded.append(tokens)
        return {"text": tokens}


class _AnimaModel:
    def __init__(self):
        self.model = SimpleNamespace(
            model_config=SimpleNamespace(unet_config={"image_model": "anima"})
        )


class _FakePack:
    calls = []

    def pack(self, **kwargs):
        self.calls.append(kwargs)
        return ({"packed": kwargs},)


class _FakeMixer:
    calls = []

    def patch(self, **kwargs):
        self.calls.append(kwargs)
        return ("patched-model", {"conditioning": "mixed"})


class ArtistTests(unittest.TestCase):
    def setUp(self):
        _FakePack.calls.clear()
        _FakeMixer.calls.clear()

    def test_builtin_templates_normalize_prefix_and_existing_weight(self):
        self.assertEqual(ANIMA_ARTIST_TEMPLATE.format("fkey"), "@fkey")
        self.assertEqual(ANIMA_ARTIST_TEMPLATE.format("@fkey"), "@fkey")
        self.assertEqual(ANIMA_ARTIST_TEMPLATE.format("@@fkey"), "@fkey")
        self.assertEqual(
            ANIMA_ARTIST_TEMPLATE.format("(@fkey:3.0)", 1.25),
            "(@fkey:1.25)",
        )
        self.assertEqual(DANBOORU_ARTIST_TEMPLATE.format("@fkey"), "fkey")
        self.assertEqual(
            DANBOORU_ARTIST_TEMPLATE.format("Artist Name", 1.2),
            "(artist_name:1.2)",
        )
        self.assertEqual(DANBOORU_ARTIST_TEMPLATE.format("fkey", 1.2), "(fkey:1.2)")
        self.assertEqual(
            split_artist_tags(" @first, second\n(@third:1.7) "),
            ("first", "second", "third"),
        )

    def test_custom_template_supports_weight_formatting_and_rejects_bad_fields(self):
        template = ArtistTagTemplate("style:{tag}", "[style:{tag}|{weight:.2f}]")
        self.assertEqual(template.format("@Foo"), "style:Foo")
        self.assertEqual(template.format("Foo", 0.75), "[style:Foo|0.75]")
        with self.assertRaisesRegex(ValueError, "must contain"):
            ArtistTagTemplate("artist", "({tag}:{weight})")
        with self.assertRaisesRegex(ValueError, "only support"):
            ArtistTagTemplate("{tag}", "({name}:{weight})")

    def test_model_family_uses_comfy_model_config_not_checkpoint_name(self):
        self.assertEqual(detect_model_family(_AnimaModel()), "anima")
        self.assertEqual(detect_model_family(object()), "danbooru")

    def test_anima_prompt_artist_tags_are_extracted_with_weights(self):
        cleaned, artists = extract_anima_artist_tags(
            "masterpiece, @first artist, (@second_artist:1.25), portrait"
        )
        self.assertEqual(cleaned, "masterpiece, portrait")
        self.assertEqual(artists, (("first artist", 1.0), ("second_artist", 1.25)))
        self.assertEqual(
            extract_anima_artist_tags("contact user@example.com, portrait"),
            ("contact user@example.com, portrait", ()),
        )

    def test_mixer_config_defaults_match_supplied_workflow_screenshot(self):
        config = AnimaArtistMixerConfig()
        self.assertEqual(config.strength, 1.6)
        self.assertTrue(config.normalize_weights)
        self.assertEqual(config.alignment_mode, "base_anchored")
        self.assertTrue(config.enabled)
        self.assertFalse(config.apply_to_uncond)
        self.assertEqual(config.uncond_strength, 0.0)

    def test_non_anima_and_single_anima_artist_use_native_prompt(self):
        for model, entries in (
            (object(), (("first", 1.0), ("second", 1.2))),
            (_AnimaModel(), (("first", 1.0),)),
        ):
            with self.subTest(model=type(model).__name__, count=len(entries)):
                clip = _Clip()
                with patch(
                    "lora_tester.artist._resolve_anima_mixer_nodes",
                    side_effect=AssertionError("Mixer lookup must be skipped"),
                ):
                    route = route_artist_prompt(
                        model=model,
                        clip=clip,
                        mixer_base_prompt="portrait",
                        fallback_prompt="rendered, portrait",
                        artist_entries=entries,
                    )
                self.assertEqual(route.mode, "native_prompt")
                self.assertEqual(route.positive, {"text": "rendered, portrait"})

    def test_missing_mixer_falls_back_only_for_multi_artist_anima(self):
        clip = _Clip()
        with patch("lora_tester.artist._resolve_anima_mixer_nodes", return_value=None):
            route = route_artist_prompt(
                model=_AnimaModel(),
                clip=clip,
                mixer_base_prompt="portrait",
                fallback_prompt="@first, (@second:1.4), portrait",
                artist_entries=(("first", 1.0), ("second", 1.4)),
            )
        self.assertEqual(route.mode, "native_prompt_missing_mixer")
        self.assertEqual(route.positive["text"], "@first, (@second:1.4), portrait")

    def test_prompt_artist_combines_with_test_artist_for_mixer_threshold(self):
        with patch(
            "lora_tester.artist._resolve_anima_mixer_nodes",
            return_value=(_FakePack, _FakeMixer),
        ):
            route = route_artist_prompt(
                model=_AnimaModel(),
                clip=_Clip(),
                mixer_base_prompt="@prompt_artist, portrait",
                fallback_prompt="@test_artist, @prompt_artist, portrait",
                artist_entries=(("test_artist", 1.0),),
            )
        self.assertTrue(route.used_external_mixer)
        self.assertEqual(
            _FakePack.calls[0]["artist_chain"],
            "@test_artist\n@prompt_artist",
        )
        self.assertEqual(_FakePack.calls[0]["base_prompt"], "portrait")

    def test_loaded_mixer_receives_pack_then_exact_adapter_parameters(self):
        model = _AnimaModel()
        clip = _Clip()
        config = AnimaArtistMixerConfig(
            strength=2.0,
            normalize_weights=False,
            alignment_mode="shared_base_ids",
            enabled=True,
            apply_to_uncond=True,
            uncond_strength=0.25,
        )
        with patch(
            "lora_tester.artist._resolve_anima_mixer_nodes",
            return_value=(_FakePack, _FakeMixer),
        ):
            route = route_artist_prompt(
                model=model,
                clip=clip,
                mixer_base_prompt="lora trigger, @prompt_artist, portrait",
                fallback_prompt=(
                    "@first, (@second:1.4), lora trigger, @prompt_artist, portrait"
                ),
                artist_entries=(("@first", 1.0), ("second", 1.4)),
                mixer_config=config,
            )

        self.assertEqual(route.mode, "anima_artist_mixer")
        self.assertEqual(route.model, "patched-model")
        self.assertEqual(route.positive, {"conditioning": "mixed"})
        self.assertEqual(
            _FakePack.calls,
            [
                {
                    "clip": clip,
                    "artist_chain": "@first\n(@second:1.4)\n@prompt_artist",
                    "base_prompt": "lora trigger, portrait",
                }
            ],
        )
        call = _FakeMixer.calls[0]
        self.assertIs(call["model"], model)
        self.assertEqual(call["artist_pack"], {"packed": _FakePack.calls[0]})
        self.assertEqual(
            {key: call[key] for key in (
                "strength",
                "normalize_weights",
                "alignment_mode",
                "enabled",
                "apply_to_uncond",
                "uncond_strength",
            )},
            {
                "strength": 2.0,
                "normalize_weights": False,
                "alignment_mode": "shared_base_ids",
                "enabled": True,
                "apply_to_uncond": True,
                "uncond_strength": 0.25,
            },
        )


if __name__ == "__main__":
    unittest.main()
