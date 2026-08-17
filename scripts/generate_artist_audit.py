"""Generate a deterministic audit of artist-tag routing using production code."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.artist import (  # noqa: E402
    ANIMA_ARTIST_TEMPLATE,
    DANBOORU_ARTIST_TEMPLATE,
    ArtistTagTemplate,
    AnimaArtistMixerConfig,
    parse_artist_tag_entries,
)
from lora_tester.nodes import (  # noqa: E402
    _combination_preflight,
    _route_prompt_entries,
)
from lora_tester.stack import LoraStack, LoraStackItem  # noqa: E402


class _FakeClip:
    def encode_from_tokens_scheduled(self, tokens):
        return {"text": tokens}

    def tokenize(self, text):
        return text


class _FakeAnimaModel:
    model = SimpleNamespace(
        model_config=SimpleNamespace(unet_config={"image_model": "anima"})
    )


class _FakePack:
    calls: list[dict[str, object]] = []

    def pack(self, **kwargs):
        self.calls.append(dict(kwargs))
        return ({"artist_chain": kwargs["artist_chain"], "base_prompt": kwargs["base_prompt"]},)


class _FakeMixer:
    calls: list[dict[str, object]] = []

    def patch(self, **kwargs):
        self.calls.append(dict(kwargs))
        return (
            ("mixed-model", kwargs["artist_pack"]["artist_chain"]),
            {"text": "MIXED", "artist_chain": kwargs["artist_pack"]["artist_chain"]},
        )


def _route_case(
    name: str,
    *,
    model: object,
    positive: str,
    entries: tuple[tuple[bool, str, float], ...],
    independent: str = "",
    mixer_available: bool,
    template: ArtistTagTemplate | None = None,
    config: AnimaArtistMixerConfig | None = None,
    use_anima_artist_mixer: bool = True,
) -> dict[str, object]:
    _FakePack.calls.clear()
    _FakeMixer.calls.clear()
    with patch(
        "lora_tester.artist._resolve_anima_mixer_nodes",
        return_value=((_FakePack, _FakeMixer) if mixer_available else None),
    ):
        route = _route_prompt_entries(
            model=model,
            clip=_FakeClip(),
            prefix_parts=(),
            entries=entries,
            suffix_parts=(positive,),
            artist_template=template,
            mixer_config=config,
            independent_artist_tags=independent,
            use_anima_artist_mixer=use_anima_artist_mixer,
        )
    pack = _FakePack.calls[0] if _FakePack.calls else None
    return {
        "case": name,
        "model_family": "anima" if isinstance(model, _FakeAnimaModel) else "danbooru",
        "positive_prompt": positive,
        "test_entries": [list(item) for item in entries],
        "independent_artist_tags": independent,
        "mixer_available": mixer_available,
        "mixer_switch_enabled": use_anima_artist_mixer,
        "route": route.mode,
        "rendered_tags": list(route.rendered_tags),
        "artist_chain": pack["artist_chain"] if pack else None,
        "mixer_base_prompt": pack["base_prompt"] if pack else None,
        "encoded_positive": route.positive,
    }


def build_audit() -> dict[str, object]:
    parser_inputs = (
        "fkey",
        "@fkey",
        "@@fkey",
        "(@fkey:0.4)",
        "fkey\n(@wlop:1.2), second",
    )
    parser_cases = [
        {"input": value, "entries": [list(item) for item in parse_artist_tag_entries(value)]}
        for value in parser_inputs
    ]
    template_cases = [
        {
            "model_family": "anima",
            "default": ANIMA_ARTIST_TEMPLATE.format("@fkey"),
            "weighted": ANIMA_ARTIST_TEMPLATE.format("@fkey", 1.2),
        },
        {
            "model_family": "danbooru",
            "default": DANBOORU_ARTIST_TEMPLATE.format("Artist Name"),
            "weighted": DANBOORU_ARTIST_TEMPLATE.format("Artist Name", 1.2),
        },
    ]
    anima = _FakeAnimaModel()
    routing_cases = [
        _route_case(
            "anima_lora_only_is_native",
            model=anima,
            positive="portrait",
            entries=((False, "@trigger_artist", 0.25),),
            mixer_available=True,
        ),
        _route_case(
            "anima_single_artist_plus_lora_is_native",
            model=anima,
            positive="portrait",
            entries=(
                (True, "test_artist", 0.25),
                (False, "@trigger_artist", 0.25),
            ),
            mixer_available=True,
        ),
        _route_case(
            "anima_prompt_tag_is_base_text",
            model=anima,
            positive="@prompt_artist, portrait",
            entries=((True, "test_artist", 1.0),),
            mixer_available=True,
        ),
        _route_case(
            "anima_independent_tags_use_mixer",
            model=anima,
            positive="@prompt_artist, portrait",
            entries=((True, "test_artist", 1.0),),
            independent="@independent, (@second:0.5)",
            mixer_available=True,
        ),
        _route_case(
            "anima_independent_tags_fallback_without_mixer",
            model=anima,
            positive="@prompt_artist, portrait",
            entries=((True, "test_artist", 1.0),),
            independent="@independent, (@second:0.5)",
            mixer_available=False,
        ),
        _route_case(
            "anima_lora_trigger_is_not_extracted",
            model=anima,
            positive="portrait",
            entries=((True, "test_artist", 1.0), (False, "@trigger_artist", 1.0)),
            independent="@independent",
            mixer_available=True,
        ),
        _route_case(
            "anima_multi_artist_switch_disabled",
            model=anima,
            positive="portrait",
            entries=((True, "first", 1.0), (True, "second", 1.0)),
            mixer_available=True,
            use_anima_artist_mixer=False,
        ),
        _route_case(
            "danbooru_uses_native_tag_template",
            model=object(),
            positive="@prompt_artist, portrait",
            entries=((True, "Artist Name", 1.2),),
            independent="fkey",
            mixer_available=True,
        ),
    ]

    two_artists = LoraStack(
        (
            LoraStackItem("__lora_tester_artist_tag__", "@first", 0.5),
            LoraStackItem("__lora_tester_artist_tag__", "@second", 1.0),
        )
    )
    one_artist = LoraStack(
        (LoraStackItem("__lora_tester_artist_tag__", "@first", 1.0),)
    )
    preflight_cases = []
    for name, model, stacks, available, use_mixer in (
        ("anima_mixer_present", anima, (two_artists,), True, True),
        ("anima_mixer_missing", anima, (two_artists,), False, True),
        ("anima_switch_disabled", anima, (two_artists,), True, False),
        ("anima_single_artist", anima, (one_artist,), True, True),
        ("danbooru_multi_artist", object(), (two_artists,), True, True),
    ):
        with patch("lora_tester.nodes.anima_artist_mixer_available", return_value=available):
            result = _combination_preflight(
                model, stacks, None, AnimaArtistMixerConfig(), use_mixer
            )
        preflight_cases.append(
            {
                "case": name,
                "model_family": result.model_family,
                "has_artist_tags": result.has_artist_tags,
                "has_multi_artist_tests": result.has_multi_artist_tests,
                "mixer_available": result.mixer_available,
                "mixer_switch_enabled": result.mixer_switch_enabled,
                "mixer_enabled": result.mixer_enabled,
                "mixer_active": result.mixer_active,
                "mixer_combinations": list(result.mixer_combinations),
            }
        )
    return {
        "scope": "production routing functions; no image generation",
        "parser_cases": parser_cases,
        "template_cases": template_cases,
        "routing_cases": routing_cases,
        "preflight_cases": preflight_cases,
    }


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def render_markdown(audit: dict[str, object]) -> str:
    lines = [
        "# Artist Tag Routing Audit",
        "",
        "Generated by `scripts/generate_artist_audit.py` from the production routing functions.",
        "This report does not generate images; it records deterministic routing and preflight behavior.",
        "",
        "## Parser and Templates",
        "",
        "| Input | Parsed entries |",
        "| --- | --- |",
    ]
    for case in audit["parser_cases"]:
        lines.append(f"| `{_cell(case['input'])}` | `{_cell(case['entries'])}` |")
    lines.extend([
        "",
        "| Model family | Default form | Weighted form |",
        "| --- | --- | --- |",
    ])
    for case in audit["template_cases"]:
        lines.append(
            f"| {case['model_family']} | `{case['default']}` | `{case['weighted']}` |"
        )
    lines.extend([
        "",
        "## Routing Cases",
        "",
        "| Case | Family | Mixer | Switch | Route | Artist chain | Mixer base prompt | Encoded/native result |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for case in audit["routing_cases"]:
        lines.append(
            "| " + " | ".join(
                _cell(case[key])
                for key in (
                    "case", "model_family", "mixer_available",
                    "mixer_switch_enabled", "route",
                    "artist_chain", "mixer_base_prompt", "encoded_positive",
                )
            ) + " |"
        )
    lines.extend([
        "",
        "## Combination Preflight",
        "",
        "| Case | Family | Artist tags | Multi-artist tests | Available | Switch | Enabled | Active | Mixer combinations |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for case in audit["preflight_cases"]:
        lines.append(
            "| " + " | ".join(
                _cell(case[key])
                for key in (
                    "case", "model_family", "has_artist_tags",
                    "has_multi_artist_tests", "mixer_available",
                    "mixer_switch_enabled", "mixer_enabled", "mixer_active",
                    "mixer_combinations",
                )
            ) + " |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `@tag` in a normal positive prompt or LoRA trigger remains ordinary base prompt text.",
        "- Only explicit artist-mode entries and the direct sampler's independent artist field become artist entries.",
        "- Anima uses the optional external Mixer only for a multi-artist test when it is available and enabled.",
        "- Disabling the sampler's advanced Mixer switch forces native prompt encoding before external-node lookup.",
        "- Non-Anima models use their native artist-tag template and never activate the Anima Mixer.",
    ])
    return "\n".join(lines) + "\n"


def write_report(output_dir: Path = ROOT / "audit") -> tuple[Path, Path]:
    audit = build_audit()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "artist_routing_report.json"
    markdown_path = output_dir / "artist_routing_report.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")
    return markdown_path, json_path


if __name__ == "__main__":
    for path in write_report():
        print(path)
