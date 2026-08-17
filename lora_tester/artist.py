from __future__ import annotations

import importlib
import logging
import math
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


ARTIST_TAG_MODE = "__lora_tester_artist_tag__"
MODEL_FAMILY_ANIMA = "anima"
MODEL_FAMILY_DANBOORU = "danbooru"
ANIMA_ARTIST_PACK_NODE = "AnimaArtistPack"
ANIMA_ADAPTER_MIXER_NODE = "AnimaArtistAdapterMixer"

_EXPLICIT_WEIGHT = re.compile(
    r"^\(\s*(.*?)\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\)$"
)
_ANIMA_PROMPT_WEIGHTED = re.compile(
    r"^\(\s*@(.+?)\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\)$"
)
_ANIMA_PROMPT_PLAIN = re.compile(r"^@(.+?)$")

logger = logging.getLogger(__name__)


def _template_fields(value: str) -> set[str]:
    fields: set[str] = set()
    try:
        parts = string.Formatter().parse(str(value))
        for _literal, field, _format_spec, _conversion in parts:
            if field is not None:
                fields.add(field)
    except ValueError as error:
        raise ValueError(f"Invalid artist tag template: {error}") from error
    unknown = fields - {"tag", "weight"}
    if unknown:
        raise ValueError(
            "Artist tag templates only support {tag} and {weight}; "
            f"received {sorted(unknown)!r}"
        )
    return fields


def normalize_artist_tag(value: str) -> str:
    tag = str(value).strip()
    match = _EXPLICIT_WEIGHT.fullmatch(tag)
    if match:
        tag = match.group(1).strip()
    tag = tag.lstrip("@").strip()
    if not tag:
        raise ValueError("Artist tag cannot be empty")
    return tag


def split_artist_tags(value: str) -> tuple[str, ...]:
    parts = re.split(r"[,\r\n\uff0c]+", str(value or ""))
    return tuple(normalize_artist_tag(part) for part in parts if part.strip())


def parse_artist_tag_entries(value: str) -> tuple[tuple[str, float], ...]:
    """Parse artist-only text where every comma/newline item is an artist tag."""
    entries: list[tuple[str, float]] = []
    for raw_part in re.split(r"[,\r\n\uff0c]+", str(value or "")):
        part = raw_part.strip()
        if not part:
            continue
        weighted = _EXPLICIT_WEIGHT.fullmatch(part)
        weight = float(weighted.group(2)) if weighted else 1.0
        if not math.isfinite(weight):
            raise ValueError("Artist tag weight must be finite")
        tag = weighted.group(1) if weighted else part
        entries.append((normalize_artist_tag(tag), weight))
    return tuple(entries)


def extract_anima_artist_tags(text: str) -> tuple[str, tuple[tuple[str, float], ...]]:
    original = str(text or "").strip()
    if not original or "@" not in original:
        return original, ()
    retained: list[str] = []
    artists: list[tuple[str, float]] = []
    for raw_part in re.split(r"[,\r\n\uff0c]+", original):
        part = raw_part.strip()
        if not part:
            continue
        weighted = _ANIMA_PROMPT_WEIGHTED.fullmatch(part)
        if weighted:
            weight = float(weighted.group(2))
            if not math.isfinite(weight):
                retained.append(part)
                continue
            artists.append((normalize_artist_tag(weighted.group(1)), weight))
            continue
        plain = _ANIMA_PROMPT_PLAIN.fullmatch(part)
        if plain:
            artists.append((normalize_artist_tag(plain.group(1)), 1.0))
            continue
        retained.append(part)
    if not artists:
        return original, ()
    return ", ".join(retained), tuple(artists)


@dataclass(frozen=True, slots=True)
class ArtistTagTemplate:
    default_template: str
    weighted_template: str
    tag_style: str = "literal"

    def __post_init__(self) -> None:
        default = str(self.default_template).strip()
        weighted = str(self.weighted_template).strip()
        if not default or not weighted:
            raise ValueError("Artist tag templates cannot be empty")
        if "tag" not in _template_fields(default):
            raise ValueError("The default artist tag template must contain {tag}")
        weighted_fields = _template_fields(weighted)
        if "tag" not in weighted_fields or "weight" not in weighted_fields:
            raise ValueError(
                "The weighted artist tag template must contain {tag} and {weight}"
            )
        if self.tag_style not in {"literal", "danbooru"}:
            raise ValueError("Artist tag template style must be literal or danbooru")
        object.__setattr__(self, "default_template", default)
        object.__setattr__(self, "weighted_template", weighted)

    def format(self, tag: str, weight: float = 1.0) -> str:
        normalized = normalize_artist_tag(tag)
        if self.tag_style == "danbooru":
            normalized = re.sub(r"\s+", "_", normalized.lower())
        numeric_weight = float(weight)
        if not math.isfinite(numeric_weight):
            raise ValueError("Artist tag weight must be finite")
        template = (
            self.default_template
            if math.isclose(numeric_weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
            else self.weighted_template
        )
        return template.format(tag=normalized, weight=numeric_weight).strip()


ANIMA_ARTIST_TEMPLATE = ArtistTagTemplate("@{tag}", "(@{tag}:{weight})")
DANBOORU_ARTIST_TEMPLATE = ArtistTagTemplate(
    "{tag}",
    "({tag}:{weight})",
    tag_style="danbooru",
)


@dataclass(frozen=True, slots=True)
class AnimaArtistMixerConfig:
    strength: float = 1.6
    normalize_weights: bool = True
    alignment_mode: str = "base_anchored"
    enabled: bool = True
    apply_to_uncond: bool = False
    uncond_strength: float = 0.0

    def __post_init__(self) -> None:
        strength = float(self.strength)
        uncond_strength = float(self.uncond_strength)
        if not math.isfinite(strength) or not 0.0 <= strength <= 4.0:
            raise ValueError("Mixer strength must be finite and between 0 and 4")
        if self.alignment_mode not in {"base_anchored", "shared_base_ids"}:
            raise ValueError(
                "Mixer alignment_mode must be base_anchored or shared_base_ids"
            )
        if not math.isfinite(uncond_strength) or not 0.0 <= uncond_strength <= 1.0:
            raise ValueError("Mixer uncond_strength must be finite and between 0 and 1")
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "normalize_weights", bool(self.normalize_weights))
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "apply_to_uncond", bool(self.apply_to_uncond))
        object.__setattr__(self, "uncond_strength", uncond_strength)


@dataclass(frozen=True, slots=True)
class ArtistPromptRoute:
    model: Any
    positive: Any
    rendered_tags: tuple[str, ...]
    mode: str

    @property
    def used_external_mixer(self) -> bool:
        return self.mode == "anima_artist_mixer"


def detect_model_family(model: Any) -> str:
    inner_model = getattr(model, "model", None)
    model_config = getattr(inner_model, "model_config", None)
    unet_config = getattr(model_config, "unet_config", None)
    if isinstance(unet_config, Mapping):
        image_model = str(unet_config.get("image_model", "")).strip().lower()
        if image_model == MODEL_FAMILY_ANIMA:
            return MODEL_FAMILY_ANIMA
    if type(model_config).__name__.lower() == MODEL_FAMILY_ANIMA:
        return MODEL_FAMILY_ANIMA

    diffusion_model = None
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except (AttributeError, KeyError, TypeError):
        diffusion_model = getattr(inner_model, "diffusion_model", None)
    if diffusion_model is not None:
        model_type = type(diffusion_model)
        if (
            model_type.__name__.lower() == MODEL_FAMILY_ANIMA
            and ".anima" in model_type.__module__.lower()
        ):
            return MODEL_FAMILY_ANIMA
    return MODEL_FAMILY_DANBOORU


def artist_template_for_model(model: Any) -> ArtistTagTemplate:
    if detect_model_family(model) == MODEL_FAMILY_ANIMA:
        return ANIMA_ARTIST_TEMPLATE
    return DANBOORU_ARTIST_TEMPLATE


def _encode_prompt(clip: Any, text: str) -> Any:
    return clip.encode_from_tokens_scheduled(clip.tokenize(str(text)))


def _resolve_anima_mixer_nodes() -> tuple[type[Any], type[Any]] | None:
    try:
        comfy_nodes = importlib.import_module("nodes")
    except ImportError:
        return None
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mappings, Mapping):
        return None
    pack = mappings.get(ANIMA_ARTIST_PACK_NODE)
    mixer = mappings.get(ANIMA_ADAPTER_MIXER_NODE)
    if not isinstance(pack, type) or not isinstance(mixer, type):
        return None
    return pack, mixer


def anima_artist_mixer_available() -> bool:
    return _resolve_anima_mixer_nodes() is not None


def route_artist_prompt(
    *,
    model: Any,
    clip: Any,
    mixer_base_prompt: str,
    fallback_prompt: str,
    artist_entries: Sequence[tuple[str, float]],
    artist_template: ArtistTagTemplate | None = None,
    mixer_config: AnimaArtistMixerConfig | None = None,
) -> ArtistPromptRoute:
    template = artist_template or artist_template_for_model(model)
    expanded_entries = tuple(
        (tag, float(weight))
        for value, weight in artist_entries
        for tag in split_artist_tags(value)
    )
    rendered_tags = tuple(template.format(tag, weight) for tag, weight in expanded_entries)
    if detect_model_family(model) != MODEL_FAMILY_ANIMA or len(rendered_tags) <= 1:
        return ArtistPromptRoute(
            model=model,
            positive=_encode_prompt(clip, fallback_prompt),
            rendered_tags=rendered_tags,
            mode="native_prompt",
        )

    mixer_nodes = _resolve_anima_mixer_nodes()
    if mixer_nodes is None:
        logger.warning(
            "Anima multi-artist test is using native prompt concatenation because "
            "Anima Artist Mixer is not loaded"
        )
        return ArtistPromptRoute(
            model=model,
            positive=_encode_prompt(clip, fallback_prompt),
            rendered_tags=rendered_tags,
            mode="native_prompt_missing_mixer",
        )

    config = mixer_config or AnimaArtistMixerConfig()
    if not isinstance(config, AnimaArtistMixerConfig):
        raise TypeError(
            "anima_mixer_config must come from an Anima Artist Mixer Configuration node"
        )
    pack_class, mixer_class = mixer_nodes
    artist_chain = "\n".join(rendered_tags)
    try:
        artist_pack = pack_class().pack(
            clip=clip,
            artist_chain=artist_chain,
            base_prompt=str(mixer_base_prompt),
        )[0]
        patched_model, positive = mixer_class().patch(
            model=model,
            artist_pack=artist_pack,
            strength=config.strength,
            normalize_weights=config.normalize_weights,
            alignment_mode=config.alignment_mode,
            enabled=config.enabled,
            apply_to_uncond=config.apply_to_uncond,
            uncond_strength=config.uncond_strength,
        )
    except Exception as error:
        raise RuntimeError(
            "Anima Artist Mixer failed while processing a multi-artist test: "
            f"{error}"
        ) from error
    return ArtistPromptRoute(
        model=patched_model,
        positive=positive,
        rendered_tags=rendered_tags,
        mode="anima_artist_mixer",
    )


__all__ = [
    "ANIMA_ADAPTER_MIXER_NODE",
    "ANIMA_ARTIST_PACK_NODE",
    "ANIMA_ARTIST_TEMPLATE",
    "ARTIST_TAG_MODE",
    "ArtistPromptRoute",
    "ArtistTagTemplate",
    "AnimaArtistMixerConfig",
    "DANBOORU_ARTIST_TEMPLATE",
    "MODEL_FAMILY_ANIMA",
    "MODEL_FAMILY_DANBOORU",
    "anima_artist_mixer_available",
    "artist_template_for_model",
    "detect_model_family",
    "extract_anima_artist_tags",
    "normalize_artist_tag",
    "parse_artist_tag_entries",
    "route_artist_prompt",
    "split_artist_tags",
]
