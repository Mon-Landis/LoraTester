from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

from .artist import (
    ARTIST_TAG_MODE,
    ArtistPromptRoute,
    ArtistTagTemplate,
    AnimaArtistMixerConfig,
    MODEL_FAMILY_ANIMA,
    anima_artist_mixer_available,
    artist_template_for_model,
    detect_model_family,
    parse_artist_tag_entries,
    route_artist_prompt,
    split_artist_tags,
)
from .comfy_adapter import pil_to_comfy_image
from .compositor import LoraComparisonCompositor, image_to_pil
from .layout import LoraSpec, RenderTask, build_layout
from .node_contract import (
    COLOR_MODE_INPUT,
    LOG_TEST_DETAILS_INPUT,
    SHOW_LORA_DETAILS_INPUT,
    USE_ANIMA_ARTIST_MIXER_INPUT,
)
from .stack import LoraStack, LoraStackItem, LoraStackList, split_lora_stack
from .stack_compositor import LoraStackMatrixCompositor
from .styles import StyleConfig, available_style_decorators


MAX_CACHED_LORAS = 3
MAX_STACK_ITEMS = 16
MAX_STACK_INPUTS = 16
DEFAULT_MAX_CANVAS_MEGAPIXELS = 150.0

logger = logging.getLogger(__name__)


def _get_lora_names() -> list[str]:
    import folder_paths

    names = list(folder_paths.get_filename_list("loras"))
    if ARTIST_TAG_MODE not in names:
        names.append(ARTIST_TAG_MODE)
    return names


def _get_sampler_names() -> Sequence[str]:
    import comfy.samplers

    return comfy.samplers.KSampler.SAMPLERS


def _get_scheduler_names() -> Sequence[str]:
    import comfy.samplers

    return comfy.samplers.KSampler.SCHEDULERS


def _resolve_lora_path(name: str) -> str:
    import folder_paths

    return folder_paths.get_full_path_or_raise("loras", name)


def _validate_active_lora_names(
    count: Any,
    maximum: int,
    names: Sequence[Any],
) -> bool | str:
    """Validate only slots that the node will consume for this execution.

    ComfyUI performs built-in combo validation for every declared widget before
    invoking a node. Dynamic nodes therefore need an explicit validator so a
    hidden, saved file from another machine does not block an otherwise valid
    lower-count workflow.
    """
    try:
        normalized_count = int(count)
    except (TypeError, ValueError):
        return f"LoRA count must be an integer between 1 and {maximum}"
    if not 1 <= normalized_count <= maximum:
        return f"LoRA count must be between 1 and {maximum}"
    for index, raw_name in enumerate(tuple(names)[:normalized_count], start=1):
        name = str(raw_name or "").strip()
        if not name:
            return f"LoRA slot {index} cannot be empty"
        if name == ARTIST_TAG_MODE:
            continue
        try:
            _resolve_lora_path(name)
        except (OSError, KeyError, TypeError, ValueError):
            return f"LoRA file for active slot {index} is unavailable: {name}"
    return True


def _load_lora_file(path: str) -> tuple[Any, Any]:
    import comfy.utils

    return comfy.utils.load_torch_file(
        path,
        safe_load=True,
        return_metadata=True,
    )


def _apply_lora_to_models(
    model: Any,
    clip: Any,
    lora: Any,
    weight: float,
    metadata: Any,
) -> tuple[Any, Any]:
    import comfy.sd

    return comfy.sd.load_lora_for_models(
        model,
        clip,
        lora,
        weight,
        weight,
        lora_metadata=metadata,
    )


def _common_ksampler(
    model: Any,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    latent: dict[str, Any],
    denoise: float,
    *,
    progress: Any = None,
    completed_tasks: int = 0,
    total_tasks: int | None = None,
) -> dict[str, Any]:
    if progress is None:
        import nodes as comfy_nodes

        return comfy_nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

    if total_tasks is None or int(total_tasks) <= 0:
        raise ValueError("total_tasks must be positive when reporting sampling progress")

    # The stock wrapper creates a second per-sample ProgressBar; use the lower-level
    # sampler here so the node can publish one monotonic overall progress stream.
    import comfy.sample
    import comfy.utils

    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    batch_indices = latent.get("batch_index")
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_indices)
    callback = _make_sampling_progress_callback(
        model,
        progress,
        completed_tasks=int(completed_tasks),
        total_tasks=int(total_tasks),
    )
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        noise_mask=latent.get("noise_mask"),
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=seed,
    )
    result = latent.copy()
    result.pop("downscale_ratio_spacial", None)
    result.pop("downscale_ratio_temporal", None)
    result["samples"] = samples
    return result


def _sampling_progress_value(completed_tasks: int, step: int, total_steps: int) -> float:
    step_count = max(1, int(total_steps))
    completed_steps = min(step_count, max(0, int(step) + 1))
    # Reserve one step-equivalent for VAE decoding and placement so 100% means finished.
    return float(completed_tasks) + completed_steps / (step_count + 1)


def _make_sampling_progress_callback(
    model: Any,
    progress: Any,
    *,
    completed_tasks: int,
    total_tasks: int,
) -> Any:
    import latent_preview

    previewer = latent_preview.get_previewer(
        model.load_device,
        model.model.latent_format,
    )

    def callback(step: int, x0: Any, _x: Any, total_steps: int) -> None:
        preview = None
        if previewer is not None:
            preview = previewer.decode_latent_to_preview_image("JPEG", x0)
        value = min(
            float(total_tasks),
            _sampling_progress_value(completed_tasks, step, total_steps),
        )
        progress.update_absolute(value, total_tasks, preview)

    return callback


def _throw_if_interrupted() -> None:
    import comfy.model_management

    comfy.model_management.throw_exception_if_processing_interrupted()


def _make_progress_bar(total: int, node_id: Any = None) -> Any:
    import comfy.utils

    return comfy.utils.ProgressBar(total, node_id=node_id)


def _decode_vae(vae: Any, samples: dict[str, Any]) -> Any:
    latent = samples["samples"]
    if getattr(latent, "is_nested", False):
        latent = latent.unbind()[0]
    images = vae.decode(latent)
    if len(images.shape) == 5:
        images = images.reshape(
            -1,
            images.shape[-3],
            images.shape[-2],
            images.shape[-1],
        )
    return images


def _encode_prompt(clip: Any, text: str) -> Any:
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def compose_positive_prompt(base_prompt: str, additions: Sequence[str]) -> str:
    parts = [str(value).strip() for value in additions if str(value).strip()]
    normalized_base = str(base_prompt).strip()
    if normalized_base:
        parts.append(normalized_base)
    return ", ".join(parts)


def _join_prompt_parts(*parts: str) -> str:
    return ", ".join(str(part).strip() for part in parts if str(part).strip())


def _route_prompt_entries(
    *,
    model: Any,
    clip: Any,
    prefix_parts: Sequence[str],
    entries: Sequence[tuple[bool, str, float]],
    suffix_parts: Sequence[str],
    artist_template: ArtistTagTemplate | None,
    mixer_config: AnimaArtistMixerConfig | None,
    independent_artist_tags: str = "",
    use_anima_artist_mixer: bool = True,
) -> ArtistPromptRoute:
    if artist_template is not None and not isinstance(artist_template, ArtistTagTemplate):
        raise TypeError("artist_tag_template must come from an Artist Tag Template node")
    template = artist_template or artist_template_for_model(model)
    fallback_additions: list[str] = []
    lora_additions: list[str] = []
    artist_entries: list[tuple[str, float]] = []
    for is_artist, value, weight in entries:
        if math.isclose(float(weight), 0.0, abs_tol=1e-12):
            continue
        if is_artist:
            for tag in split_artist_tags(value):
                artist_entries.append((tag, float(weight)))
                fallback_additions.append(template.format(tag, float(weight)))
            continue
        trigger = str(value).strip()
        if trigger:
            fallback_additions.append(trigger)
            lora_additions.append(trigger)

    independent_entries = parse_artist_tag_entries(independent_artist_tags)
    artist_entries.extend(independent_entries)
    fallback_additions.extend(
        template.format(tag, weight) for tag, weight in independent_entries
    )

    fallback_prompt = _join_prompt_parts(
        *prefix_parts,
        *fallback_additions,
        *suffix_parts,
    )
    mixer_base_prompt = _join_prompt_parts(
        *prefix_parts,
        *lora_additions,
        *suffix_parts,
    )
    return route_artist_prompt(
        model=model,
        clip=clip,
        mixer_base_prompt=mixer_base_prompt,
        fallback_prompt=fallback_prompt,
        artist_entries=artist_entries,
        artist_template=template,
        mixer_config=mixer_config,
        use_anima_artist_mixer=use_anima_artist_mixer,
    )


@dataclass(frozen=True, slots=True)
class _CombinationPreflight:
    model_family: str
    has_artist_tags: bool
    has_multi_artist_tests: bool
    mixer_available: bool
    mixer_switch_enabled: bool
    mixer_enabled: bool
    mixer_active: bool
    mixer_combinations: tuple[str, ...]


def _combination_preflight(
    model: Any,
    stacks: Sequence[LoraStack],
    artist_template: ArtistTagTemplate | None,
    mixer_config: AnimaArtistMixerConfig | None,
    use_anima_artist_mixer: bool = True,
) -> _CombinationPreflight:
    family = detect_model_family(model)
    config = mixer_config or AnimaArtistMixerConfig()
    if not isinstance(config, AnimaArtistMixerConfig):
        raise TypeError(
            "anima_mixer_config must come from an Anima Artist Mixer Configuration node"
        )
    combinations: list[str] = []
    has_artist_tags = False
    for stack in stacks:
        entries = stack.artist_entries
        has_artist_tags = has_artist_tags or bool(entries)
        if len(entries) <= 1:
            continue
        template = artist_template or stack.artist_template or artist_template_for_model(model)
        rendered = " + ".join(template.format(tag, weight) for tag, weight in entries)
        if rendered not in combinations:
            combinations.append(rendered)
    available = anima_artist_mixer_available()
    switch_enabled = bool(use_anima_artist_mixer)
    enabled = bool(switch_enabled and config.enabled and config.strength > 0.0)
    multi_artist = bool(combinations)
    return _CombinationPreflight(
        model_family=family,
        has_artist_tags=has_artist_tags,
        has_multi_artist_tests=multi_artist,
        mixer_available=available,
        mixer_switch_enabled=switch_enabled,
        mixer_enabled=enabled,
        mixer_active=bool(
            family == MODEL_FAMILY_ANIMA and multi_artist and available and enabled
        ),
        mixer_combinations=tuple(combinations if family == MODEL_FAMILY_ANIMA else ()),
    )


def _stack_log_details(stack: LoraStack | None) -> tuple[str, str, str]:
    if stack is None:
        return "BASE", "[]", "[]"
    loras = [
        f"{item.display_name}@{float(item.strength):g}"
        for item in stack.items
        if not item.is_artist_tag
    ]
    artists = [
        f"{tag}@{float(weight):g}" for tag, weight in stack.artist_entries
    ]
    return stack.label, f"[{', '.join(loras)}]", f"[{', '.join(artists)}]"


def _format_log_weight(value: float) -> str:
    return f"{float(value):g}"


def _log_test_usage(
    *,
    label: str,
    loras: Sequence[tuple[str, float, str, str]],
    artists: Sequence[tuple[str, float, str]],
    rendered_tags: Sequence[str],
    route: ArtistPromptRoute,
    model_cache: str | None = None,
) -> None:
    """Emit one readable, opt-in-by-caller audit record for a comparison cell."""
    lines = [f"[LoraTester] {label}", f"  Route: {route.mode}"]
    if model_cache:
        lines.append(f"  Patched model: {model_cache}")
    lines.append("  LoRA files:")
    if loras:
        for name, weight, cache_status, trigger in loras:
            trigger_suffix = f' | trigger="{trigger}"' if trigger else ""
            lines.append(
                f'    - "{name}" | weight={_format_log_weight(weight)} | '
                f"cache={cache_status}{trigger_suffix}"
            )
    else:
        lines.append("    - none")
    lines.append("  Artist tags:")
    if artists:
        for index, (tag, weight, cache_status) in enumerate(artists):
            rendered = rendered_tags[index] if index < len(rendered_tags) else ""
            rendered_suffix = f' | rendered="{rendered}"' if rendered else ""
            lines.append(
                f'    - "{tag}" | weight={_format_log_weight(weight)} | '
                f"cache={cache_status}{rendered_suffix}"
            )
    else:
        lines.append("    - none")
    logger.info("%s", "\n".join(lines))


def _log_combination_preflight(preflight: _CombinationPreflight) -> None:
    lines = [
        "[LoraTester] Combination test preflight",
        f"  Model family: {preflight.model_family}",
        f"  Artist tags present: {'yes' if preflight.has_artist_tags else 'no'}",
        f"  Multi-artist tests: {'yes' if preflight.has_multi_artist_tests else 'no'}",
        f"  Anima Artist Mixer available: {'yes' if preflight.mixer_available else 'no'}",
        f"  Anima Artist Mixer switch: {'on' if preflight.mixer_switch_enabled else 'off'}",
        f"  Anima Artist Mixer enabled: {'yes' if preflight.mixer_enabled else 'no'}",
        f"  Anima Artist Mixer active: {'yes' if preflight.mixer_active else 'no'}",
        "  Mixer combinations:",
    ]
    if preflight.mixer_combinations:
        lines.extend(f"    - {combination}" for combination in preflight.mixer_combinations)
    else:
        lines.append("    - none")
    logger.info("%s", "\n".join(lines))


def _validate_single_latent(latent_image: dict[str, Any]) -> None:
    if not isinstance(latent_image, dict) or "samples" not in latent_image:
        raise ValueError("latent_image must be a ComfyUI LATENT containing 'samples'")
    samples = latent_image["samples"]
    if getattr(samples, "is_nested", False):
        nested_items = samples.unbind()
        if len(nested_items) != 1:
            raise ValueError(
                "LoRA Tester requires a latent batch size of 1; nested latent contains "
                f"{len(nested_items)} items"
            )
        return
    shape = getattr(samples, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError("latent_image samples must expose a batch dimension")
    if int(shape[0]) != 1:
        raise ValueError(
            "LoRA Tester requires a latent batch size of 1 because it produces one large "
            f"comparison sheet; received batch size {int(shape[0])}"
        )


def _style_for_mode(color_mode: str, custom_style: StyleConfig | None) -> StyleConfig:
    if color_mode == "black":
        return StyleConfig.black()
    if color_mode == "white":
        return StyleConfig.white()
    if color_mode == "custom":
        if custom_style is not None and not isinstance(custom_style, StyleConfig):
            raise TypeError("custom_style must come from a LoRA Tester Style node")
        return custom_style or StyleConfig.custom()
    raise ValueError(f"Unknown color mode: {color_mode!r}")


@dataclass(frozen=True, slots=True)
class _CachedLora:
    path: str
    state_dict: Any
    metadata: Any
    file_signature: tuple[int, int, int] | None


def _lora_file_signature(path: str) -> tuple[int, int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))


def _lora_input_fingerprint(names: Sequence[Any]) -> tuple[tuple[str, Any], ...]:
    fingerprint: list[tuple[str, Any]] = []
    for value in names:
        name = str(value or "").strip()
        if not name or name == ARTIST_TAG_MODE:
            continue
        try:
            path = _resolve_lora_path(name)
        except (OSError, KeyError, TypeError, ValueError):
            fingerprint.append((name, "missing"))
            continue
        normalized_path = os.path.normcase(os.path.abspath(path))
        fingerprint.append((normalized_path, _lora_file_signature(path)))
    return tuple(fingerprint)


def _bounded_count(value: Any, maximum: int) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 1
    return min(maximum, max(1, count))


def _with_run_local_lora_cache(function: Any) -> Any:
    @wraps(function)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        self._lora_cache.clear()
        try:
            return function(self, *args, **kwargs)
        finally:
            # State dicts are CPU tensors; never retain them after this execution.
            self._lora_cache.clear()

    return wrapped


class LoraTesterSampler:
    LORA_CACHE_LIMIT = MAX_CACHED_LORAS

    def __init__(self) -> None:
        self._lora_cache: OrderedDict[str, _CachedLora] = OrderedDict()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        lora_names = _get_lora_names()
        lora_name_input = (
            lora_names,
            {
                "tooltip": "LoRA file for this slot. Slots after lora_count are ignored.",
            },
        )
        trigger_input = (
            "STRING",
            {
                "default": "",
                "multiline": False,
                "tooltip": "Prepended to the positive prompt whenever this LoRA is active.",
            },
        )
        strength_input = (
            "FLOAT",
            {
                "default": 1.0,
                "min": -100.0,
                "max": 100.0,
                "step": 0.01,
                "round": 0.001,
                "tooltip": "Maximum model and CLIP strength. Tests use 25%, 50%, 75%, and 100%.",
            },
        )
        min_strength_input = (
            "FLOAT",
            {
                "default": 0.0,
                "min": -100.0,
                "max": 100.0,
                "step": 0.01,
                "round": 0.001,
                "tooltip": "Minimum model and CLIP strength. Must be lower than the maximum.",
            },
        )
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "latent_image": ("LATENT",),
                "positive_prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": True},
                ),
                "independent_artist_tags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": (
                            "Artist-only tags parsed separately for optional Anima mixing. "
                            "Normal positive prompts and LoRA triggers are never extracted."
                        ),
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": True},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
                ),
                "sampler_name": (_get_sampler_names(),),
                "scheduler": (_get_scheduler_names(),),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "lora_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 3,
                        "step": 1,
                        "tooltip": "Use the first one, two, or three configured LoRA slots.",
                    },
                ),
                "lora_a_name": lora_name_input,
                "lora_a_trigger": trigger_input,
                "lora_a_min_strength": min_strength_input,
                "lora_a_max_strength": strength_input,
                "lora_b_name": lora_name_input,
                "lora_b_trigger": trigger_input,
                "lora_b_min_strength": min_strength_input,
                "lora_b_max_strength": strength_input,
                "lora_c_name": lora_name_input,
                "lora_c_trigger": trigger_input,
                "lora_c_min_strength": min_strength_input,
                "lora_c_max_strength": strength_input,
                "color_mode": COLOR_MODE_INPUT,
                "show_lora_details": SHOW_LORA_DETAILS_INPUT,
                "log_test_details": LOG_TEST_DETAILS_INPUT,
                "use_anima_artist_mixer": USE_ANIMA_ARTIST_MIXER_INPUT,
                "max_canvas_megapixels": (
                    "FLOAT",
                    {
                        "default": DEFAULT_MAX_CANVAS_MEGAPIXELS,
                        "min": 1.0,
                        "max": 1000.0,
                        "step": 1.0,
                        "advanced": True,
                        "tooltip": "Stops the run if the final RGB comparison sheet would be larger.",
                    },
                ),
            },
            "optional": {
                "custom_style": (
                    "LORA_TESTER_STYLE",
                    {"tooltip": "Used only when color_mode is custom."},
                ),
                "artist_tag_template": (
                    "ARTIST_TAG_TEMPLATE",
                    {"tooltip": "Overrides the model-specific artist tag syntax."},
                ),
                "anima_mixer_config": (
                    "ANIMA_ARTIST_MIXER_CONFIG",
                    {
                        "tooltip": (
                            "Overrides Anima Artist Adapter Mixer settings for multi-artist "
                            "Anima tests when that custom node is loaded."
                        )
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison_sheet",)
    OUTPUT_TOOLTIPS = ("The decoded LoRA comparison sheet.",)
    FUNCTION = "sample"
    CATEGORY = "Lora Tester"
    DESCRIPTION = (
        "Tests one to three LoRAs with fixed sampling inputs, decodes every result, and "
        "assembles a labeled comparison sheet."
    )

    @classmethod
    def IS_CHANGED(cls, lora_count: int = 1, **values: Any) -> Any:
        count = _bounded_count(lora_count, 3)
        return _lora_input_fingerprint(
            values.get(f"lora_{slot}_name", "")
            for slot in ("a", "b", "c")[:count]
        )

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        lora_count: int = 1,
        lora_a_name: str = "",
        lora_b_name: str = "",
        lora_c_name: str = "",
    ) -> bool | str:
        return _validate_active_lora_names(
            lora_count,
            3,
            (lora_a_name, lora_b_name, lora_c_name),
        )

    @_with_run_local_lora_cache
    def sample(
        self,
        model: Any,
        clip: Any,
        vae: Any,
        latent_image: dict[str, Any],
        positive_prompt: str,
        independent_artist_tags: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        lora_count: int,
        lora_a_name: str,
        lora_a_trigger: str,
        lora_a_min_strength: float,
        lora_a_max_strength: float,
        lora_b_name: str,
        lora_b_trigger: str,
        lora_b_min_strength: float,
        lora_b_max_strength: float,
        lora_c_name: str,
        lora_c_trigger: str,
        lora_c_min_strength: float,
        lora_c_max_strength: float,
        color_mode: str,
        show_lora_details: bool,
        log_test_details: bool = True,
        use_anima_artist_mixer: bool = True,
        max_canvas_megapixels: float = DEFAULT_MAX_CANVAS_MEGAPIXELS,
        custom_style: StyleConfig | None = None,
        artist_tag_template: ArtistTagTemplate | None = None,
        anima_mixer_config: AnimaArtistMixerConfig | None = None,
        unique_id: Any = None,
    ) -> tuple[Any]:
        _validate_single_latent(latent_image)
        specs = self._make_specs(
            lora_count,
            (
                (lora_a_name, lora_a_max_strength, lora_a_trigger, lora_a_min_strength),
                (lora_b_name, lora_b_max_strength, lora_b_trigger, lora_b_min_strength),
                (lora_c_name, lora_c_max_strength, lora_c_trigger, lora_c_min_strength),
            ),
        )
        plan = build_layout(specs)
        style = _style_for_mode(color_mode, custom_style)
        load_results = tuple(
            (None, False) if spec.is_artist_tag else self._load_lora_with_status(spec.name)
            for spec in specs
        )
        loaded_loras = tuple(result[0] for result in load_results)
        lora_cache_status = tuple(
            "run-local:miss" if result[1] is False else "run-local:hit"
            for result in load_results
        )
        progress = _make_progress_bar(plan.unique_task_count, node_id=unique_id)

        if bool(log_test_details):
            logger.info(
                "[LoraTester] Direct test run | model_family=%s | cells=%d | "
                "artist_mixer_available=%s",
                detect_model_family(model),
                plan.unique_task_count,
                anima_artist_mixer_available(),
            )

        session = None
        used_lora_names: set[str] = set()
        for index, task in enumerate(plan.tasks, start=1):
            _throw_if_interrupted()
            task_model, task_clip = self._apply_task_loras(
                model,
                clip,
                task,
                loaded_loras,
            )
            route = _route_prompt_entries(
                model=task_model,
                clip=task_clip,
                prefix_parts=(),
                entries=tuple(
                    (spec.is_artist_tag, spec.trigger_word, weight)
                    for spec, weight in zip(specs, task.weights)
                ),
                suffix_parts=(positive_prompt,),
                artist_template=artist_tag_template,
                mixer_config=anima_mixer_config,
                independent_artist_tags=independent_artist_tags,
                use_anima_artist_mixer=use_anima_artist_mixer,
            )
            task_model = route.model
            positive = route.positive
            active_lora_values: list[tuple[str, float, str, str]] = []
            for slot_index, (spec, weight) in enumerate(zip(specs, task.weights)):
                if spec.is_artist_tag or math.isclose(float(weight), 0.0, abs_tol=1e-12):
                    continue
                cache_status = lora_cache_status[slot_index]
                if spec.name in used_lora_names:
                    cache_status = "run-local:hit"
                active_lora_values.append(
                    (spec.name, float(weight), cache_status, spec.trigger_word)
                )
                used_lora_names.add(spec.name)
            active_loras = tuple(active_lora_values)
            active_artists: list[tuple[str, float, str]] = []
            for spec, weight in zip(specs, task.weights):
                if not spec.is_artist_tag or math.isclose(float(weight), 0.0, abs_tol=1e-12):
                    continue
                active_artists.extend(
                    (tag, float(weight), "external:lazy-per-sample" if route.used_external_mixer else "none")
                    for tag in split_artist_tags(spec.trigger_word)
                )
            active_artists.extend(
                (tag, float(weight), "external:lazy-per-sample" if route.used_external_mixer else "none")
                for tag, weight in parse_artist_tag_entries(independent_artist_tags)
            )
            if bool(log_test_details):
                _log_test_usage(
                    label=(
                        f"Direct test image {index}/{plan.unique_task_count} "
                        f"| task={task.task_id}"
                    ),
                    loras=active_loras,
                    artists=tuple(active_artists),
                    rendered_tags=route.rendered_tags,
                    route=route,
                    model_cache="recomputed from base model",
                )
            negative = _encode_prompt(task_clip, negative_prompt)
            sampled = _common_ksampler(
                task_model,
                int(seed),
                int(steps),
                float(cfg),
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                float(denoise),
                progress=progress,
                completed_tasks=index - 1,
                total_tasks=plan.unique_task_count,
            )
            decoded = _decode_vae(vae, sampled)
            decoded_image = image_to_pil(decoded)

            if session is None:
                compositor = LoraComparisonCompositor(
                    specs,
                    decoded_image.width,
                    decoded_image.height,
                    show_lora_details=bool(show_lora_details),
                    style=style,
                    image_fit="strict",
                    max_canvas_pixels=max(1, round(float(max_canvas_megapixels) * 1_000_000)),
                )
                session = compositor.start()
            session.submit(decoded_image, task_id=task.task_id)
            progress.update_absolute(index, plan.unique_task_count)

            del route, task_model, task_clip, positive, negative, sampled, decoded, decoded_image

        if session is None:
            raise RuntimeError("LoRA Tester generated no render tasks")
        return (pil_to_comfy_image(session.finalize(strict=True)),)

    @staticmethod
    def _make_specs(
        lora_count: int,
        values: Sequence[tuple[str, float, str, float]],
    ) -> tuple[LoraSpec, ...]:
        count = int(lora_count)
        if count < 1 or count > 3:
            raise ValueError(f"lora_count must be between 1 and 3; received {lora_count!r}")
        return tuple(
            LoraSpec(
                name=name,
                max_weight=float(max_strength),
                trigger_word=trigger,
                min_weight=float(min_strength),
            )
            for name, max_strength, trigger, min_strength in values[:count]
        )

    def _load_lora(self, name: str) -> _CachedLora:
        return self._load_lora_with_status(name)[0]

    def _load_lora_with_status(self, name: str) -> tuple[_CachedLora, bool]:
        if not str(name).strip():
            raise ValueError(
                "No LoRA file is selected. Add a LoRA to ComfyUI's models/loras directory "
                "and refresh the node list."
            )
        path = _resolve_lora_path(name)
        cache_key = os.path.normcase(os.path.abspath(path))
        file_signature = _lora_file_signature(path)
        cached = self._lora_cache.pop(cache_key, None)
        if cached is not None and cached.file_signature == file_signature:
            self._lora_cache[cache_key] = cached
            return cached, True

        state_dict, metadata = _load_lora_file(path)
        cached = _CachedLora(
            path=path,
            state_dict=state_dict,
            metadata=metadata,
            file_signature=_lora_file_signature(path),
        )
        self._lora_cache[cache_key] = cached
        while len(self._lora_cache) > int(getattr(self, "LORA_CACHE_LIMIT", MAX_CACHED_LORAS)):
            self._lora_cache.popitem(last=False)
        return cached, False

    @staticmethod
    def _apply_task_loras(
        base_model: Any,
        base_clip: Any,
        task: RenderTask,
        loaded_loras: Sequence[_CachedLora | None],
    ) -> tuple[Any, Any]:
        task_model = base_model
        task_clip = base_clip
        for weight, loaded in zip(task.weights, loaded_loras):
            if loaded is None or math.isclose(weight, 0.0, abs_tol=1e-12):
                continue
            task_model, task_clip = _apply_lora_to_models(
                task_model,
                task_clip,
                loaded.state_dict,
                float(weight),
                loaded.metadata,
            )
        return task_model, task_clip


class LoraTesterStyleNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        color = lambda default, tooltip: (  # noqa: E731
            "STRING",
            {"default": default, "multiline": False, "tooltip": tooltip},
        )
        return {
            "required": {
                "background_color": color("#111416", "Canvas background color."),
                "panel_color": color("#1B2022", "Footer and information panel color."),
                "placeholder_color": color("#252A2C", "Color of image slots before they are filled."),
                "text_color": color("#F2F4F3", "Primary axis and footer text color."),
                "muted_text_color": color("#9AA3A6", "Secondary text color."),
                "frame_color": color("#687175", "Region and structural frame color."),
                "lora_a_color": color("#C8F04B", "Accent color for LoRA A."),
                "lora_b_color": color("#31BFC4", "Accent color for LoRA B."),
                "lora_c_color": color("#F0785A", "Accent color for LoRA C."),
                "background_fit": (["cover", "contain", "stretch", "tile"], {"default": "cover"}),
                "background_opacity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "decorator": (available_style_decorators(), {"default": "technical"}),
                "outer_margin": ("INT", {"default": 24, "min": 0, "max": 1024, "step": 1}),
                "cell_gap": ("INT", {"default": 4, "min": 0, "max": 512, "step": 1}),
                "region_gap": ("INT", {"default": 36, "min": 0, "max": 1024, "step": 1}),
                "footer_gap": ("INT", {"default": 18, "min": 0, "max": 1024, "step": 1}),
                "footer_padding": ("INT", {"default": 14, "min": 0, "max": 512, "step": 1}),
                "frame_width": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
                "cell_frame_width": ("INT", {"default": 1, "min": 0, "max": 64, "step": 1}),
                "font_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "advanced": True,
                        "tooltip": "Optional local .ttf/.ttc/.otf path. Empty uses an automatic system font.",
                    },
                ),
                "font_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "advanced": True,
                        "tooltip": "Axis title size. 0 selects a size from the full sheet dimensions.",
                    },
                ),
                "small_font_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "advanced": True,
                        "tooltip": "Axis value size. 0 selects a proportional automatic size.",
                    },
                ),
                "show_axis_labels": ("BOOLEAN", {"default": True}),
                "show_region_frames": ("BOOLEAN", {"default": True}),
                "show_special_cell_labels": ("BOOLEAN", {"default": True}),
                "show_cell_captions": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "advanced": True,
                        "tooltip": "Show a weight caption above every occupied image cell.",
                    },
                ),
                "show_region_labels": ("BOOLEAN", {"default": False, "advanced": True}),
                "show_coordinates": ("BOOLEAN", {"default": False, "advanced": True}),
            },
            "optional": {
                "background_image": (
                    "IMAGE",
                    {"tooltip": "Optional single image used behind the comparison matrix."},
                ),
            },
        }

    RETURN_TYPES = ("LORA_TESTER_STYLE",)
    RETURN_NAMES = ("style",)
    FUNCTION = "build_style"
    CATEGORY = "Lora Tester"
    DESCRIPTION = "Builds an optional custom visual style for the LoRA Tester sampler."

    def build_style(
        self,
        background_color: str,
        panel_color: str,
        placeholder_color: str,
        text_color: str,
        muted_text_color: str,
        frame_color: str,
        lora_a_color: str,
        lora_b_color: str,
        lora_c_color: str,
        background_fit: str,
        background_opacity: float,
        decorator: str,
        outer_margin: int,
        cell_gap: int,
        region_gap: int,
        footer_gap: int,
        footer_padding: int,
        frame_width: int,
        cell_frame_width: int,
        font_path: str,
        font_size: int,
        small_font_size: int,
        show_axis_labels: bool,
        show_region_frames: bool,
        show_special_cell_labels: bool,
        show_cell_captions: bool,
        show_region_labels: bool,
        show_coordinates: bool,
        background_image: Any = None,
    ) -> tuple[StyleConfig]:
        if background_image is not None:
            shape = getattr(background_image, "shape", None)
            if shape is not None and len(shape) == 4 and int(shape[0]) != 1:
                raise ValueError(
                    "LoRA Tester Style accepts one background image; "
                    f"received a batch of {int(shape[0])}"
                )
        style = StyleConfig.custom(
            background_color=background_color,
            panel_color=panel_color,
            placeholder_color=placeholder_color,
            text_color=text_color,
            muted_text_color=muted_text_color,
            frame_color=frame_color,
            accent_colors=(lora_a_color, lora_b_color, lora_c_color),
            background_image=background_image,
            background_fit=background_fit,
            background_opacity=float(background_opacity),
            decorator=decorator,
            outer_margin=int(outer_margin),
            cell_gap=int(cell_gap),
            region_gap=int(region_gap),
            footer_gap=int(footer_gap),
            footer_padding=int(footer_padding),
            frame_width=int(frame_width),
            cell_frame_width=int(cell_frame_width),
            font_path=font_path.strip() or None,
            font_size=int(font_size) or None,
            small_font_size=int(small_font_size) or None,
            show_axis_labels=bool(show_axis_labels),
            show_region_frames=bool(show_region_frames),
            show_special_cell_labels=bool(show_special_cell_labels),
            show_cell_captions=bool(show_cell_captions),
            show_region_labels=bool(show_region_labels),
            show_coordinates=bool(show_coordinates),
        )
        return (style,)


class ArtistTagTemplateNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "default_template": (
                    "STRING",
                    {
                        "default": "@{tag}",
                        "multiline": False,
                        "tooltip": "Artist syntax used when the item weight is exactly 1.",
                    },
                ),
                "weighted_template": (
                    "STRING",
                    {
                        "default": "(@{tag}:{weight})",
                        "multiline": False,
                        "tooltip": "Artist syntax used for an explicit non-default weight.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("ARTIST_TAG_TEMPLATE",)
    RETURN_NAMES = ("artist_tag_template",)
    OUTPUT_TOOLTIPS = ("Artist-tag formatting rules for compatible tester nodes.",)
    FUNCTION = "build_template"
    CATEGORY = "Lora Tester/Artist Tags"
    DESCRIPTION = "Defines default and weighted artist tag expressions."

    @staticmethod
    def build_template(
        default_template: str,
        weighted_template: str,
    ) -> tuple[ArtistTagTemplate]:
        return (ArtistTagTemplate(default_template, weighted_template),)


class AnimaArtistMixerConfigNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "strength": (
                    "FLOAT",
                    {"default": 1.6, "min": 0.0, "max": 4.0, "step": 0.05},
                ),
                "normalize_weights": ("BOOLEAN", {"default": True}),
                "alignment_mode": (
                    ["base_anchored", "shared_base_ids"],
                    {"default": "base_anchored"},
                ),
                "enabled": ("BOOLEAN", {"default": True}),
                "apply_to_uncond": ("BOOLEAN", {"default": False}),
                "uncond_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("ANIMA_ARTIST_MIXER_CONFIG",)
    RETURN_NAMES = ("anima_mixer_config",)
    OUTPUT_TOOLTIPS = ("Settings passed to Anima Artist Adapter Mixer when active.",)
    FUNCTION = "build_config"
    CATEGORY = "Lora Tester/Artist Tags"
    DESCRIPTION = (
        "Configures the optional Anima Artist Adapter Mixer integration. "
        "It is ignored for non-Anima models and native single-artist tests."
    )

    @staticmethod
    def build_config(
        strength: float,
        normalize_weights: bool,
        alignment_mode: str,
        enabled: bool,
        apply_to_uncond: bool,
        uncond_strength: float,
    ) -> tuple[AnimaArtistMixerConfig]:
        return (
            AnimaArtistMixerConfig(
                strength=strength,
                normalize_weights=normalize_weights,
                alignment_mode=alignment_mode,
                enabled=enabled,
                apply_to_uncond=apply_to_uncond,
                uncond_strength=uncond_strength,
            ),
        )


def _stack_lora_name_input() -> tuple[Any, dict[str, Any]]:
    return (
        _get_lora_names(),
        {
            "tooltip": "LoRA file or Artist Tag Mode for this stack entry.",
        },
    )


def _stack_inputs() -> dict[str, tuple[Any, dict[str, Any]]]:
    inputs: dict[str, tuple[Any, dict[str, Any]]] = {
        "lora_count": (
            "INT",
            {
                "default": 1,
                "min": 1,
                "max": MAX_STACK_ITEMS,
                "step": 1,
                "tooltip": "Number of LoRA entries in this stack.",
            },
        ),
    }
    for index in range(1, MAX_STACK_ITEMS + 1):
        inputs[f"lora_{index}_name"] = _stack_lora_name_input()
        inputs[f"lora_{index}_trigger"] = (
            "STRING",
            {
                "default": "",
                "multiline": False,
                "tooltip": f"LoRA trigger words or artist tag for entry {index}.",
            },
        )
        inputs[f"lora_{index}_strength"] = (
            "FLOAT",
            {
                "default": 1.0,
                "min": -100.0,
                "max": 100.0,
                "step": 0.01,
                "round": 0.001,
                "tooltip": f"LoRA strength or artist tag weight for entry {index}.",
            },
        )
    return inputs


class LoraStackNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": _stack_inputs(),
            "optional": {
                "artist_tag_template": (
                    "ARTIST_TAG_TEMPLATE",
                    {"tooltip": "Optional artist syntax stored with this stack."},
                )
            },
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "build_stack"
    CATEGORY = "Lora Tester/Stacks"
    DESCRIPTION = "Builds an ordered stack of LoRA files and artist tags."

    @classmethod
    def IS_CHANGED(cls, lora_count: int = 1, **values: Any) -> Any:
        count = _bounded_count(lora_count, MAX_STACK_ITEMS)
        return _lora_input_fingerprint(
            values.get(f"lora_{index}_name", "")
            for index in range(1, count + 1)
        )

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        lora_count: int = 1,
        lora_1_name: str = "",
        lora_2_name: str = "",
        lora_3_name: str = "",
        lora_4_name: str = "",
        lora_5_name: str = "",
        lora_6_name: str = "",
        lora_7_name: str = "",
        lora_8_name: str = "",
        lora_9_name: str = "",
        lora_10_name: str = "",
        lora_11_name: str = "",
        lora_12_name: str = "",
        lora_13_name: str = "",
        lora_14_name: str = "",
        lora_15_name: str = "",
        lora_16_name: str = "",
    ) -> bool | str:
        return _validate_active_lora_names(
            lora_count,
            MAX_STACK_ITEMS,
            (
                lora_1_name,
                lora_2_name,
                lora_3_name,
                lora_4_name,
                lora_5_name,
                lora_6_name,
                lora_7_name,
                lora_8_name,
                lora_9_name,
                lora_10_name,
                lora_11_name,
                lora_12_name,
                lora_13_name,
                lora_14_name,
                lora_15_name,
                lora_16_name,
            ),
        )

    def build_stack(
        self,
        lora_count: int,
        artist_tag_template: ArtistTagTemplate | None = None,
        **values: Any,
    ) -> tuple[LoraStack]:
        count = int(lora_count)
        if not 1 <= count <= MAX_STACK_ITEMS:
            raise ValueError(f"lora_count must be between 1 and {MAX_STACK_ITEMS}")
        items: list[LoraStackItem] = []
        for index in range(1, count + 1):
            name = str(values.get(f"lora_{index}_name", "")).strip()
            if not name:
                raise ValueError(f"LoRA or artist mode for stack entry {index} cannot be empty")
            items.append(
                LoraStackItem(
                    name=name,
                    trigger_word=str(values.get(f"lora_{index}_trigger", "")),
                    strength=float(values.get(f"lora_{index}_strength", 1.0)),
                )
            )
        return (LoraStack(tuple(items), artist_template=artist_tag_template),)


class LoraStackSplitterNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"lora_stack": ("LORA_STACK",)}}

    RETURN_TYPES = ("LORA_STACK_LIST",)
    RETURN_NAMES = ("lora_stack_list",)
    FUNCTION = "split_stack"
    CATEGORY = "Lora Tester/Stacks"
    DESCRIPTION = "Splits one LoRA stack into every non-empty combination."

    @staticmethod
    def split_stack(lora_stack: LoraStack) -> tuple[LoraStackList]:
        return (split_lora_stack(lora_stack),)


class LoraStackListerNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        required = {"stack_1": ("LORA_STACK",)}
        optional = {f"stack_{index}": ("LORA_STACK",) for index in range(2, MAX_STACK_INPUTS + 1)}
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("LORA_STACK_LIST",)
    RETURN_NAMES = ("lora_stack_list",)
    FUNCTION = "list_stacks"
    CATEGORY = "Lora Tester/Stacks"
    DESCRIPTION = "Collects multiple LoRA stacks into one ordered stack list."

    @staticmethod
    def list_stacks(stack_1: LoraStack | None = None, **values: Any) -> tuple[LoraStackList]:
        candidates = [stack_1]
        candidates.extend(values.get(f"stack_{index}") for index in range(2, MAX_STACK_INPUTS + 1))
        stacks = [value for value in candidates if value is not None]
        if not stacks:
            raise ValueError("Connect at least one LoRA stack")
        return (LoraStackList.merge(stacks),)

STACK_SHOW_LORA_DETAILS_INPUT = (
    "BOOLEAN",
    {
        "default": True,
        "label_on": "show LoRA names and strengths",
        "label_off": "hide LoRA names and strengths",
        "tooltip": "Show each distinct LoRA file, trigger-word, and strength configuration in the footer.",
    },
)


class MultiPromptSampleNode(LoraTesterSampler):
    """Sample a prompt-by-stack matrix while reusing one seed per run."""

    LORA_CACHE_LIMIT = MAX_CACHED_LORAS

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        prompts: dict[str, tuple[str, dict[str, Any]]] = {
            "prompt_count": (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": MAX_STACK_ITEMS,
                    "step": 1,
                    "tooltip": "Number of positive prompt rows.",
                },
            ),
            "prompt_prefix": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Shared positive prompt prefix used in every row.",
                },
            ),
            "negative_prompt": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Negative prompt used for every matrix cell.",
                },
            ),
        }
        for index in range(1, MAX_STACK_ITEMS + 1):
            prompts[f"positive_prompt_{index}"] = (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": f"Positive prompt for matrix row {index}.",
                },
            )
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "latent_image": ("LATENT",),
                "lorastacks": ("LORA_STACK_LIST",),
                **prompts,
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (_get_sampler_names(),),
                "scheduler": (_get_scheduler_names(),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_mode": COLOR_MODE_INPUT,
                "show_lora_details": STACK_SHOW_LORA_DETAILS_INPUT,
                "log_test_details": LOG_TEST_DETAILS_INPUT,
                "use_anima_artist_mixer": USE_ANIMA_ARTIST_MIXER_INPUT,
                "control_gap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "advanced": True,
                        "tooltip": "Gap after the no-LoRA control column. Zero uses at least one eighth of the image width.",
                    },
                ),
                "max_canvas_megapixels": (
                    "FLOAT",
                    {
                        "default": DEFAULT_MAX_CANVAS_MEGAPIXELS,
                        "min": 1.0,
                        "max": 1000.0,
                        "step": 1.0,
                        "advanced": True,
                    },
                ),
            },
            "optional": {
                "custom_style": ("LORA_TESTER_STYLE", {"tooltip": "Used when color_mode is custom."}),
                "artist_tag_template": (
                    "ARTIST_TAG_TEMPLATE",
                    {
                        "tooltip": (
                            "Overrides model-specific and stack-specific artist tag syntax."
                        )
                    },
                ),
                "anima_mixer_config": (
                    "ANIMA_ARTIST_MIXER_CONFIG",
                    {
                        "tooltip": (
                            "Overrides Anima Artist Adapter Mixer settings for multi-artist "
                            "Anima tests when that custom node is loaded."
                        )
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison_sheet",)
    FUNCTION = "sample"
    CATEGORY = "Lora Tester"
    DESCRIPTION = "Samples multiple positive prompt rows against LoRA stacks and builds an XY comparison sheet."

    def sample(
        self,
        model: Any,
        clip: Any,
        vae: Any,
        latent_image: dict[str, Any],
        lorastacks: LoraStackList,
        prompt_count: int,
        prompt_prefix: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        color_mode: str,
        show_lora_details: bool,
        log_test_details: bool = True,
        use_anima_artist_mixer: bool = True,
        control_gap: int = 0,
        max_canvas_megapixels: float = DEFAULT_MAX_CANVAS_MEGAPIXELS,
        custom_style: StyleConfig | None = None,
        artist_tag_template: ArtistTagTemplate | None = None,
        anima_mixer_config: AnimaArtistMixerConfig | None = None,
        unique_id: Any = None,
        **values: Any,
    ) -> tuple[Any]:
        _validate_single_latent(latent_image)
        if not isinstance(lorastacks, LoraStackList):
            raise TypeError("lorastacks must be a LoRA Stack List")
        count = int(prompt_count)
        if not 1 <= count <= MAX_STACK_ITEMS:
            raise ValueError(f"prompt_count must be between 1 and {MAX_STACK_ITEMS}")
        prompts = []
        for index in range(1, count + 1):
            prompt = str(values.get(f"positive_prompt_{index}", "")).strip()
            if not prompt:
                raise ValueError(f"Positive prompt row {index} cannot be empty")
            prompts.append(prompt)
        style = _style_for_mode(color_mode, custom_style)
        stacks = tuple(lorastacks.stacks)
        preflight = _combination_preflight(
            model,
            stacks,
            artist_tag_template,
            anima_mixer_config,
            use_anima_artist_mixer,
        )
        if bool(log_test_details):
            _log_combination_preflight(preflight)
        total_tasks = len(prompts) * (len(stacks) + 1)
        progress = _make_progress_bar(total_tasks, node_id=unique_id)
        compositor: LoraStackMatrixCompositor | None = None
        session = None
        task_index = 0
        self._lora_cache.clear()
        try:
            # Column-major execution reuses one patched model, CLIP, and negative
            # conditioning across every prompt row while preserving output coordinates.
            for column, stack in enumerate((None, *stacks)):
                column_model = model
                column_clip = clip
                prompt_entries: tuple[tuple[bool, str, float], ...] = ()
                stack_template = artist_tag_template
                column_lora_usage: list[tuple[str, float, str, str]] = []
                if stack is not None:
                    stack_template = artist_tag_template or stack.artist_template
                    prompt_entries = tuple(
                        (item.is_artist_tag, item.trigger_word, float(item.strength))
                        for item in stack.items
                    )
                    for item in stack.items:
                        if item.is_artist_tag:
                            continue
                        loaded, cache_hit = self._load_lora_with_status(item.name)
                        column_lora_usage.append(
                            (
                                item.name,
                                float(item.strength),
                                "run-local:hit" if cache_hit else "run-local:miss",
                                item.trigger_word,
                            )
                        )
                        column_model, column_clip = _apply_lora_to_models(
                            column_model,
                            column_clip,
                            loaded.state_dict,
                            float(item.strength),
                            loaded.metadata,
                        )
                negative = _encode_prompt(column_clip, negative_prompt)
                for row, prompt in enumerate(prompts):
                    _throw_if_interrupted()
                    route = _route_prompt_entries(
                        model=column_model,
                        clip=column_clip,
                        prefix_parts=(prompt_prefix,),
                        entries=prompt_entries,
                        suffix_parts=(prompt,),
                        artist_template=stack_template,
                        mixer_config=anima_mixer_config,
                        use_anima_artist_mixer=use_anima_artist_mixer,
                    )
                    task_model = route.model
                    positive = route.positive
                    stack_label, _, _ = _stack_log_details(stack)
                    if bool(log_test_details):
                        artist_cache_status = (
                            "external:lazy-per-sample"
                            if route.used_external_mixer
                            else "none"
                        )
                        artist_usage = tuple(
                            (tag, float(weight), artist_cache_status)
                            for tag, weight in (stack.artist_entries if stack is not None else ())
                        )
                        _log_test_usage(
                            label=(
                                f"Combination image {task_index + 1}/{total_tasks} "
                                f"| prompt_row={row + 1}/{len(prompts)} "
                                f"| column={column + 1}/{len(stacks) + 1} "
                                f"| stack={stack_label!r}"
                            ),
                            loras=tuple(
                                (
                                    name,
                                    weight,
                                    "run-local:hit" if row > 0 else cache_status,
                                    trigger,
                                )
                                for name, weight, cache_status, trigger in column_lora_usage
                            ),
                            artists=artist_usage,
                            rendered_tags=route.rendered_tags,
                            route=route,
                            model_cache=(
                                "column build" if row == 0 else "column reuse"
                            ),
                        )
                    sampled = _common_ksampler(
                        task_model,
                        int(seed),
                        int(steps),
                        float(cfg),
                        sampler_name,
                        scheduler,
                        positive,
                        negative,
                        latent_image,
                        float(denoise),
                        progress=progress,
                        completed_tasks=task_index,
                        total_tasks=total_tasks,
                    )
                    decoded = _decode_vae(vae, sampled)
                    decoded_image = image_to_pil(decoded)
                    if compositor is None:
                        compositor = LoraStackMatrixCompositor(
                            stacks,
                            prompts,
                            decoded_image.width,
                            decoded_image.height,
                            style=style,
                            show_stack_details=bool(show_lora_details),
                            image_fit="strict",
                            max_canvas_pixels=max(
                                1,
                                round(float(max_canvas_megapixels) * 1_000_000),
                            ),
                            control_gap=int(control_gap) or None,
                        )
                        session = compositor.start()
                    session.submit(decoded_image, coordinate=(row, column))
                    task_index += 1
                    progress.update_absolute(task_index, total_tasks)
                    del route, task_model, positive, sampled, decoded, decoded_image
                del column_model, column_clip, negative
        finally:
            # LoRA state dicts are CPU tensors. Keep only a tiny run-local LRU and
            # release it on success, error, or user interruption.
            self._lora_cache.clear()
        if session is None:
            raise RuntimeError("Multi Prompt Sample generated no images")
        return (pil_to_comfy_image(session.finalize(strict=True)),)


NODE_CLASS_MAPPINGS = {
    "LoraTesterSampler": LoraTesterSampler,
    "LoraTesterStyle": LoraTesterStyleNode,
    "ArtistTagTemplate": ArtistTagTemplateNode,
    "AnimaArtistMixerConfig": AnimaArtistMixerConfigNode,
    "LoraStack": LoraStackNode,
    "LoraStackSplitter": LoraStackSplitterNode,
    "LoraStackLister": LoraStackListerNode,
    "MultiPromptSample": MultiPromptSampleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraTesterSampler": "LoRA Tester (KSampler)",
    "LoraTesterStyle": "LoRA Tester Style",
    "ArtistTagTemplate": "Artist Tag Template",
    "AnimaArtistMixerConfig": "Anima Artist Mixer Configuration",
    "LoraStack": "LoRA Stack",
    "LoraStackSplitter": "LoRA Stack Splitter",
    "LoraStackLister": "LoRA Stack Lister",
    "MultiPromptSample": "Multi Prompt Sample",
}


__all__ = [
    "LoraTesterSampler",
    "LoraTesterStyleNode",
    "ArtistTagTemplateNode",
    "AnimaArtistMixerConfigNode",
    "LoraStackNode",
    "LoraStackSplitterNode",
    "LoraStackListerNode",
    "MultiPromptSampleNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "compose_positive_prompt",
]
