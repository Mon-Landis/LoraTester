from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from collections.abc import Sequence
from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

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
from .anima_patch import anima_remap_diagnosis, warn_missing_anima_remap
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
from .styles import StyleConfig, available_style_decorators
from .xy import (
    MAX_AXIS_ENTRIES,
    MAX_SEED,
    PromptEntry,
    PromptList,
    SeedList,
    XYAxis,
    build_lora_stack_axis,
    build_prompt_axis,
    build_seed_axis,
    merge_axis_parameters,
)
from .xy_compositor import XYMatrixCompositor


MAX_CACHED_LORAS = 3
MAX_STACK_ITEMS = 16
MAX_STACK_INPUTS = 16
DEFAULT_MAX_CANVAS_MEGAPIXELS = 150.0

logger = logging.getLogger(__name__)


XYParameterHandler = Callable[[dict[str, Any], Any], None]
_XY_PARAMETER_HANDLERS: dict[str, XYParameterHandler] = {}


def register_xy_parameter_handler(
    name: str,
    handler: XYParameterHandler,
    *,
    replace_existing: bool = False,
) -> None:
    """Register how one axis parameter overrides a sampler configuration."""

    key = str(name).strip()
    if not key:
        raise ValueError("XY parameter handler name cannot be empty")
    if not callable(handler):
        raise TypeError("XY parameter handler must be callable")
    if key in _XY_PARAMETER_HANDLERS and not replace_existing:
        raise ValueError(f"XY parameter handler is already registered: {key}")
    _XY_PARAMETER_HANDLERS[key] = handler


def _set_int_parameter(name: str, minimum: int, maximum: int) -> XYParameterHandler:
    def apply(values: dict[str, Any], raw: Any) -> None:
        value = int(raw)
        if not minimum <= value <= maximum:
            raise ValueError(f"XY parameter {name} must be between {minimum} and {maximum}")
        values[name] = value

    return apply


def _set_float_parameter(name: str, minimum: float, maximum: float) -> XYParameterHandler:
    def apply(values: dict[str, Any], raw: Any) -> None:
        value = float(raw)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"XY parameter {name} must be between {minimum:g} and {maximum:g}")
        values[name] = value

    return apply


def _set_choice_parameter(name: str) -> XYParameterHandler:
    def apply(values: dict[str, Any], raw: Any) -> None:
        value = str(raw).strip()
        if not value:
            raise ValueError(f"XY parameter {name} cannot be empty")
        values[name] = value

    return apply


def _set_prompt_parameter(values: dict[str, Any], raw: Any) -> None:
    if not isinstance(raw, PromptEntry):
        raise TypeError("XY prompt parameters must come from a Prompt Axis node")
    values["positive_prompt"] = raw.prompt
    values["prompt_prefix"] = raw.prefix
    values["prompt_suffix"] = raw.suffix
    values["independent_artist_tags"] = raw.independent_artist_tags


def _set_lora_stack_parameter(values: dict[str, Any], raw: Any) -> None:
    if not isinstance(raw, LoraStack):
        raise TypeError("XY LoRA parameters must come from a LoRA Stack Axis node")
    values["lora_stack"] = raw


register_xy_parameter_handler("prompt", _set_prompt_parameter)
register_xy_parameter_handler("seed", _set_int_parameter("seed", 0, MAX_SEED))
register_xy_parameter_handler("steps", _set_int_parameter("steps", 1, 10000))
register_xy_parameter_handler("cfg", _set_float_parameter("cfg", 0.0, 100.0))
register_xy_parameter_handler("denoise", _set_float_parameter("denoise", 0.0, 1.0))
register_xy_parameter_handler("sampler_name", _set_choice_parameter("sampler_name"))
register_xy_parameter_handler("scheduler", _set_choice_parameter("scheduler"))
register_xy_parameter_handler("lora_stack", _set_lora_stack_parameter)


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


def _missing_anima_remap_ui(diagnosis: Mapping[str, Any]) -> dict[str, Any] | None:
    if not diagnosis.get("required"):
        return None
    return {
        "lora_tester_anima_remap": [{
            "message": (
                "A 28-block Anima LoRA is applied to a 40-block Anima 2.9B model "
                "without ComfyUI-Anima-2.9B-loraPatch."
            ),
        }],
    }


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


def _release_temporary_model(value: Any, base_value: Any) -> None:
    """Release a per-test MODEL/CLIP clone through ComfyUI's model manager.

    DynamicVRAM shares pinned host buffers between patcher clones.  The normal
    execution boundary eventually releases those clones, but a long-running
    comparison node creates many of them inside one execution.  Releasing each
    finished clone keeps its host-buffer pins bounded without touching global
    CUDA caches or passing the caller's base model as the unload target.
    ComfyUI still owns clone-group eviction and may evict another loaded member
    of that group as needed.
    """
    if value is None or value is base_value:
        return
    patcher = getattr(value, "patcher", value)
    base_patcher = getattr(base_value, "patcher", base_value)
    if patcher is base_patcher or not hasattr(patcher, "clone_base_uuid"):
        return
    try:
        import comfy.model_management

        unload = getattr(comfy.model_management, "unload_model_and_clones", None)
        if unload is not None:
            unload(patcher)
    except (ImportError, AttributeError, RuntimeError, TypeError):
        # Cleanup must not turn a completed image test into a node failure.
        logger.debug("[LoraTester] temporary model cleanup was unavailable", exc_info=True)


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


def _direct_mixer_labels_possible(
    model: Any,
    specs: Sequence[LoraSpec],
    independent_artist_tags: str,
    mixer_config: AnimaArtistMixerConfig | None,
    use_anima_artist_mixer: bool,
) -> bool:
    """Reserve compositor label space when a direct test can route through the mixer."""
    if detect_model_family(model) != MODEL_FAMILY_ANIMA:
        return False
    if not bool(use_anima_artist_mixer):
        return False
    config = mixer_config or AnimaArtistMixerConfig()
    if not isinstance(config, AnimaArtistMixerConfig):
        raise TypeError(
            "anima_mixer_config must come from an Anima Artist Mixer Configuration node"
        )
    if not config.enabled or config.strength <= 0.0:
        return False
    artist_count = sum(
        len(split_artist_tags(spec.trigger_word))
        for spec in specs
        if spec.is_artist_tag
    )
    artist_count += len(parse_artist_tag_entries(independent_artist_tags))
    if artist_count <= 1:
        return False
    return anima_artist_mixer_available()


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
    independent_artist_tags: str = "",
) -> _CombinationPreflight:
    family = detect_model_family(model)
    config = mixer_config or AnimaArtistMixerConfig()
    if not isinstance(config, AnimaArtistMixerConfig):
        raise TypeError(
            "anima_mixer_config must come from an Anima Artist Mixer Configuration node"
        )
    combinations: list[str] = []
    independent_entries = parse_artist_tag_entries(independent_artist_tags)
    has_artist_tags = bool(independent_entries)

    def add_combination(
        entries: Sequence[tuple[str, float]],
        template: ArtistTagTemplate,
    ) -> None:
        if len(entries) <= 1:
            return
        rendered = " + ".join(template.format(tag, weight) for tag, weight in entries)
        if rendered not in combinations:
            combinations.append(rendered)

    add_combination(
        independent_entries,
        artist_template or artist_template_for_model(model),
    )
    for stack in stacks:
        entries = (*stack.artist_entries, *independent_entries)
        has_artist_tags = has_artist_tags or bool(stack.artist_entries)
        template = artist_template or stack.artist_template or artist_template_for_model(model)
        add_combination(entries, template)
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
        anima_diagnosis = anima_remap_diagnosis(
            model=model,
            state_dicts=tuple(result[0].state_dict for result in load_results),
        )
        warn_missing_anima_remap(anima_diagnosis, unique_id=unique_id)
        lora_cache_status = tuple(
            "run-local:miss" if result[1] is False else "run-local:hit"
            for result in load_results
        )
        progress = _make_progress_bar(plan.unique_task_count, node_id=unique_id)
        reserve_artist_mixer_labels = _direct_mixer_labels_possible(
            model,
            specs,
            independent_artist_tags,
            anima_mixer_config,
            use_anima_artist_mixer,
        )

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
            applied_model = applied_clip = task_model = task_clip = None
            route = positive = negative = sampled = decoded = decoded_image = None
            try:
                applied_model, applied_clip = self._apply_task_loras(
                    model,
                    clip,
                    task,
                    loaded_loras,
                )
                task_model = applied_model
                task_clip = applied_clip
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
                        reserve_artist_mixer_labels=reserve_artist_mixer_labels,
                    )
                    session = compositor.start()
                session.submit(
                    decoded_image,
                    task_id=task.task_id,
                    artist_mixer=route.used_external_mixer,
                )
                progress.update_absolute(index, plan.unique_task_count)
            finally:
                # A route can replace the LoRA-patched model with an external
                # mixer clone, so release both objects while the local references
                # still identify them.  The base MODEL/CLIP are never passed as
                # unload targets; clone-group eviction remains ComfyUI-owned.
                _release_temporary_model(task_model, model)
                if applied_model is not task_model:
                    _release_temporary_model(applied_model, model)
                _release_temporary_model(task_clip, clip)
                if applied_clip is not task_clip:
                    _release_temporary_model(applied_clip, clip)
                applied_model = applied_clip = task_model = task_clip = None
                route = positive = negative = sampled = decoded = decoded_image = None

        if session is None:
            raise RuntimeError("LoRA Tester generated no render tasks")
        output = pil_to_comfy_image(session.finalize(strict=True))
        remap_ui = _missing_anima_remap_ui(anima_diagnosis)
        if remap_ui:
            return {"ui": remap_ui, "result": (output,)}
        return (output,)

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


def _resolve_xy_values(
    base_values: dict[str, Any],
    x_entry: Any,
    y_entry: Any,
) -> dict[str, Any]:
    values = dict(base_values)
    for name, raw in merge_axis_parameters(x_entry, y_entry).items():
        try:
            handler = _XY_PARAMETER_HANDLERS[name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported XY parameter {name!r}; register a parameter handler before sampling"
            ) from exc
        handler(values, raw)
    return values


def _validate_xy_axis_conflicts(x_axis: XYAxis, y_axis: XYAxis) -> None:
    conflicts = sorted(x_axis.parameter_names & y_axis.parameter_names)
    if conflicts:
        joined = ", ".join(repr(value) for value in conflicts)
        raise ValueError(
            "X and Y axes cannot both modify the same sampling parameter: " + joined
        )


def _warn_xy_scale(
    latent_image: dict[str, Any],
    x_axis: XYAxis,
    y_axis: XYAxis,
    max_canvas_megapixels: float,
) -> None:
    cell_count = len(x_axis.entries) * len(y_axis.entries)
    if len(x_axis.entries) > 32 or len(y_axis.entries) > 32 or cell_count > 128:
        logger.warning(
            "[LoraTester] XY axis contains %d x %d = %d cells; large axes increase "
            "sampling time and raw IMAGE memory.",
            len(x_axis.entries),
            len(y_axis.entries),
            cell_count,
        )
    samples = latent_image.get("samples") if isinstance(latent_image, dict) else None
    shape = getattr(samples, "shape", ())
    if len(shape) < 4:
        return
    latent_area = int(shape[-1]) * int(shape[-2])
    # SD-family latent pixels generally decode to 8x8 image pixels. This is an
    # intentionally conservative preflight estimate; the compositor remains the
    # final authority once the VAE returns the real image dimensions.
    estimated_pixels = cell_count * latent_area * 64
    configured_limit = max(1, round(float(max_canvas_megapixels) * 1_000_000))
    if estimated_pixels > configured_limit:
        logger.warning(
            "[LoraTester] XY preflight estimates %d image pixels before labels "
            "for %d cells, above the configured %d-pixel canvas limit; sampling may be rejected.",
            estimated_pixels,
            cell_count,
            configured_limit,
        )
    if latent_area > 512 * 512:
        logger.warning(
            "[LoraTester] Latent spatial area %d is unusually large; XY decoding "
            "can consume substantial CPU RAM and VRAM.",
            latent_area,
        )


def _xy_stack_key(stack: LoraStack | None) -> tuple[Any, ...]:
    if stack is None:
        return ()
    return (stack.signature(), stack.artist_template)


def _xy_mixer_labels_possible(
    model: Any,
    tasks: Sequence[tuple[int, int, dict[str, Any]]],
    mixer_config: AnimaArtistMixerConfig | None,
    use_anima_artist_mixer: bool,
) -> bool:
    if detect_model_family(model) != MODEL_FAMILY_ANIMA or not bool(use_anima_artist_mixer):
        return False
    config = mixer_config or AnimaArtistMixerConfig()
    if not config.enabled or config.strength <= 0.0 or not anima_artist_mixer_available():
        return False
    for _, _, values in tasks:
        stack = values.get("lora_stack")
        stack_artists = stack.artist_entries if isinstance(stack, LoraStack) else ()
        if len((*stack_artists, *parse_artist_tag_entries(values.get("independent_artist_tags", "")))) > 1:
            return True
    return False


def _raw_batch_slot(decoded: Any, raw_batch: Any, index: int, total: int) -> Any:
    shape = getattr(decoded, "shape", None)
    if shape is None or len(shape) != 4 or int(shape[0]) != 1:
        raise ValueError(
            "XY sampler requires VAE decode to return one IMAGE per cell with shape [1,H,W,C]"
        )
    if raw_batch is None:
        try:
            raw_batch = decoded.detach().new_empty((int(total), *tuple(shape[1:])))
        except AttributeError:
            raw_batch = decoded.new_empty((int(total), *tuple(shape[1:])))
    raw_batch[index].copy_(decoded[0])
    return raw_batch


class XYTestSampler(LoraTesterSampler):
    _last_anima_diagnosis: dict[str, Any] = {"required": False}

    """Generic XY sampler; axis producers define data and presentation metadata."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "latent_image": ("LATENT",),
                "x_axis": ("XY_AXIS", {"tooltip": "Axis rendered as image columns."}),
                "y_axis": ("XY_AXIS", {"tooltip": "Axis rendered as image rows."}),
                "positive_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Base positive prompt. A Prompt Axis entry overrides this value.",
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
                        "max": MAX_SEED,
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
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_mode": COLOR_MODE_INPUT,
                "show_axis_details": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Render categorized axis detail tables and text below the XY grid.",
                    },
                ),
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
                        "tooltip": "Stops when the labeled XY sheet would exceed this size.",
                    },
                ),
                "extra_footer_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "advanced": True,
                        "tooltip": "Optional text rendered as a final NOTES section below all axis details.",
                    },
                ),
            },
            "optional": {
                "custom_style": ("LORA_TESTER_STYLE", {"tooltip": "Used when Color Mode is custom."}),
                "artist_tag_template": (
                    "ARTIST_TAG_TEMPLATE",
                    {"tooltip": "Overrides model-specific and stack-specific artist syntax."},
                ),
                "anima_mixer_config": (
                    "ANIMA_ARTIST_MIXER_CONFIG",
                    {"tooltip": "Optional settings for multi-artist Anima cells."},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("comparison_sheet", "raw_images")
    OUTPUT_TOOLTIPS = (
        "Labeled XY comparison sheet.",
        "Every decoded cell as a row-major IMAGE batch, independent of sheet layout.",
    )
    FUNCTION = "sample"
    CATEGORY = "Lora Tester/XY"
    DESCRIPTION = (
        "Samples every combination of two extensible axes and returns both a labeled sheet "
        "and the original decoded image sequence."
    )

    @classmethod
    def IS_CHANGED(cls, **values: Any) -> Any:
        # Axis producers carry file fingerprints and deterministic random seeds in their
        # serialized values.  Keep this node cacheable without guessing their internals.
        return tuple(
            (key, repr(values[key]))
            for key in sorted(values)
            if key not in {"unique_id", "custom_style"}
        )

    def sample(
        self,
        model: Any,
        clip: Any,
        vae: Any,
        latent_image: dict[str, Any],
        x_axis: XYAxis,
        y_axis: XYAxis,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        color_mode: str,
        show_axis_details: bool,
        log_test_details: bool = True,
        use_anima_artist_mixer: bool = True,
        max_canvas_megapixels: float = DEFAULT_MAX_CANVAS_MEGAPIXELS,
        extra_footer_text: str = "",
        custom_style: StyleConfig | None = None,
        artist_tag_template: ArtistTagTemplate | None = None,
        anima_mixer_config: AnimaArtistMixerConfig | None = None,
        unique_id: Any = None,
    ) -> tuple[Any, Any]:
        sampled = self._sample_xy(
            model=model,
            clip=clip,
            vae=vae,
            latent_image=latent_image,
            x_axis=x_axis,
            y_axis=y_axis,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            color_mode=color_mode,
            show_axis_details=show_axis_details,
            log_test_details=log_test_details,
            use_anima_artist_mixer=use_anima_artist_mixer,
            max_canvas_megapixels=max_canvas_megapixels,
            extra_footer_text=extra_footer_text,
            custom_style=custom_style,
            artist_tag_template=artist_tag_template,
            anima_mixer_config=anima_mixer_config,
            unique_id=unique_id,
            return_raw=True,
            log_label="XY image",
        )
        return self._with_anima_remap_ui(model, sampled)

    @classmethod
    def _with_anima_remap_ui(cls, model: Any, sampled: Any) -> Any:
        diagnosis = cls._last_anima_diagnosis
        if not diagnosis.get("required"):
            return sampled
        remap_ui = _missing_anima_remap_ui(diagnosis)
        if not remap_ui:
            return sampled
        sheet, raw_batch = sampled
        return {"ui": remap_ui, "result": (sheet, raw_batch)}

    def _sample_xy(
        self,
        *,
        model: Any,
        clip: Any,
        vae: Any,
        latent_image: dict[str, Any],
        x_axis: XYAxis,
        y_axis: XYAxis,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        color_mode: str,
        show_axis_details: bool,
        log_test_details: bool,
        use_anima_artist_mixer: bool,
        max_canvas_megapixels: float,
        extra_footer_text: str,
        custom_style: StyleConfig | None,
        artist_tag_template: ArtistTagTemplate | None,
        anima_mixer_config: AnimaArtistMixerConfig | None,
        unique_id: Any,
        return_raw: bool,
        log_label: str,
        x_group_gap: int | None = None,
    ) -> tuple[Any, Any]:
        _validate_single_latent(latent_image)
        if not isinstance(x_axis, XYAxis) or not isinstance(y_axis, XYAxis):
            raise TypeError("x_axis and y_axis must come from XY Axis nodes")
        _validate_xy_axis_conflicts(x_axis, y_axis)
        _warn_xy_scale(latent_image, x_axis, y_axis, max_canvas_megapixels)
        style = _style_for_mode(color_mode, custom_style)
        base_values = {
            "positive_prompt": str(positive_prompt).strip(),
            "prompt_prefix": "",
            "prompt_suffix": "",
            "independent_artist_tags": "",
            "lora_stack": None,
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": str(sampler_name),
            "scheduler": str(scheduler),
            "denoise": float(denoise),
        }
        _set_int_parameter("seed", 0, MAX_SEED)(base_values, seed)
        _set_int_parameter("steps", 1, 10000)(base_values, steps)
        _set_float_parameter("cfg", 0.0, 100.0)(base_values, cfg)
        _set_float_parameter("denoise", 0.0, 1.0)(base_values, denoise)
        tasks: list[tuple[int, int, dict[str, Any]]] = []
        for row, y_entry in enumerate(y_axis.entries):
            for column, x_entry in enumerate(x_axis.entries):
                tasks.append((row, column, _resolve_xy_values(base_values, x_entry, y_entry)))
        if not tasks:
            raise ValueError("XY axes must produce at least one sampling cell")

        groups: dict[tuple[Any, ...], tuple[LoraStack | None, list[tuple[int, int, dict[str, Any]]]]] = {}
        for task in tasks:
            stack = task[2].get("lora_stack")
            if stack is not None and not isinstance(stack, LoraStack):
                raise TypeError("lora_stack axis values must be LoraStack instances")
            key = _xy_stack_key(stack)
            if key not in groups:
                groups[key] = (stack, [])
            groups[key][1].append(task)

        stacks = tuple(
            stack for stack, _ in groups.values() if isinstance(stack, LoraStack)
        )
        independent_values = "\n".join(
            values.get("independent_artist_tags", "")
            for _, _, values in tasks
            if values.get("independent_artist_tags", "")
        )
        preflight = _combination_preflight(
            model,
            stacks,
            artist_tag_template,
            anima_mixer_config,
            use_anima_artist_mixer,
            independent_values,
        )
        if bool(log_test_details):
            _log_combination_preflight(preflight)

        total_tasks = len(tasks)
        progress = _make_progress_bar(total_tasks, node_id=unique_id)
        anima_state_dicts: list[Mapping[Any, Any]] = []
        compositor: XYMatrixCompositor | None = None
        session = None
        raw_batch = None
        task_index = 0
        self._lora_cache.clear()
        try:
            for stack, group_tasks in groups.values():
                column_model = model
                column_clip = clip
                column_lora_usage: list[tuple[str, float, str, str]] = []
                prompt_entries: tuple[tuple[bool, str, float], ...] = ()
                stack_template = artist_tag_template
                negative = None
                try:
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
                            anima_state_dicts.append(loaded.state_dict)
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
                    for group_task_index, (row, column, values) in enumerate(group_tasks):
                        _throw_if_interrupted()
                        task_model = task_clip = route = positive = sampled = decoded = None
                        try:
                            task_model = column_model
                            task_clip = column_clip
                            route = _route_prompt_entries(
                                model=column_model,
                                clip=column_clip,
                                prefix_parts=(values.get("prompt_prefix", ""),),
                                entries=prompt_entries,
                                suffix_parts=(values.get("positive_prompt", ""), values.get("prompt_suffix", "")),
                                artist_template=stack_template,
                                mixer_config=anima_mixer_config,
                                independent_artist_tags=values.get("independent_artist_tags", ""),
                                use_anima_artist_mixer=use_anima_artist_mixer,
                            )
                            task_model = route.model
                            positive = route.positive
                            if bool(log_test_details):
                                artists = (
                                    *(stack.artist_entries if stack is not None else ()),
                                    *parse_artist_tag_entries(values.get("independent_artist_tags", "")),
                                )
                                artist_cache = "external:lazy-per-sample" if route.used_external_mixer else "none"
                                _log_test_usage(
                                    label=f"{log_label} {task_index + 1}/{total_tasks} | row={row + 1} | column={column + 1}",
                                    loras=tuple(
                                        (
                                            name,
                                            weight,
                                            "run-local:hit" if group_task_index else status,
                                            trigger,
                                        )
                                        for name, weight, status, trigger in column_lora_usage
                                    ),
                                    artists=tuple((tag, float(weight), artist_cache) for tag, weight in artists),
                                    rendered_tags=route.rendered_tags,
                                    route=route,
                                    model_cache=(
                                        "column reuse" if group_task_index else "column build"
                                    ),
                                )
                            sampled = _common_ksampler(
                                task_model,
                                int(values["seed"]),
                                int(values["steps"]),
                                float(values["cfg"]),
                                values["sampler_name"],
                                values["scheduler"],
                                positive,
                                negative,
                                latent_image,
                                float(values["denoise"]),
                                progress=progress,
                                completed_tasks=task_index,
                                total_tasks=total_tasks,
                            )
                            decoded = _decode_vae(vae, sampled)
                            shape = getattr(decoded, "shape", None)
                            if shape is None or len(shape) != 4 or int(shape[0]) != 1:
                                raise ValueError("VAE decode must return shape [1,H,W,C] for each XY cell")
                            if compositor is None:
                                compositor = XYMatrixCompositor(
                                    x_axis,
                                    y_axis,
                                    int(shape[-2]),
                                    int(shape[-3]),
                                    style=style,
                                    show_details=bool(show_axis_details),
                                    image_fit="strict",
                                    max_canvas_pixels=max(1, round(float(max_canvas_megapixels) * 1_000_000)),
                                    reserve_artist_mixer_labels=preflight.mixer_active,
                                    extra_detail_text=extra_footer_text,
                                    x_group_gap=x_group_gap,
                                )
                                session = compositor.start()
                            session.submit(decoded, coordinate=(row, column), artist_mixer=route.used_external_mixer)
                            if return_raw:
                                raw_batch = _raw_batch_slot(
                                    decoded,
                                    raw_batch,
                                    row * len(x_axis.entries) + column,
                                    total_tasks,
                                )
                            task_index += 1
                            progress.update_absolute(task_index, total_tasks)
                        finally:
                            if task_model is not column_model:
                                _release_temporary_model(task_model, column_model)
                            if task_clip is not column_clip:
                                _release_temporary_model(task_clip, column_clip)
                            task_model = task_clip = route = positive = sampled = decoded = None
                finally:
                    if stack is not None:
                        _release_temporary_model(column_model, model)
                        _release_temporary_model(column_clip, clip)
                    column_model = column_clip = negative = None
        finally:
            self._lora_cache.clear()
        anima_diagnosis = anima_remap_diagnosis(
            model=model,
            state_dicts=tuple(anima_state_dicts),
        )
        warn_missing_anima_remap(anima_diagnosis, unique_id=unique_id)
        type(self)._last_anima_diagnosis = anima_diagnosis
        if session is None:
            raise RuntimeError("XY sampler generated no images")
        sheet_pil = session.finalize(strict=True)
        try:
            sheet = pil_to_comfy_image(sheet_pil)
        finally:
            sheet_pil.close()
        return sheet, raw_batch


class MultiPromptInputNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        required: dict[str, tuple[str, dict[str, Any]]] = {
            "prompt_count": (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": MAX_STACK_ITEMS,
                    "step": 1,
                    "tooltip": "Number of prompt input rows.",
                },
            )
        }
        for index in range(1, MAX_STACK_ITEMS + 1):
            required[f"positive_prompt_{index}"] = (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": f"Positive prompt row {index}.",
                },
            )
        return {"required": required}

    RETURN_TYPES = ("LORA_TESTER_PROMPT_LIST",)
    RETURN_NAMES = ("prompt_list",)
    OUTPUT_TOOLTIPS = ("Structured positive prompts ready for a Prompt Axis node.",)
    FUNCTION = "build_prompts"
    CATEGORY = "Lora Tester/XY/Prompt"
    DESCRIPTION = "Builds an ordered prompt list from separate input rows."

    @staticmethod
    def build_prompts(prompt_count: int, positive_prompt_1: str, **values: Any) -> tuple[PromptList]:
        count = int(prompt_count)
        if not 1 <= count <= MAX_STACK_ITEMS:
            raise ValueError(f"prompt_count must be between 1 and {MAX_STACK_ITEMS}")
        prompts = []
        prompt_values = {"positive_prompt_1": positive_prompt_1, **values}
        for index in range(1, count + 1):
            prompt = str(prompt_values.get(f"positive_prompt_{index}", "")).strip()
            if not prompt:
                raise ValueError(f"Positive prompt row {index} cannot be empty")
            prompts.append(PromptEntry(prompt))
        return (PromptList(tuple(prompts)),)


class GlobalPromptAppendNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt_list": ("LORA_TESTER_PROMPT_LIST",),
                "addition": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "position": (["before", "after"], {"default": "before"}),
                "independent_artist_tags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Artist-only tags kept separate from ordinary prompts and LoRA triggers.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LORA_TESTER_PROMPT_LIST",)
    RETURN_NAMES = ("prompt_list",)
    OUTPUT_TOOLTIPS = ("Prompt list with global text and independent artist tags applied.",)
    FUNCTION = "append_prompt"
    CATEGORY = "Lora Tester/XY/Prompt"
    DESCRIPTION = "Adds shared prompt text before or after every prompt without parsing ordinary @ text as artists."

    @staticmethod
    def append_prompt(
        prompt_list: PromptList,
        addition: str,
        position: str,
        independent_artist_tags: str,
    ) -> tuple[PromptList]:
        if not isinstance(prompt_list, PromptList):
            raise TypeError("prompt_list must come from a Multi Prompt Input node")
        return (prompt_list.append_global(addition, position=position, independent_artist_tags=independent_artist_tags),)


class PromptAxisNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt_list": ("LORA_TESTER_PROMPT_LIST",),
                "axis_title": ("STRING", {"default": "PROMPT", "multiline": False, "tooltip": "Overall axis heading; row/column labels come from individual entries."}),
            }
        }

    RETURN_TYPES = ("XY_AXIS",)
    RETURN_NAMES = ("axis",)
    OUTPUT_TOOLTIPS = ("Prompt entries as a grouped XY axis.",)
    FUNCTION = "build_axis"
    CATEGORY = "Lora Tester/XY/Prompt"
    DESCRIPTION = "Converts structured prompts into an XY axis with readable prompt detail text."

    @staticmethod
    def build_axis(prompt_list: PromptList, axis_title: str) -> tuple[XYAxis]:
        return (build_prompt_axis(prompt_list, title=axis_title.strip() or "PROMPT"),)


class LoraStackAxisNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "lorastacks": ("LORA_STACK_LIST",),
                "include_base": ("BOOLEAN", {"default": True, "tooltip": "Keep BASE in its own separated group."}),
                "axis_title": ("STRING", {"default": "STYLE", "multiline": False, "tooltip": "Overall axis heading; row/column labels come from individual entries."}),
            }
        }

    RETURN_TYPES = ("XY_AXIS",)
    RETURN_NAMES = ("axis",)
    OUTPUT_TOOLTIPS = ("LoRA/artist stack configurations as a grouped XY axis.",)
    FUNCTION = "build_axis"
    CATEGORY = "Lora Tester/XY/Style"
    DESCRIPTION = "Converts LoRA and artist-tag stacks into an X axis with grouped BASE separation and categorized tables."

    @staticmethod
    def build_axis(lorastacks: LoraStackList, include_base: bool, axis_title: str) -> tuple[XYAxis]:
        return (build_lora_stack_axis(lorastacks, include_base=include_base, title=axis_title.strip() or "STYLE"),)


class SeedListNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "mode": (["list", "random"], {"default": "list"}),
                "seed_text": ("STRING", {"default": "0", "multiline": False, "tooltip": "Comma/space separated decimal seeds in list mode."}),
                "random_count": ("INT", {"default": 4, "min": 1, "max": MAX_AXIS_ENTRIES, "step": 1}),
                "random_source_seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
            }
        }

    RETURN_TYPES = ("LORA_TESTER_SEED_LIST",)
    RETURN_NAMES = ("seed_list",)
    OUTPUT_TOOLTIPS = ("Ordered explicit or deterministic random seeds.",)
    FUNCTION = "build_seeds"
    CATEGORY = "Lora Tester/XY/Seed"
    DESCRIPTION = "Builds a seed list from explicit values or a deterministic random generator."

    @staticmethod
    def build_seeds(mode: str, seed_text: str, random_count: int, random_source_seed: int) -> tuple[SeedList]:
        if mode == "list":
            return (SeedList.parse(seed_text),)
        if mode == "random":
            return (SeedList.random(random_count, random_source_seed),)
        raise ValueError("mode must be list or random")


class SeedAxisNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "seed_list": ("LORA_TESTER_SEED_LIST",),
                "axis_title": ("STRING", {"default": "SEED", "multiline": False, "tooltip": "Overall axis heading; row/column labels come from individual entries."}),
            }
        }

    RETURN_TYPES = ("XY_AXIS",)
    RETURN_NAMES = ("axis",)
    OUTPUT_TOOLTIPS = ("Seeds as an XY axis, ready for either sampler socket.",)
    FUNCTION = "build_axis"
    CATEGORY = "Lora Tester/XY/Seed"
    DESCRIPTION = "Converts a seed list into an XY axis."

    @staticmethod
    def build_axis(seed_list: SeedList, axis_title: str) -> tuple[XYAxis]:
        return (build_seed_axis(seed_list, title=axis_title.strip() or "SEED"),)


class AxisComposerNode:
    """Convert any supported raw axis source into one orientation-neutral XY axis."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "axis_title": (
                    "STRING",
                    {
                        "default": "AXIS",
                        "multiline": False,
                        "tooltip": "Overall title rendered beside the axis; entry labels come from each source item.",
                    },
                ),
                "include_base": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "For a style stack source, place BASE in its own group.",
                    },
                ),
            },
            "optional": {
                "source": (
                    "*",
                    {
                        "forceInput": True,
                        "tooltip": "Connect one prompt list, style stack/list, or seed list source.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("XY_AXIS",)
    RETURN_NAMES = ("axis",)
    OUTPUT_TOOLTIPS = ("A grouped, orientation-neutral XY axis.",)
    FUNCTION = "compose_axis"
    CATEGORY = "Lora Tester/XY/Axis"
    DESCRIPTION = "Converts one supported prompt, style, or seed source into an axis that can be connected to X or Y."

    @staticmethod
    def compose_axis(
        axis_title: str,
        include_base: bool,
        source: Any = None,
    ) -> tuple[XYAxis]:
        if isinstance(source, XYAxis):
            axis = XYAxis(
                title=axis_title.strip() or source.title,
                groups=source.groups,
                detail_blocks=source.detail_blocks,
            )
        elif isinstance(source, PromptList):
            axis = build_prompt_axis(source, title=axis_title.strip() or "AXIS")
        elif isinstance(source, LoraStackList):
            axis = build_lora_stack_axis(
                source,
                include_base=bool(include_base),
                title=axis_title.strip() or "AXIS",
            )
        elif isinstance(source, LoraStack):
            axis = build_lora_stack_axis(
                LoraStackList((source,)),
                include_base=bool(include_base),
                title=axis_title.strip() or "AXIS",
            )
        elif isinstance(source, SeedList):
            axis = build_seed_axis(source, title=axis_title.strip() or "AXIS")
        else:
            raise TypeError(
                "Axis Composer expects a PromptList, LoraStack, LoraStackList, SeedList, or XYAxis source"
            )
        return (axis,)


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
    CATEGORY = "Lora Tester/XY/Style"
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
    CATEGORY = "Lora Tester/XY/Style"
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
    CATEGORY = "Lora Tester/XY/Style"
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
    DEPRECATED = True

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
                "independent_artist_tags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": (
                            "Artist-only tags applied to every matrix cell and parsed "
                            "separately for optional Anima mixing. Shared/row prompts "
                            "and LoRA triggers are never extracted."
                        ),
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
    CATEGORY = "Lora Tester/Deprecated"
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
        independent_artist_tags: str = "",
        custom_style: StyleConfig | None = None,
        artist_tag_template: ArtistTagTemplate | None = None,
        anima_mixer_config: AnimaArtistMixerConfig | None = None,
        unique_id: Any = None,
        **values: Any,
    ) -> tuple[Any]:
        if not isinstance(lorastacks, LoraStackList):
            raise TypeError("lorastacks must be a LoRA Stack List")
        count = int(prompt_count)
        if not 1 <= count <= MAX_STACK_ITEMS:
            raise ValueError(f"prompt_count must be between 1 and {MAX_STACK_ITEMS}")
        prompts: list[PromptEntry] = []
        for index in range(1, count + 1):
            prompt = str(values.get(f"positive_prompt_{index}", "")).strip()
            if not prompt:
                raise ValueError(f"Positive prompt row {index} cannot be empty")
            prompts.append(
                PromptEntry(
                    prompt=prompt,
                    prefix=str(prompt_prefix).strip(),
                    independent_artist_tags=str(independent_artist_tags).strip(),
                )
            )
        x_axis = build_lora_stack_axis(lorastacks, include_base=True, title="STYLE")
        y_axis = build_prompt_axis(PromptList(tuple(prompts)), title="PROMPT")
        sampler = XYTestSampler()
        sheet, _ = sampler._sample_xy(
            model=model,
            clip=clip,
            vae=vae,
            latent_image=latent_image,
            x_axis=x_axis,
            y_axis=y_axis,
            positive_prompt="",
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            color_mode=color_mode,
            show_axis_details=show_lora_details,
            log_test_details=log_test_details,
            use_anima_artist_mixer=use_anima_artist_mixer,
            max_canvas_megapixels=max_canvas_megapixels,
            extra_footer_text="",
            custom_style=custom_style,
            artist_tag_template=artist_tag_template,
            anima_mixer_config=anima_mixer_config,
            unique_id=unique_id,
            return_raw=False,
            log_label="Combination image",
            x_group_gap=control_gap,
        )
        remap_ui = _missing_anima_remap_ui(XYTestSampler._last_anima_diagnosis)
        if remap_ui:
            return {"ui": remap_ui, "result": (sheet,)}
        return (sheet,)


NODE_CLASS_MAPPINGS = {
    "LoraTesterSampler": LoraTesterSampler,
    "LoraTesterStyle": LoraTesterStyleNode,
    "ArtistTagTemplate": ArtistTagTemplateNode,
    "AnimaArtistMixerConfig": AnimaArtistMixerConfigNode,
    "LoraStack": LoraStackNode,
    "LoraStackSplitter": LoraStackSplitterNode,
    "LoraStackLister": LoraStackListerNode,
    "MultiPromptSample": MultiPromptSampleNode,
    "LoraTesterXYSampler": XYTestSampler,
    "LoraTesterMultiPromptInput": MultiPromptInputNode,
    "LoraTesterGlobalPromptAppend": GlobalPromptAppendNode,
    "LoraTesterPromptAxis": PromptAxisNode,
    "LoraTesterLoraStackAxis": LoraStackAxisNode,
    "LoraTesterSeedList": SeedListNode,
    "LoraTesterSeedAxis": SeedAxisNode,
    "LoraTesterAxisComposer": AxisComposerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraTesterSampler": "Style Component Tester",
    "LoraTesterStyle": "LoRA Tester Style",
    "ArtistTagTemplate": "Artist Tag Template",
    "AnimaArtistMixerConfig": "Anima Artist Mixer Configuration",
    "LoraStack": "Style Stack",
    "LoraStackSplitter": "Style Stack Splitter",
    "LoraStackLister": "Style Stack Lister",
    "MultiPromptSample": "Style Combination Tester",
    "LoraTesterXYSampler": "XY Test Sampler",
    "LoraTesterMultiPromptInput": "Multi Prompt Input",
    "LoraTesterGlobalPromptAppend": "Global Prompt Append",
    "LoraTesterPromptAxis": "Prompt Axis",
    "LoraTesterLoraStackAxis": "Style Axis",
    "LoraTesterSeedList": "Seed List / Random Seeds",
    "LoraTesterSeedAxis": "Seed Axis",
    "LoraTesterAxisComposer": "Axis Composer",
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
    "XYTestSampler",
    "MultiPromptInputNode",
    "GlobalPromptAppendNode",
    "PromptAxisNode",
    "LoraStackAxisNode",
    "SeedListNode",
    "SeedAxisNode",
    "AxisComposerNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "compose_positive_prompt",
    "register_xy_parameter_handler",
]
