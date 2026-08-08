from __future__ import annotations

import math
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .comfy_adapter import pil_to_comfy_image
from .compositor import LoraComparisonCompositor, image_to_pil
from .layout import LoraSpec, RenderTask, build_layout
from .node_contract import COLOR_MODE_INPUT, SHOW_LORA_DETAILS_INPUT
from .styles import StyleConfig, available_style_decorators


MAX_CACHED_LORAS = 3
DEFAULT_MAX_CANVAS_MEGAPIXELS = 150.0


def _get_lora_names() -> list[str]:
    import folder_paths

    names = list(folder_paths.get_filename_list("loras"))
    return names or [""]


def _get_sampler_names() -> Sequence[str]:
    import comfy.samplers

    return comfy.samplers.KSampler.SAMPLERS


def _get_scheduler_names() -> Sequence[str]:
    import comfy.samplers

    return comfy.samplers.KSampler.SCHEDULERS


def _resolve_lora_path(name: str) -> str:
    import folder_paths

    return folder_paths.get_full_path_or_raise("loras", name)


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


class LoraTesterSampler:
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

    def sample(
        self,
        model: Any,
        clip: Any,
        vae: Any,
        latent_image: dict[str, Any],
        positive_prompt: str,
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
        max_canvas_megapixels: float = DEFAULT_MAX_CANVAS_MEGAPIXELS,
        custom_style: StyleConfig | None = None,
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
        loaded_loras = tuple(self._load_lora(spec.name) for spec in specs)
        progress = _make_progress_bar(plan.unique_task_count, node_id=unique_id)

        session = None
        for index, task in enumerate(plan.tasks, start=1):
            _throw_if_interrupted()
            task_model, task_clip = self._apply_task_loras(
                model,
                clip,
                task,
                loaded_loras,
            )
            task_positive_prompt = compose_positive_prompt(
                positive_prompt,
                task.prompt_additions,
            )
            positive = _encode_prompt(task_clip, task_positive_prompt)
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

            del task_model, task_clip, positive, negative, sampled, decoded, decoded_image

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
        if not str(name).strip():
            raise ValueError(
                "No LoRA file is selected. Add a LoRA to ComfyUI's models/loras directory "
                "and refresh the node list."
            )
        path = _resolve_lora_path(name)
        cache_key = os.path.normcase(os.path.abspath(path))
        cached = self._lora_cache.pop(cache_key, None)
        if cached is not None:
            self._lora_cache[cache_key] = cached
            return cached

        state_dict, metadata = _load_lora_file(path)
        cached = _CachedLora(path=path, state_dict=state_dict, metadata=metadata)
        self._lora_cache[cache_key] = cached
        while len(self._lora_cache) > MAX_CACHED_LORAS:
            self._lora_cache.popitem(last=False)
        return cached

    @staticmethod
    def _apply_task_loras(
        base_model: Any,
        base_clip: Any,
        task: RenderTask,
        loaded_loras: Sequence[_CachedLora],
    ) -> tuple[Any, Any]:
        task_model = base_model
        task_clip = base_clip
        for weight, loaded in zip(task.weights, loaded_loras):
            if math.isclose(weight, 0.0, abs_tol=1e-12):
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


NODE_CLASS_MAPPINGS = {
    "LoraTesterSampler": LoraTesterSampler,
    "LoraTesterStyle": LoraTesterStyleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraTesterSampler": "LoRA Tester (KSampler)",
    "LoraTesterStyle": "LoRA Tester Style",
}


__all__ = [
    "LoraTesterSampler",
    "LoraTesterStyleNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "compose_positive_prompt",
]
