from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence


MODEL_FAMILY_ANIMA_29B = "anima-2.9b"
BASE_BLOCK_COUNT = 28
EXPANDED_BLOCK_COUNT = 40

_LORA_BLOCK_RES = (
    re.compile(r"^lora_unet_blocks_(\d+)_"),
    re.compile(r"^diffusion_model\.blocks\.(\d+)\."),
    re.compile(r"^blocks\.(\d+)\."),
)


def _is_anima_key_map(key_map: Mapping[Any, Any]) -> bool:
    return any(
        isinstance(value, str) and value.startswith("diffusion_model.llm_adapter.")
        for value in key_map.values()
    )


def _remap_patch_installed() -> bool:
    try:
        import comfy.lora
    except ImportError:
        return False
    return getattr(comfy.lora.load_lora, "_anima29b_lora_patch", False)


def _model_block_count(model: Any) -> int:
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except (AttributeError, KeyError, TypeError):
        diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)

    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is not None:
        try:
            return len(blocks)
        except TypeError:
            pass

    config = getattr(getattr(model, "model", None), "model_config", None)
    unet_config = getattr(config, "unet_config", None)
    depth = unet_config.get("depth") if isinstance(unet_config, Mapping) else None
    return int(depth) if depth is not None else 0


def _lora_block_indices(state_dict: Mapping[Any, Any]) -> set[int]:
    indices: set[int] = set()
    for key in state_dict:
        if not isinstance(key, str):
            continue
        for pattern in _LORA_BLOCK_RES:
            match = pattern.match(key)
            if match is not None:
                indices.add(int(match.group(1)))
                break
    return indices


def anima_remap_diagnosis(
    *,
    model: Any,
    state_dicts: Sequence[Mapping[Any, Any] | None],
) -> dict[str, Any]:
    """Diagnose a missing 28-to-40-block Anima LoRA remap before sampling."""

    active_state_dicts = tuple(state_dict for state_dict in state_dicts if state_dict)
    if not active_state_dicts or _remap_patch_installed():
        return {"required": False}

    key_map: dict[str, str] = {}
    try:
        import comfy.lora

        comfy.lora.model_lora_keys_unet(getattr(model, "model", None), key_map)
    except Exception:
        logging.debug("Unable to inspect the Anima LoRA key map", exc_info=True)
        return {"required": False}
    if not _is_anima_key_map(key_map):
        return {"required": False}

    lora_blocks = set[int]()
    for state_dict in active_state_dicts:
        lora_blocks.update(_lora_block_indices(state_dict))
    model_blocks = _model_block_count(model)
    required = (
        model_blocks == EXPANDED_BLOCK_COUNT
        and lora_blocks == set(range(BASE_BLOCK_COUNT))
    )
    return {
        "required": required,
        "model_family": MODEL_FAMILY_ANIMA_29B,
        "model_blocks": model_blocks,
        "lora_blocks": sorted(lora_blocks),
    }


def warn_missing_anima_remap(diagnosis: Mapping[str, Any], *, unique_id: Any = None) -> None:
    if not diagnosis.get("required"):
        return
    suffix = f" (node {unique_id})" if unique_id is not None else ""
    logging.warning(
        "[LoraTester] A 28-block Anima LoRA may be applied to a 40-block Anima "
        "2.9B model, but ComfyUI-Anima-2.9B-loraPatch is not installed%s.",
        suffix,
    )
