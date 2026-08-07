from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from PIL import Image

from .compositor import CompositionSession, LoraComparisonCompositor


def iter_comfy_image_batch(images: Any) -> Iterable[Any]:
    shape = getattr(images, "shape", None)
    if shape is None:
        raise TypeError("ComfyUI IMAGE input must expose a shape")
    if len(shape) == 3:
        yield images
        return
    if len(shape) != 4:
        raise ValueError(f"Expected ComfyUI IMAGE shape [B,H,W,C], received {tuple(shape)}")
    for index in range(int(shape[0])):
        yield images[index]


def submit_comfy_batch(
    session: CompositionSession,
    images: Any,
    *,
    task_ids: Sequence[str] | None = None,
) -> int:
    submitted = 0
    iterator = iter(iter_comfy_image_batch(images))
    if task_ids is None:
        for image in iterator:
            session.submit(image)
            submitted += 1
        return submitted

    ids = iter(task_ids)
    while True:
        try:
            image = next(iterator)
        except StopIteration:
            break
        try:
            task_id = next(ids)
        except StopIteration as exc:
            raise ValueError("ComfyUI image batch contains more images than task_ids") from exc
        session.submit(image, task_id=task_id)
        submitted += 1
    try:
        next(ids)
    except StopIteration:
        return submitted
    raise ValueError("task_ids contains more entries than the ComfyUI image batch")


def pil_to_comfy_image(image: Image.Image, *, device: Any = None, dtype: Any = None) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to create a ComfyUI IMAGE tensor") from exc

    array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    if dtype is not None or device is not None:
        tensor = tensor.to(device=device, dtype=dtype)
    return tensor


def compose_comfy_batch(
    compositor: LoraComparisonCompositor,
    images: Any,
    *,
    strict: bool = True,
    output_device: Any = None,
    output_dtype: Any = None,
) -> Any:
    session = compositor.start()
    submit_comfy_batch(session, images)
    result = session.finalize(strict=strict)
    return pil_to_comfy_image(result, device=output_device, dtype=output_dtype)


__all__ = [
    "compose_comfy_batch",
    "iter_comfy_image_batch",
    "pil_to_comfy_image",
    "submit_comfy_batch",
]
