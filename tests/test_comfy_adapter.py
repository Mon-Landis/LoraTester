from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from lora_tester.comfy_adapter import (
    compose_comfy_batch,
    iter_comfy_image_batch,
    pil_to_comfy_image,
)
from lora_tester.compositor import LoraComparisonCompositor
from lora_tester.layout import LoraSpec
from lora_tester.styles import StyleConfig


class ComfyAdapterTests(unittest.TestCase):
    def test_pil_to_comfy_tensor_shape_and_range(self) -> None:
        tensor = pil_to_comfy_image(Image.new("RGB", (5, 4), (255, 128, 0)))
        self.assertEqual(tuple(tensor.shape), (1, 4, 5, 3))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 1.0)

    def test_batch_iterator_accepts_single_and_batch(self) -> None:
        self.assertEqual(len(list(iter_comfy_image_batch(torch.zeros((4, 5, 3))))), 1)
        self.assertEqual(len(list(iter_comfy_image_batch(torch.zeros((3, 4, 5, 3))))), 3)
        with self.assertRaisesRegex(ValueError, "shape"):
            list(iter_comfy_image_batch(torch.zeros((3, 4))))

    def test_compose_batch_returns_one_comfy_image(self) -> None:
        compositor = LoraComparisonCompositor(
            (LoraSpec("A"),),
            8,
            6,
            style=StyleConfig.black(decorator="none", show_cell_captions=False),
            show_lora_details=False,
            max_canvas_pixels=None,
        )
        batch = torch.zeros((5, 6, 8, 3), dtype=torch.float32)
        result = compose_comfy_batch(compositor, batch)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(tuple(result.shape[1:3]), (compositor.geometry.canvas_size[1], compositor.geometry.canvas_size[0]))
        self.assertEqual(result.shape[-1], 3)


if __name__ == "__main__":
    unittest.main()
