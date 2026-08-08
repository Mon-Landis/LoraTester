from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_tester import LoraComparisonCompositor, StyleConfig


def background_tile() -> Image.Image:
    tile = Image.new("RGB", (96, 96), "#17211F")
    draw = ImageDraw.Draw(tile)
    draw.line((0, 24, 96, 24), fill="#2B3A36", width=1)
    draw.line((24, 0, 24, 96), fill="#2B3A36", width=1)
    draw.line((0, 95, 95, 0), fill="#26332F", width=1)
    return tile


def write_manifest(compositor: LoraComparisonCompositor, path: Path) -> None:
    payload = {
        "lora_count": compositor.plan.lora_count,
        "unique_tasks": compositor.plan.unique_task_count,
        "occupied_cells": compositor.plan.occupied_cell_count,
        "duplicate_placements": compositor.plan.duplicate_placement_count,
        "queue": [
            {
                "index": task.sequence_index,
                "task_id": task.task_id,
                "weights": list(task.weights),
                "active_slots": list(task.active_slots),
                "placements": [list(item) for item in compositor.plan.placements_for(task.task_id)],
            }
            for task in compositor.plan.tasks
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    output = ROOT / "previews"
    output.mkdir(parents=True, exist_ok=True)

    one = LoraComparisonCompositor.from_values(
        ["character_style_v12.safetensors"],
        [0.8],
        180,
        120,
        style=StyleConfig.black(),
        max_canvas_pixels=None,
    )
    one.render_template().save(output / "1_lora_black.png")

    two = LoraComparisonCompositor.from_values(
        ["linework_style.safetensors", "color_script_v3.safetensors"],
        [0.8, 1.0],
        120,
        90,
        style=StyleConfig.white(outer_margin=20, region_gap=32),
        max_canvas_pixels=None,
    )
    two.render_template().save(output / "2_lora_white.png")

    three = LoraComparisonCompositor.from_values(
        [
            "LoraX",
            "LoraY",
            "LoraZZZ",
        ],
        [0.9, 3.0, 2.0],
        96,
        72,
        style=StyleConfig.black(outer_margin=22, region_gap=32),
        max_canvas_pixels=None,
    )
    three.render_template().save(output / "3_lora_black.png")
    write_manifest(three, output / "3_lora_manifest.json")

    requested = LoraComparisonCompositor.from_values(
        ["LoraX", "LoraY", "LoraZZZ"],
        [0.9, 3.0, 2.0],
        640,
        800,
        style=StyleConfig.black(
            outer_margin=36,
            cell_gap=6,
            region_gap=72,
        ),
    )
    requested.render_template().save(output / "3_lora_axes_640x800.png")

    min_bound = LoraComparisonCompositor.from_values(
        ["LoraX", "LoraY", "LoraZZZ"],
        [0.9, 3.0, 2.0],
        640,
        800,
        lora_min_weights=[0.2, 0.75, 0.5],
        style=StyleConfig.black(
            outer_margin=36,
            cell_gap=6,
            region_gap=72,
        ),
    )
    min_bound.render_template().save(output / "3_lora_min_bound_640x800.png")

    custom_style = StyleConfig.custom(
        background_color="#101715",
        background_image=background_tile(),
        background_fit="tile",
        background_opacity=0.68,
        panel_color="#202A27",
        placeholder_color="#E7E9E2",
        text_color="#F2F3EF",
        muted_text_color="#9DA9A4",
        frame_color="#697771",
        accent_colors=("#D8F04A", "#45BFC5", "#F08866"),
        decorator="technical",
        show_coordinates=False,
        outer_margin=22,
        region_gap=32,
    )
    custom = LoraComparisonCompositor.from_values(
        ["A.safetensors", "B.safetensors", "C.safetensors"],
        [0.8, 1.0, 2.0],
        96,
        72,
        style=custom_style,
        max_canvas_pixels=None,
    )
    custom.render_template().save(output / "3_lora_custom.png")


if __name__ == "__main__":
    main()
