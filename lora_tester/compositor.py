from __future__ import annotations

import math
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .layout import AxisSpec, Coordinate, LayoutPlan, LoraSpec, RenderTask, SLOTS, build_layout
from .styles import DecorationContext, RGBColor, StyleConfig, get_style_decorator


Rect: TypeAlias = tuple[int, int, int, int]
ImageInput: TypeAlias = Image.Image | np.ndarray | Any


def _format_weight(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        value = 0.0
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _inclusive(rect: Rect) -> Rect:
    left, top, right, bottom = rect
    return (left, top, max(left, right - 1), max(top, bottom - 1))


def _rect_size(rect: Rect) -> tuple[int, int]:
    return rect[2] - rect[0], rect[3] - rect[1]


def _font_text_width(font: ImageFont.ImageFont, text: str) -> float:
    if hasattr(font, "getlength"):
        return float(font.getlength(text))
    box = font.getbbox(text)
    return float(box[2] - box[0])


def _font_line_height(font: ImageFont.ImageFont) -> int:
    box = font.getbbox("Ag")
    return max(1, box[3] - box[1] + 2)


def _wrap_text_pixels(text: str, font: ImageFont.ImageFont, max_width: int) -> tuple[str, ...]:
    if max_width <= 0:
        return ("",)
    lines: list[str] = []
    paragraphs = str(text).splitlines() or [""]
    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for character in paragraph:
            candidate = current + character
            if current and _font_text_width(font, candidate) > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
        lines.append(current)
    return tuple(lines or [""])


@dataclass(frozen=True, slots=True)
class CompositionOptions:
    image_width: int
    image_height: int
    show_lora_details: bool = True
    image_fit: str = "strict"
    max_canvas_pixels: int | None = 150_000_000

    def __post_init__(self) -> None:
        if int(self.image_width) <= 0 or int(self.image_height) <= 0:
            raise ValueError("image_width and image_height must be positive")
        if self.image_fit not in {"strict", "contain", "cover", "stretch"}:
            raise ValueError("image_fit must be strict, contain, cover, or stretch")
        if self.max_canvas_pixels is not None and int(self.max_canvas_pixels) <= 0:
            raise ValueError("max_canvas_pixels must be positive or None")


@dataclass(frozen=True, slots=True)
class CellGeometry:
    coordinate: Coordinate
    cell_rect: Rect
    caption_rect: Rect | None
    image_rect: Rect
    special_label_rect: Rect | None


@dataclass(frozen=True, slots=True)
class LayoutGeometry:
    image_width: int
    image_height: int
    caption_height: int
    special_label_height: int
    footer_gap: int
    footer_padding: int
    footer_title_gap: int
    footer_name_line_gap: int
    canvas_size: tuple[int, int]
    grid_bounds: Rect
    footer_rect: Rect | None
    cells: Mapping[Coordinate, CellGeometry]
    region_rects: Mapping[str, Rect]
    axis_rects: Mapping[str, Rect]
    axis_font_size: int
    axis_value_font_size: int
    axis_padding: int

    def cell(self, coordinate: Coordinate) -> CellGeometry:
        try:
            return self.cells[coordinate]
        except KeyError as exc:
            raise KeyError(f"Coordinate is outside this geometry: {coordinate}") from exc


class FontResolver:
    def __init__(self, font_path: str | Path | None = None) -> None:
        self.font_path = Path(font_path).expanduser() if font_path else None
        self._cache: dict[tuple[int, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def get(self, size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        key = (max(1, int(size)), bool(bold))
        if key in self._cache:
            return self._cache[key]
        for path in self._candidate_paths(bold):
            try:
                font = ImageFont.truetype(str(path), size=key[0])
                self._cache[key] = font
                return font
            except (OSError, ValueError):
                continue
        font = ImageFont.load_default()
        self._cache[key] = font
        return font

    def _candidate_paths(self, bold: bool) -> Iterable[Path | str]:
        if self.font_path is not None:
            yield self.font_path
        windows_fonts = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
        names = (
            ("msyhbd.ttc", "seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")
            if bold
            else ("msyh.ttc", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")
        )
        for name in names:
            yield windows_fonts / name
        for root in (
            Path("/usr/share/fonts/truetype/dejavu"),
            Path("/usr/share/fonts/opentype/noto"),
        ):
            for name in names:
                yield root / name
        yield names[-1]


class LoraComparisonCompositor:
    def __init__(
        self,
        loras: Sequence[LoraSpec],
        image_width: int,
        image_height: int,
        *,
        show_lora_details: bool = True,
        style: StyleConfig | None = None,
        image_fit: str = "strict",
        max_canvas_pixels: int | None = 150_000_000,
    ) -> None:
        self.plan = build_layout(tuple(loras))
        self.options = CompositionOptions(
            image_width=int(image_width),
            image_height=int(image_height),
            show_lora_details=bool(show_lora_details),
            image_fit=image_fit,
            max_canvas_pixels=max_canvas_pixels,
        )
        self.style = style or StyleConfig.black()
        self.fonts = FontResolver(self.style.font_path)
        self.geometry = self._build_geometry()
        pixels = self.geometry.canvas_size[0] * self.geometry.canvas_size[1]
        if self.options.max_canvas_pixels is not None and pixels > self.options.max_canvas_pixels:
            rgb_mib = pixels * 3 / (1024 * 1024)
            raise MemoryError(
                f"Composite canvas would contain {pixels:,} pixels (about {rgb_mib:.0f} MiB as RGB). "
                "Use smaller tile dimensions or explicitly raise max_canvas_pixels."
            )

    @classmethod
    def from_values(
        cls,
        lora_names: Sequence[str],
        lora_weights: Sequence[float],
        image_width: int,
        image_height: int,
        *,
        lora_min_weights: Sequence[float] | None = None,
        trigger_words: Sequence[str] | None = None,
        show_lora_details: bool = True,
        style: StyleConfig | None = None,
        image_fit: str = "strict",
        max_canvas_pixels: int | None = 150_000_000,
    ) -> "LoraComparisonCompositor":
        names = tuple(lora_names)
        weights = tuple(lora_weights)
        min_weights = (
            (0.0,) * len(names)
            if lora_min_weights is None
            else tuple(lora_min_weights)
        )
        triggers = tuple(trigger_words or ("",) * len(names))
        if len(names) != len(weights):
            raise ValueError("lora_names and lora_weights must have the same length")
        if len(min_weights) != len(names):
            raise ValueError("lora_min_weights must have the same length as lora_names")
        if len(triggers) != len(names):
            raise ValueError("trigger_words must have the same length as lora_names")
        specs = tuple(
            LoraSpec(
                name=name,
                max_weight=weight,
                trigger_word=trigger,
                min_weight=min_weight,
            )
            for name, weight, min_weight, trigger in zip(names, weights, min_weights, triggers)
        )
        return cls(
            specs,
            image_width,
            image_height,
            show_lora_details=show_lora_details,
            style=style,
            image_fit=image_fit,
            max_canvas_pixels=max_canvas_pixels,
        )

    def start(self) -> "CompositionSession":
        return CompositionSession(self)

    def render_template(self) -> Image.Image:
        return self.start().finalize(strict=False)

    def compose(
        self,
        images: Sequence[ImageInput] | Mapping[str, ImageInput] | Iterable[ImageInput],
        *,
        strict: bool = True,
    ) -> Image.Image:
        session = self.start()
        if isinstance(images, Mapping):
            for task_id, image in images.items():
                session.submit(image, task_id=task_id)
        else:
            iterator = iter(images)
            for task in self.plan.tasks:
                try:
                    image = next(iterator)
                except StopIteration:
                    break
                session.submit(image, task_id=task.task_id)
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError(
                    f"Image sequence contains more than {self.plan.unique_task_count} items"
                )
        return session.finalize(strict=strict)

    def _build_geometry(self) -> LayoutGeometry:
        style = self.style
        options = self.options
        minimum_dimension = min(options.image_width, options.image_height)

        matrix_long_edge = max(
            len(self.plan.x_positions) * options.image_width,
            len(self.plan.y_positions) * options.image_height,
        )
        axis_font_size = style.font_size or max(16, min(160, round(matrix_long_edge / 64)))
        axis_value_font_size = style.small_font_size or max(12, round(axis_font_size * 0.72))
        axis_padding = max(8, min(32, round(axis_font_size * 0.24)))
        axis_font = self.fonts.get(axis_font_size, bold=True)
        axis_value_font = self.fonts.get(axis_value_font_size, bold=True)

        caption_height = 0
        if style.show_cell_captions:
            caption_height = style.caption_height or max(24, min(52, options.image_height // 8))

        special_coordinates: set[Coordinate] = set()
        if style.show_special_cell_labels and not style.show_cell_captions:
            special_coordinates = {
                cell.coordinate
                for cell in self.plan.cells
                if cell.occupied and self.plan.task(cell.task_id or "").kind == "triple"
            }
        special_label_height = (
            _font_line_height(axis_value_font) + axis_padding * 2
            if special_coordinates
            else 0
        )
        row_heights = {
            y: options.image_height + caption_height
            for y in self.plan.y_positions
        }
        axes_by_side = {
            side: tuple(axis for axis in self.plan.axes if axis.side == side)
            for side in ("top", "bottom", "left", "right")
        }
        if not style.show_axis_labels:
            axes_by_side = {side: () for side in axes_by_side}

        horizontal_band = (
            _font_line_height(axis_font)
            + _font_line_height(axis_value_font)
            + axis_padding * 3
        )
        top_band = horizontal_band if axes_by_side["top"] else 0
        bottom_band = horizontal_band if axes_by_side["bottom"] else 0
        bottom_special_band = (
            special_label_height
            if any(y == self.plan.y_positions[-1] for _, y in special_coordinates)
            else 0
        )
        bottom_content_band = max(bottom_band, bottom_special_band)

        def side_band_width(side: str) -> int:
            axes = axes_by_side[side]
            if not axes:
                return 0
            return (
                _font_line_height(axis_font)
                + _font_line_height(axis_value_font)
                + axis_padding * 4
            )

        left_band = side_band_width("left")
        right_band = side_band_width("right")
        effective_cell_gap = max(style.cell_gap, minimum_dimension // 160)
        effective_region_gap = max(style.region_gap, minimum_dimension // 10)

        x_origins: dict[int, int] = {}
        cursor_x = style.outer_margin + left_band
        for index, x in enumerate(self.plan.x_positions):
            x_origins[x] = cursor_x
            cursor_x += options.image_width
            if index < len(self.plan.x_positions) - 1:
                cursor_x += (
                    effective_region_gap
                    if index in self.plan.major_column_breaks
                    else effective_cell_gap
                )

        y_origins: dict[int, int] = {}
        cursor_y = style.outer_margin + top_band
        for index, y in enumerate(self.plan.y_positions):
            y_origins[y] = cursor_y
            cursor_y += row_heights[y]
            if index < len(self.plan.y_positions) - 1:
                cursor_y += (
                    effective_region_gap
                    if index in self.plan.major_row_breaks
                    else effective_cell_gap
                )

        grid_bounds = (
            style.outer_margin + left_band,
            style.outer_margin + top_band,
            cursor_x,
            cursor_y,
        )
        cell_geometries: dict[Coordinate, CellGeometry] = {}
        for cell in self.plan.cells:
            x, y = cell.coordinate
            left = x_origins[x]
            top = y_origins[y]
            caption_rect = None
            if caption_height:
                caption_rect = (left, top, left + options.image_width, top + caption_height)
            image_top = top + caption_height
            image_rect = (
                left,
                image_top,
                left + options.image_width,
                image_top + options.image_height,
            )
            special_label_rect = None
            if cell.coordinate in special_coordinates:
                label_top = image_rect[3]
                special_label_rect = (
                    left,
                    label_top,
                    left + options.image_width,
                    label_top + special_label_height,
                )
            cell_geometries[cell.coordinate] = CellGeometry(
                coordinate=cell.coordinate,
                cell_rect=(left, top, left + options.image_width, top + row_heights[y]),
                caption_rect=caption_rect,
                image_rect=image_rect,
                special_label_rect=special_label_rect,
            )

        region_rects: dict[str, Rect] = {}
        for region in self.plan.regions:
            geometries = [cell_geometries[coordinate] for coordinate in region.coordinates]
            region_rects[region.key] = (
                min(item.cell_rect[0] for item in geometries),
                min(item.cell_rect[1] for item in geometries),
                max(item.cell_rect[2] for item in geometries),
                max(item.cell_rect[3] for item in geometries),
            )

        axis_rects: dict[str, Rect] = {}
        for axis in self.plan.axes:
            if not style.show_axis_labels:
                break
            if axis.side in {"top", "bottom"}:
                geometries = [
                    cell_geometries[(position, self.plan.y_positions[0])]
                    for position in axis.positions
                ]
                left = min(item.cell_rect[0] for item in geometries)
                right = max(item.cell_rect[2] for item in geometries)
                if axis.side == "top":
                    axis_rects[axis.key] = (left, style.outer_margin, right, grid_bounds[1])
                else:
                    axis_rects[axis.key] = (
                        left,
                        grid_bounds[3],
                        right,
                        grid_bounds[3] + bottom_band,
                    )
            else:
                geometries = [
                    cell_geometries[(self.plan.x_positions[0], position)]
                    for position in axis.positions
                ]
                top = min(item.cell_rect[1] for item in geometries)
                bottom = max(item.cell_rect[3] for item in geometries)
                if axis.side == "left":
                    axis_rects[axis.key] = (style.outer_margin, top, grid_bounds[0], bottom)
                else:
                    axis_rects[axis.key] = (
                        grid_bounds[2],
                        top,
                        grid_bounds[2] + right_band,
                        bottom,
                    )

        footer_rect = None
        footer_gap = max(style.footer_gap, round(axis_font_size * 0.55))
        footer_padding = max(style.footer_padding, axis_padding)
        footer_title_gap = max(10, round(axis_value_font_size * 0.28))
        footer_name_line_gap = max(3, round(axis_value_font_size * 0.14))
        content_bottom = grid_bounds[3] + bottom_content_band
        canvas_bottom = content_bottom + style.outer_margin
        if options.show_lora_details:
            font_size = axis_font_size
            small_size = axis_value_font_size
            title_font = self.fonts.get(font_size, bold=True)
            name_font = self.fonts.get(small_size)
            column_width = (grid_bounds[2] - grid_bounds[0]) / len(self.plan.loras)
            available_width = max(1, round(column_width) - footer_padding * 2)
            max_name_lines = max(
                len(_wrap_text_pixels(lora.footer_text, name_font, available_width))
                for lora in self.plan.loras
            )
            footer_height = max(
                54,
                footer_padding * 2
                + _font_line_height(title_font)
                + footer_title_gap
                + max_name_lines
                * (_font_line_height(name_font) + footer_name_line_gap),
            )
            footer_top = content_bottom + footer_gap
            footer_rect = (grid_bounds[0], footer_top, grid_bounds[2], footer_top + footer_height)
            canvas_bottom = footer_rect[3] + style.outer_margin

        canvas_size = (grid_bounds[2] + right_band + style.outer_margin, canvas_bottom)
        return LayoutGeometry(
            image_width=options.image_width,
            image_height=options.image_height,
            caption_height=caption_height,
            special_label_height=special_label_height,
            footer_gap=footer_gap,
            footer_padding=footer_padding,
            footer_title_gap=footer_title_gap,
            footer_name_line_gap=footer_name_line_gap,
            canvas_size=canvas_size,
            grid_bounds=grid_bounds,
            footer_rect=footer_rect,
            cells=MappingProxyType(cell_geometries),
            region_rects=MappingProxyType(region_rects),
            axis_rects=MappingProxyType(axis_rects),
            axis_font_size=axis_font_size,
            axis_value_font_size=axis_value_font_size,
            axis_padding=axis_padding,
        )

    def _axis_value_labels(self, axis: AxisSpec) -> tuple[str, ...]:
        lora = self.plan.loras[SLOTS.index(axis.slot)]
        return tuple(
            _format_weight(lora.min_weight + (lora.max_weight - lora.min_weight) * value)
            for value in axis.multipliers
        )


class CompositionSession:
    """Consumes render results by task ID and pastes them without retaining source images."""

    def __init__(self, compositor: LoraComparisonCompositor) -> None:
        self.compositor = compositor
        self.plan = compositor.plan
        self.options = compositor.options
        self.style = compositor.style
        self.geometry = compositor.geometry
        self.fonts = compositor.fonts
        self._lock = threading.RLock()
        self._submitted: set[str] = set()
        self._canvas = self._create_canvas()
        self._draw = ImageDraw.Draw(self._canvas)
        self._draw_template()

    @property
    def submitted_count(self) -> int:
        return len(self._submitted)

    @property
    def completion(self) -> float:
        return self.submitted_count / self.plan.unique_task_count

    @property
    def pending_tasks(self) -> tuple[RenderTask, ...]:
        return tuple(task for task in self.plan.tasks if task.task_id not in self._submitted)

    @property
    def next_task(self) -> RenderTask | None:
        return next((task for task in self.plan.tasks if task.task_id not in self._submitted), None)

    def submit(
        self,
        image: ImageInput,
        *,
        task_id: str | None = None,
        replace_existing: bool = False,
    ) -> RenderTask:
        with self._lock:
            if task_id is None:
                task = self.next_task
                if task is None:
                    raise ValueError("All planned render tasks have already been submitted")
            else:
                task = self.plan.task(task_id)
            if task.task_id in self._submitted and not replace_existing:
                raise ValueError(f"Render task has already been submitted: {task.task_id}")

            prepared = self._prepare_image(image)
            for coordinate in self.plan.placements_for(task.task_id):
                geometry = self.geometry.cell(coordinate)
                self._canvas.paste(prepared, (geometry.image_rect[0], geometry.image_rect[1]))
                self._draw_image_frame(task, geometry.image_rect)
                self._draw_special_cell_label(task, coordinate)
            self._submitted.add(task.task_id)
            return task

    def finalize(self, *, strict: bool = True) -> Image.Image:
        with self._lock:
            missing = self.pending_tasks
            if strict and missing:
                preview = ", ".join(task.task_id for task in missing[:5])
                suffix = "..." if len(missing) > 5 else ""
                raise ValueError(
                    f"Composition is missing {len(missing)} render task(s): {preview}{suffix}"
                )
            result = self._canvas.copy()
            decorator = get_style_decorator(self.style.decorator)
            decorator.draw_foreground(
                DecorationContext(
                    canvas=result,
                    draw=ImageDraw.Draw(result),
                    plan=self.plan,
                    geometry=self.geometry,
                    style=self.style,
                )
            )
            return result

    def _create_canvas(self) -> Image.Image:
        canvas = Image.new("RGB", self.geometry.canvas_size, self.style.background_color)
        if self.style.background_image is None or self.style.background_opacity <= 0.0:
            return canvas
        source = self._load_background(self.style.background_image)
        fitted = self._fit_background(source, canvas.size, self.style.background_fit)
        if self.style.background_opacity >= 1.0:
            return fitted
        return Image.blend(canvas, fitted, self.style.background_opacity)

    def _draw_template(self) -> None:
        decorator = get_style_decorator(self.style.decorator)
        decorator.draw_background(
            DecorationContext(
                canvas=self._canvas,
                draw=self._draw,
                plan=self.plan,
                geometry=self.geometry,
                style=self.style,
            )
        )

        if self.style.show_axis_labels:
            self._draw_axes()

        for cell in self.plan.cells:
            if not cell.occupied:
                continue
            geometry = self.geometry.cell(cell.coordinate)
            task = self.plan.task(cell.task_id or "")
            if geometry.caption_rect is not None:
                self._draw_caption(task, cell.coordinate, geometry.caption_rect)
            self._draw.rectangle(_inclusive(geometry.image_rect), fill=self.style.placeholder_color)
            self._draw_image_frame(task, geometry.image_rect)
            self._draw_special_cell_label(task, cell.coordinate)

        if self.style.show_region_frames:
            self._draw_regions()
        if self.geometry.footer_rect is not None:
            self._draw_footer(self.geometry.footer_rect)

    def _draw_axes(self) -> None:
        for axis in self.plan.axes:
            rect = self.geometry.axis_rects.get(axis.key)
            if rect is None:
                continue
            if axis.side in {"top", "bottom"}:
                self._draw_horizontal_axis(axis, rect)
            else:
                self._draw_vertical_axis(axis, rect)

    def _draw_horizontal_axis(self, axis: AxisSpec, rect: Rect) -> None:
        padding = self.geometry.axis_padding
        title_height = _font_line_height(self.fonts.get(self.geometry.axis_font_size, bold=True))
        value_height = _font_line_height(
            self.fonts.get(self.geometry.axis_value_font_size, bold=True)
        )
        color = self.style.accent_colors[SLOTS.index(axis.slot)]
        title = f"{axis.slot} / {self.plan.loras[SLOTS.index(axis.slot)].display_name}"
        if axis.side == "top":
            title_rect = (rect[0], rect[1] + padding, rect[2], rect[1] + padding + title_height)
            tick_top = rect[3] - padding - value_height
            line_y = rect[3] - 1
        else:
            tick_top = rect[1] + padding
            title_rect = (
                rect[0],
                rect[3] - padding - title_height,
                rect[2],
                rect[3] - padding,
            )
            line_y = rect[1]
        self._draw_fitted_text(
            title,
            title_rect,
            font_size=self.geometry.axis_font_size,
            color=color,
            bold=True,
            align="center",
        )
        labels = self.compositor._axis_value_labels(axis)
        for position, label in zip(axis.positions, labels):
            cell = self.geometry.cell((position, self.plan.y_positions[0]))
            tick_rect = (
                cell.image_rect[0],
                tick_top,
                cell.image_rect[2],
                tick_top + value_height,
            )
            self._draw_fitted_text(
                label,
                tick_rect,
                font_size=self.geometry.axis_value_font_size,
                color=self.style.text_color,
                bold=True,
                align="center",
            )
        self._draw.line((rect[0], line_y, rect[2] - 1, line_y), fill=color, width=max(1, self.style.frame_width))

    def _draw_vertical_axis(self, axis: AxisSpec, rect: Rect) -> None:
        padding = self.geometry.axis_padding
        name_rail = (
            _font_line_height(self.fonts.get(self.geometry.axis_font_size, bold=True))
            + padding * 2
        )
        color = self.style.accent_colors[SLOTS.index(axis.slot)]
        title = f"{axis.slot} / {self.plan.loras[SLOTS.index(axis.slot)].display_name}"
        if axis.side == "left":
            name_rect = (rect[0], rect[1], rect[0] + name_rail, rect[3])
            value_rect = (name_rect[2], rect[1], rect[2], rect[3])
            line_x = rect[2] - 1
            clockwise = False
        else:
            name_rect = (rect[2] - name_rail, rect[1], rect[2], rect[3])
            value_rect = (rect[0], rect[1], name_rect[0], rect[3])
            line_x = rect[0]
            clockwise = True
        self._draw_vertical_text(
            title,
            name_rect,
            font_size=self.geometry.axis_font_size,
            color=color,
            clockwise=clockwise,
        )
        labels = self.compositor._axis_value_labels(axis)
        for position, label in zip(axis.positions, labels):
            cell = self.geometry.cell((self.plan.x_positions[0], position))
            tick_rect = (
                value_rect[0],
                cell.image_rect[1],
                value_rect[2],
                cell.image_rect[3],
            )
            self._draw_vertical_text(
                label,
                tick_rect,
                font_size=self.geometry.axis_value_font_size,
                color=self.style.text_color,
                clockwise=clockwise,
            )
        self._draw.line((line_x, rect[1], line_x, rect[3] - 1), fill=color, width=max(1, self.style.frame_width))

    def _draw_vertical_text(
        self,
        text: str,
        rect: Rect,
        *,
        font_size: int,
        color: RGBColor,
        clockwise: bool,
    ) -> None:
        available = max(1, rect[3] - rect[1] - self.geometry.axis_padding * 2)
        font = self.fonts.get(font_size, bold=True)
        fitted = self._ellipsize(text, font, available)
        box = font.getbbox(fitted)
        width = max(1, box[2] - box[0])
        height = max(1, box[3] - box[1])
        label = Image.new("RGBA", (width + 4, height + 4), (0, 0, 0, 0))
        label_draw = ImageDraw.Draw(label)
        label_draw.text((2 - box[0], 2 - box[1]), fitted, fill=(*color, 255), font=font)
        rotated = label.rotate(-90 if clockwise else 90, expand=True)
        x = rect[0] + max(0, (rect[2] - rect[0] - rotated.width) // 2)
        y = rect[1] + max(0, (rect[3] - rect[1] - rotated.height) // 2)
        self._canvas.paste(rotated, (x, y), rotated)

    def _draw_special_cell_label(self, task: RenderTask, coordinate: Coordinate) -> None:
        if (
            not self.style.show_special_cell_labels
            or self.style.show_cell_captions
            or task.kind != "triple"
        ):
            return
        geometry = self.geometry.cell(coordinate)
        label_rect = geometry.special_label_rect
        if label_rect is None:
            return
        self._draw.rectangle(_inclusive(label_rect), fill=self.style.panel_color)
        color = self._task_accent(task)
        accent_width = max(3, min(12, self.options.image_width // 64))
        self._draw.rectangle(
            (label_rect[0], label_rect[1], label_rect[0] + accent_width - 1, label_rect[3] - 1),
            fill=color,
        )
        padding = max(8, self.geometry.axis_padding)
        self._draw_fitted_text(
            task.caption,
            (
                label_rect[0] + accent_width + padding,
                label_rect[1],
                label_rect[2] - padding,
                label_rect[3],
            ),
            font_size=self.geometry.axis_value_font_size,
            color=self.style.text_color,
            bold=True,
        )

    def _draw_image_frame(self, task: RenderTask, rect: Rect) -> None:
        is_triple = task.kind == "triple"
        self._draw.rectangle(
            _inclusive(rect),
            outline=self._task_accent(task) if is_triple else self.style.frame_color,
            width=max(
                1,
                self.style.frame_width if is_triple else self.style.cell_frame_width,
            ),
        )

    def _draw_caption(self, task: RenderTask, coordinate: Coordinate, rect: Rect) -> None:
        self._draw.rectangle(_inclusive(rect), fill=self.style.panel_color)
        color = self._task_accent(task)
        accent_width = max(3, min(7, self.options.image_width // 64))
        self._draw.rectangle((rect[0], rect[1], rect[0] + accent_width - 1, rect[3] - 1), fill=color)
        caption = task.caption
        if self.style.show_coordinates:
            x, y = coordinate
            caption = f"[{x:+d},{y:+d}]  {caption}"
        padding = max(7, (rect[3] - rect[1]) // 6)
        text_rect = (rect[0] + accent_width + padding, rect[1], rect[2] - padding, rect[3])
        font_size = self.style.small_font_size or max(9, min(20, (rect[3] - rect[1]) // 2))
        self._draw_fitted_text(
            caption,
            text_rect,
            font_size=font_size,
            color=self.style.text_color,
            bold=True,
        )

    def _draw_regions(self) -> None:
        small_size = self.style.small_font_size or max(9, min(18, self.geometry.caption_height // 2 or 11))
        font = self.fonts.get(small_size, bold=True)
        for region in self.plan.regions:
            if region.key == "ABC":
                continue
            rect = self.geometry.region_rects[region.key]
            color = self._slots_accent(region.slots)
            self._draw.rectangle(
                _inclusive(rect),
                outline=color,
                width=max(1, self.style.frame_width),
            )
            if not self.style.show_region_labels:
                continue
            text_box = self._draw.textbbox((0, 0), region.label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            label_left = rect[0] + max(6, self.style.frame_width * 2)
            label_top = max(0, rect[1] - text_height - 6)
            label_rect = (
                label_left - 4,
                label_top - 2,
                label_left + text_width + 4,
                label_top + text_height + 3,
            )
            self._draw.rectangle(_inclusive(label_rect), fill=self.style.background_color)
            self._draw.text((label_left, label_top - text_box[1]), region.label, fill=color, font=font)

    def _draw_footer(self, rect: Rect) -> None:
        self._draw.rectangle(_inclusive(rect), fill=self.style.panel_color)
        self._draw.rectangle(
            _inclusive(rect),
            outline=self.style.frame_color,
            width=max(1, self.style.frame_width),
        )
        count = len(self.plan.loras)
        width = rect[2] - rect[0]
        column_width = width / count
        font_size = self.geometry.axis_font_size
        small_size = self.geometry.axis_value_font_size
        font = self.fonts.get(font_size, bold=True)
        small_font = self.fonts.get(small_size)
        for index, lora in enumerate(self.plan.loras):
            left = round(rect[0] + index * column_width)
            right = round(rect[0] + (index + 1) * column_width)
            if index:
                self._draw.line((left, rect[1], left, rect[3] - 1), fill=self.style.frame_color, width=1)
            padding = self.geometry.footer_padding
            title = (
                f"{SLOTS[index]} / MIN {_format_weight(lora.min_weight)}"
                f" / MAX {_format_weight(lora.max_weight)}"
            )
            title_y = rect[1] + padding
            self._draw.text((left + padding, title_y), title, fill=self.style.accent_colors[index], font=font)
            name_y = title_y + _font_line_height(font) + self.geometry.footer_title_gap
            available = max(1, right - left - padding * 2)
            for line in _wrap_text_pixels(lora.footer_text, small_font, available):
                self._draw.text((left + padding, name_y), line, fill=self.style.text_color, font=small_font)
                name_y += _font_line_height(small_font) + self.geometry.footer_name_line_gap

    def _draw_fitted_text(
        self,
        text: str,
        rect: Rect,
        *,
        font_size: int,
        color: RGBColor,
        bold: bool = False,
        align: str = "left",
    ) -> None:
        width, height = _rect_size(rect)
        if width <= 0 or height <= 0:
            return
        minimum = 7
        selected = self.fonts.get(max(minimum, font_size), bold=bold)
        fitted = str(text)
        for size in range(max(minimum, font_size), minimum - 1, -1):
            candidate = self.fonts.get(size, bold=bold)
            box = self._draw.textbbox((0, 0), fitted, font=candidate)
            if box[2] - box[0] <= width and box[3] - box[1] <= height:
                selected = candidate
                break
        fitted = self._ellipsize(fitted, selected, width)
        box = self._draw.textbbox((0, 0), fitted, font=selected)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        if align == "center":
            x = rect[0] + max(0, (width - text_width) // 2) - box[0]
        elif align == "right":
            x = rect[2] - text_width - box[0]
        elif align == "left":
            x = rect[0] - box[0]
        else:
            raise ValueError(f"Unknown text alignment: {align}")
        y = rect[1] + max(0, (height - text_height) // 2) - box[1]
        self._draw.text((x, y), fitted, fill=color, font=selected)

    def _ellipsize(self, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
        value = str(text)
        if self._draw.textlength(value, font=font) <= max_width:
            return value
        suffix = "..."
        if self._draw.textlength(suffix, font=font) > max_width:
            return ""
        low, high = 0, len(value)
        while low < high:
            middle = (low + high + 1) // 2
            if self._draw.textlength(value[:middle] + suffix, font=font) <= max_width:
                low = middle
            else:
                high = middle - 1
        return value[:low] + suffix

    def _task_accent(self, task: RenderTask) -> RGBColor:
        return self._slots_accent(task.active_slots) if task.active_slots else self.style.frame_color

    def _slots_accent(self, slots: Sequence[str]) -> RGBColor:
        colors = [self.style.accent_colors[SLOTS.index(slot)] for slot in slots]
        if not colors:
            return self.style.frame_color
        red = round(sum(color[0] for color in colors) / len(colors))
        green = round(sum(color[1] for color in colors) / len(colors))
        blue = round(sum(color[2] for color in colors) / len(colors))
        return (red, green, blue)

    def _prepare_image(self, image: ImageInput) -> Image.Image:
        converted = image_to_pil(image, alpha_background=self.style.placeholder_color)
        target = (self.options.image_width, self.options.image_height)
        if converted.size == target:
            return converted
        if self.options.image_fit == "strict":
            raise ValueError(f"Expected image size {target}, received {converted.size}")
        if self.options.image_fit == "stretch":
            return converted.resize(target, Image.Resampling.LANCZOS)
        if self.options.image_fit == "cover":
            return ImageOps.fit(converted, target, method=Image.Resampling.LANCZOS)
        contained = ImageOps.contain(converted, target, method=Image.Resampling.LANCZOS)
        result = Image.new("RGB", target, self.style.placeholder_color)
        result.paste(contained, ((target[0] - contained.width) // 2, (target[1] - contained.height) // 2))
        return result

    def _load_background(self, source: object) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, (str, Path)):
            path = Path(source).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Style background image does not exist: {path}")
            with Image.open(path) as image:
                return image.convert("RGB")
        return image_to_pil(source, alpha_background=self.style.background_color)

    def _fit_background(self, source: Image.Image, target: tuple[int, int], mode: str) -> Image.Image:
        if mode == "stretch":
            return source.resize(target, Image.Resampling.LANCZOS)
        if mode == "cover":
            return ImageOps.fit(source, target, method=Image.Resampling.LANCZOS)
        if mode == "contain":
            contained = ImageOps.contain(source, target, method=Image.Resampling.LANCZOS)
            result = Image.new("RGB", target, self.style.background_color)
            result.paste(contained, ((target[0] - contained.width) // 2, (target[1] - contained.height) // 2))
            return result
        result = Image.new("RGB", target, self.style.background_color)
        for y in range(0, target[1], source.height):
            for x in range(0, target[0], source.width):
                result.paste(source, (x, y))
        return result


def image_to_pil(image: ImageInput, *, alpha_background: RGBColor = (0, 0, 0)) -> Image.Image:
    if isinstance(image, Image.Image):
        converted = image.copy()
    else:
        value = image
        if hasattr(value, "detach") and callable(value.detach):
            value = value.detach()
        if hasattr(value, "cpu") and callable(value.cpu):
            value = value.cpu()
        if hasattr(value, "numpy") and callable(value.numpy):
            value = value.numpy()
        array = np.asarray(value)
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError(
                    "A single submitted image cannot contain a batch; iterate the batch in task order"
                )
            array = array[0]
        if array.ndim == 3 and array.shape[-1] not in {1, 3, 4} and array.shape[0] in {1, 3, 4}:
            array = np.moveaxis(array, 0, -1)
        if array.ndim not in {2, 3}:
            raise ValueError(f"Unsupported image array shape: {array.shape}")
        if array.ndim == 3 and array.shape[-1] not in {1, 3, 4}:
            raise ValueError(f"Unsupported image channel count: {array.shape[-1]}")
        if np.issubdtype(array.dtype, np.floating):
            array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
            if array.size == 0:
                raise ValueError("Image array cannot be empty")
            if float(array.min()) >= 0.0 and float(array.max()) <= 1.00001:
                array = array * 255.0
        if array.dtype == np.bool_:
            array = array.astype(np.uint8) * 255
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        converted = Image.fromarray(array)

    has_alpha = converted.mode in {"RGBA", "LA"} or (
        converted.mode == "P" and "transparency" in converted.info
    )
    if has_alpha:
        base = Image.new("RGBA", converted.size, (*alpha_background, 255))
        return Image.alpha_composite(base, converted.convert("RGBA")).convert("RGB")
    return converted.convert("RGB")


__all__ = [
    "CellGeometry",
    "CompositionOptions",
    "CompositionSession",
    "ImageInput",
    "LayoutGeometry",
    "LoraComparisonCompositor",
    "image_to_pil",
]
