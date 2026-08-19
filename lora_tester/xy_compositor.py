from __future__ import annotations

import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .compositor import FontResolver, image_to_pil
from .styles import DecorationContext, RGBColor, StyleConfig, get_style_decorator
from .xy import DetailBlock, XYAxis


Rect: TypeAlias = tuple[int, int, int, int]
Coordinate: TypeAlias = tuple[int, int]
ImageInput: TypeAlias = Image.Image | Any


def _inclusive(rect: Rect) -> Rect:
    left, top, right, bottom = rect
    return (left, top, max(left, right - 1), max(top, bottom - 1))


def _line_height(font: ImageFont.ImageFont) -> int:
    box = font.getbbox("Ag")
    return max(1, box[3] - box[1] + 2)


@dataclass(frozen=True, slots=True)
class XYMatrixOptions:
    image_width: int
    image_height: int
    image_fit: str = "strict"
    show_details: bool = True
    max_canvas_pixels: int | None = 150_000_000
    reserve_artist_mixer_labels: bool = False
    x_group_gap: int | None = None

    def __post_init__(self) -> None:
        if int(self.image_width) <= 0 or int(self.image_height) <= 0:
            raise ValueError("image_width and image_height must be positive")
        if self.image_fit not in {"strict", "contain", "cover", "stretch"}:
            raise ValueError("image_fit must be strict, contain, cover, or stretch")
        if self.max_canvas_pixels is not None and int(self.max_canvas_pixels) <= 0:
            raise ValueError("max_canvas_pixels must be positive or None")
        if self.x_group_gap is not None and int(self.x_group_gap) < 0:
            raise ValueError("x_group_gap must be non-negative or None")


@dataclass(frozen=True, slots=True)
class DetailGeometry:
    block: DetailBlock
    rect: Rect
    title_rect: Rect
    content_rect: Rect


@dataclass(frozen=True, slots=True)
class XYMatrixGeometry:
    image_width: int
    image_height: int
    canvas_size: tuple[int, int]
    grid_bounds: Rect
    cells: Mapping[Coordinate, Rect]
    column_label_rects: Mapping[int, Rect]
    row_label_rects: Mapping[int, Rect]
    x_title_rect: Rect
    y_title_rect: Rect
    detail_blocks: tuple[DetailGeometry, ...]
    artist_mixer_label_rects: Mapping[Coordinate, Rect]
    header_font_size: int
    label_font_size: int

    def cell(self, row: int, column: int) -> Rect:
        return self.cells[(int(row), int(column))]

    def artist_mixer_label(self, row: int, column: int) -> Rect | None:
        return self.artist_mixer_label_rects.get((int(row), int(column)))


class XYMatrixCompositor:
    """Render any pair of grouped XY axes without knowing their sampling semantics."""

    def __init__(
        self,
        x_axis: XYAxis,
        y_axis: XYAxis,
        image_width: int,
        image_height: int,
        *,
        style: StyleConfig | None = None,
        show_details: bool = True,
        image_fit: str = "strict",
        max_canvas_pixels: int | None = 150_000_000,
        reserve_artist_mixer_labels: bool = False,
        extra_detail_text: str = "",
        x_group_gap: int | None = None,
    ) -> None:
        if not isinstance(x_axis, XYAxis) or not isinstance(y_axis, XYAxis):
            raise TypeError("x_axis and y_axis must come from XY Axis nodes")
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.style = style or StyleConfig.black()
        self.options = XYMatrixOptions(
            image_width=int(image_width),
            image_height=int(image_height),
            image_fit=image_fit,
            show_details=bool(show_details),
            max_canvas_pixels=max_canvas_pixels,
            reserve_artist_mixer_labels=bool(reserve_artist_mixer_labels),
            x_group_gap=None if x_group_gap is None else int(x_group_gap),
        )
        self.extra_detail_text = str(extra_detail_text).strip()
        self.fonts = FontResolver(self.style.font_path)
        self.geometry = self._build_geometry()
        pixels = self.geometry.canvas_size[0] * self.geometry.canvas_size[1]
        if self.options.max_canvas_pixels is not None and pixels > self.options.max_canvas_pixels:
            raise MemoryError(
                f"Composite canvas would contain {pixels:,} pixels; reduce image size or axis counts."
            )

    @property
    def column_count(self) -> int:
        return len(self.x_axis.entries)

    @property
    def row_count(self) -> int:
        return len(self.y_axis.entries)

    @property
    def detail_blocks(self) -> tuple[DetailBlock, ...]:
        blocks: tuple[DetailBlock, ...] = ()
        if self.options.show_details:
            blocks = (*self.x_axis.detail_blocks, *self.y_axis.detail_blocks)
        if self.extra_detail_text:
            blocks = (*blocks, DetailBlock("NOTES", "text", text=(self.extra_detail_text,)))
        return blocks

    def start(self) -> "XYMatrixSession":
        return XYMatrixSession(self)

    def render_template(self) -> Image.Image:
        return self.start().finalize(strict=False)

    def compose(
        self,
        images: Sequence[ImageInput] | Mapping[Coordinate, ImageInput] | Iterable[ImageInput],
        *,
        strict: bool = True,
    ) -> Image.Image:
        session = self.start()
        if isinstance(images, Mapping):
            for coordinate, image in images.items():
                session.submit(image, coordinate=coordinate)
        else:
            iterator = iter(images)
            for row in range(self.row_count):
                for column in range(self.column_count):
                    try:
                        image = next(iterator)
                    except StopIteration:
                        return session.finalize(strict=strict)
                    session.submit(image, coordinate=(row, column))
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError("Image sequence contains more items than the XY matrix")
        return session.finalize(strict=strict)

    def _detail_height(self, block: DetailBlock, font: ImageFont.ImageFont, width: int) -> int:
        line_height = _line_height(font)
        padding = max(8, self.style.footer_padding)
        title_height = line_height + padding
        if block.mode == "table":
            content_lines = 1 + max(1, len(block.rows))
        else:
            approximate_chars = max(24, width // max(6, font.getlength("M")))
            content_lines = sum(
                max(1, len(textwrap.wrap(line, width=int(approximate_chars))))
                for line in block.text
            )
        return title_height + content_lines * line_height + padding * 2

    def _build_geometry(self) -> XYMatrixGeometry:
        style = self.style
        options = self.options
        header_font_size = int(
            style.font_size
            or max(14, min(48, round(min(options.image_width, options.image_height) * 0.07)))
        )
        label_font_size = int(style.small_font_size or max(10, round(header_font_size * 0.58)))
        header_font = self.fonts.get(header_font_size, bold=True)
        label_font = self.fonts.get(label_font_size, bold=True)
        cell_gap = max(0, int(style.cell_gap))
        automatic_group_gap = max(
            int(style.region_gap),
            round(min(options.image_width, options.image_height) * 0.08),
        )
        if options.x_group_gap is None:
            x_group_gap = automatic_group_gap
        elif int(options.x_group_gap) == 0:
            # Legacy Style Combination Tester semantics: zero means a visible
            # BASE/control separation, scaled from the image width.
            x_group_gap = max(int(style.region_gap), round(options.image_width * 0.125))
        else:
            x_group_gap = int(options.x_group_gap)
        group_gap = max(cell_gap, x_group_gap)
        padding = max(8, int(style.footer_padding))
        row_label_width = max(64, min(144, round(options.image_width * 0.16)))
        x_title_height = _line_height(label_font) + padding
        column_label_height = max(42, _line_height(header_font) + padding * 2)
        artist_label_height = (
            _line_height(label_font) + padding * 2
            if options.reserve_artist_mixer_labels
            else 0
        )

        left = style.outer_margin + row_label_width
        top = style.outer_margin + x_title_height + column_label_height
        x_origins: list[int] = []
        cursor_x = left
        for index in range(self.column_count):
            if index:
                cursor_x += group_gap if index in self.x_axis.group_breaks else cell_gap
            x_origins.append(cursor_x)
            cursor_x += options.image_width
        grid_right = cursor_x

        y_origins: list[int] = []
        row_step = options.image_height + artist_label_height
        cursor_y = top
        for index in range(self.row_count):
            if index:
                cursor_y += group_gap if index in self.y_axis.group_breaks else cell_gap
            y_origins.append(cursor_y)
            cursor_y += row_step
        grid_bottom = cursor_y

        cells: dict[Coordinate, Rect] = {}
        artist_rects: dict[Coordinate, Rect] = {}
        for row, y in enumerate(y_origins):
            for column, x in enumerate(x_origins):
                cells[(row, column)] = (x, y, x + options.image_width, y + options.image_height)
                if artist_label_height:
                    artist_rects[(row, column)] = (
                        x,
                        y + options.image_height,
                        x + options.image_width,
                        y + options.image_height + artist_label_height,
                    )

        column_label_rects = {
            column: (
                x,
                style.outer_margin + x_title_height,
                x + options.image_width,
                top,
            )
            for column, x in enumerate(x_origins)
        }
        row_label_rects = {
            row: (
                style.outer_margin,
                y,
                left - max(4, padding // 2),
                y + options.image_height,
            )
            for row, y in enumerate(y_origins)
        }
        x_title_rect = (left, style.outer_margin, grid_right, style.outer_margin + x_title_height)
        y_title_rect = (style.outer_margin, style.outer_margin, left - padding // 2, top)

        detail_geometries: list[DetailGeometry] = []
        canvas_bottom = grid_bottom + style.outer_margin
        if self.detail_blocks:
            detail_width = max(1, grid_right - left)
            detail_top = grid_bottom + max(style.footer_gap, padding)
            detail_font = self.fonts.get(label_font_size)
            for block in self.detail_blocks:
                height = self._detail_height(block, detail_font, detail_width)
                rect = (left, detail_top, grid_right, detail_top + height)
                title_height = _line_height(self.fonts.get(label_font_size, bold=True)) + padding
                detail_geometries.append(
                    DetailGeometry(
                        block=block,
                        rect=rect,
                        title_rect=(rect[0] + padding, rect[1], rect[2] - padding, rect[1] + title_height),
                        content_rect=(rect[0] + padding, rect[1] + title_height, rect[2] - padding, rect[3] - padding),
                    )
                )
                detail_top = rect[3] + max(cell_gap, padding // 2)
            canvas_bottom = detail_geometries[-1].rect[3] + style.outer_margin

        return XYMatrixGeometry(
            image_width=options.image_width,
            image_height=options.image_height,
            canvas_size=(grid_right + style.outer_margin, canvas_bottom),
            grid_bounds=(left, top, grid_right, grid_bottom),
            cells=MappingProxyType(cells),
            column_label_rects=MappingProxyType(column_label_rects),
            row_label_rects=MappingProxyType(row_label_rects),
            x_title_rect=x_title_rect,
            y_title_rect=y_title_rect,
            detail_blocks=tuple(detail_geometries),
            artist_mixer_label_rects=MappingProxyType(artist_rects),
            header_font_size=header_font_size,
            label_font_size=label_font_size,
        )


class XYMatrixSession:
    def __init__(self, compositor: XYMatrixCompositor) -> None:
        self.compositor = compositor
        self.geometry = compositor.geometry
        self.style = compositor.style
        self.fonts = compositor.fonts
        self._submitted: set[Coordinate] = set()
        self._finalized = False
        self._canvas = self._create_canvas()
        self._draw = ImageDraw.Draw(self._canvas)
        self._draw_template()

    @property
    def submitted_count(self) -> int:
        return len(self._submitted)

    @property
    def expected_count(self) -> int:
        return self.compositor.row_count * self.compositor.column_count

    def submit(
        self,
        image: ImageInput,
        *,
        coordinate: Coordinate,
        replace_existing: bool = False,
        artist_mixer: bool = False,
    ) -> None:
        if self._finalized:
            raise RuntimeError("Cannot submit images after the XY composition is finalized")
        row, column = int(coordinate[0]), int(coordinate[1])
        if not (0 <= row < self.compositor.row_count and 0 <= column < self.compositor.column_count):
            raise ValueError(f"XY coordinate is outside the layout: {(row, column)}")
        key = (row, column)
        if key in self._submitted and not replace_existing:
            raise ValueError(f"XY cell has already been submitted: {key}")
        prepared = self._prepare_image(image)
        try:
            rect = self.geometry.cell(row, column)
            self._canvas.paste(prepared, (rect[0], rect[1]))
            self._draw_frame(row, column, rect)
            if artist_mixer:
                self._draw_artist_label(row, column)
            self._submitted.add(key)
        finally:
            prepared.close()

    def finalize(self, *, strict: bool = True) -> Image.Image:
        if strict and len(self._submitted) != self.expected_count:
            raise ValueError(
                f"Composition is missing {self.expected_count - len(self._submitted)} image(s)"
            )
        if not self._finalized:
            decorator = get_style_decorator(self.style.decorator)
            decorator.draw_foreground(
                DecorationContext(
                    canvas=self._canvas,
                    draw=self._draw,
                    plan=self.compositor,
                    geometry=self.geometry,
                    style=self.style,
                )
            )
            self._finalized = True
        # Ownership is transferred to the caller; no full-canvas copy is made.
        return self._canvas

    def _create_canvas(self) -> Image.Image:
        canvas = Image.new("RGB", self.geometry.canvas_size, self.style.background_color)
        source = self.style.background_image
        if source is None or self.style.background_opacity <= 0:
            return canvas
        if isinstance(source, Image.Image):
            image = source.convert("RGB")
        elif isinstance(source, (str, Path)):
            path = Path(source).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Style background image does not exist: {path}")
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        else:
            image = image_to_pil(source, alpha_background=self.style.background_color)
        target = self.geometry.canvas_size
        fitted: Image.Image | None = None
        mask: Image.Image | None = None
        try:
            if self.style.background_fit == "stretch":
                fitted = image.resize(target, Image.Resampling.LANCZOS)
            elif self.style.background_fit == "cover":
                fitted = ImageOps.fit(image, target, method=Image.Resampling.LANCZOS)
            elif self.style.background_fit == "contain":
                fitted = Image.new("RGB", target, self.style.background_color)
                contained = ImageOps.contain(image, target, method=Image.Resampling.LANCZOS)
                try:
                    fitted.paste(
                        contained,
                        ((target[0] - contained.width) // 2, (target[1] - contained.height) // 2),
                    )
                finally:
                    contained.close()
            else:
                fitted = Image.new("RGB", target, self.style.background_color)
                for y in range(0, target[1], image.height):
                    for x in range(0, target[0], image.width):
                        fitted.paste(image, (x, y))
            opacity = float(self.style.background_opacity)
            if opacity >= 1.0:
                canvas.paste(fitted, (0, 0))
            else:
                mask = Image.new("L", target, round(opacity * 255))
                canvas.paste(fitted, (0, 0), mask)
            return canvas
        finally:
            if mask is not None:
                mask.close()
            if fitted is not None:
                fitted.close()
            image.close()

    def _draw_template(self) -> None:
        decorator = get_style_decorator(self.style.decorator)
        decorator.draw_background(
            DecorationContext(
                canvas=self._canvas,
                draw=self._draw,
                plan=self.compositor,
                geometry=self.geometry,
                style=self.style,
            )
        )
        self._draw_fitted_text(
            f"X / {self.compositor.x_axis.title}",
            self.geometry.x_title_rect,
            self.geometry.label_font_size,
            self.style.muted_text_color,
            bold=True,
        )
        self._draw_fitted_text(
            f"Y / {self.compositor.y_axis.title}",
            self.geometry.y_title_rect,
            self.geometry.label_font_size,
            self.style.muted_text_color,
            bold=True,
        )
        for column, entry in enumerate(self.compositor.x_axis.entries):
            color = self.style.accent_colors[column % len(self.style.accent_colors)]
            self._draw_fitted_text(
                entry.label,
                self.geometry.column_label_rects[column],
                self.geometry.header_font_size,
                color,
                bold=True,
                align="center",
            )
        for row, entry in enumerate(self.compositor.y_axis.entries):
            self._draw_fitted_text(
                entry.label,
                self.geometry.row_label_rects[row],
                self.geometry.label_font_size,
                self.style.text_color,
                bold=True,
                align="center",
            )
        for row in range(self.compositor.row_count):
            for column in range(self.compositor.column_count):
                rect = self.geometry.cell(row, column)
                self._draw.rectangle(_inclusive(rect), fill=self.style.placeholder_color)
                self._draw_frame(row, column, rect)
                label_rect = self.geometry.artist_mixer_label(row, column)
                if label_rect is not None:
                    self._draw.rectangle(_inclusive(label_rect), fill=self.style.panel_color)
        for detail in self.geometry.detail_blocks:
            self._draw_detail_block(detail)

    def _draw_detail_block(self, detail: DetailGeometry) -> None:
        self._draw.rectangle(
            _inclusive(detail.rect),
            fill=self.style.panel_color,
            outline=self.style.frame_color,
            width=max(1, self.style.frame_width),
        )
        accent_width = max(4, min(14, self.geometry.label_font_size // 2))
        self._draw.rectangle(
            (
                detail.rect[0],
                detail.rect[1],
                detail.rect[0] + accent_width - 1,
                detail.title_rect[3] - 1,
            ),
            fill=self.style.accent_colors[0],
        )
        self._draw_fitted_text(
            detail.block.title,
            detail.title_rect,
            self.geometry.label_font_size,
            self.style.text_color,
            bold=True,
        )
        if detail.block.mode == "table":
            self._draw_table(detail.block, detail.content_rect)
        else:
            self._draw_text_lines(detail.block.text, detail.content_rect)

    def _draw_table(self, block: DetailBlock, rect: Rect) -> None:
        rows = (block.headers, *block.rows)
        row_height = max(1, (rect[3] - rect[1]) // max(1, len(rows)))
        column_count = len(block.headers)
        column_width = (rect[2] - rect[0]) / column_count
        for row_index, row in enumerate(rows):
            top = rect[1] + row_index * row_height
            bottom = rect[3] if row_index == len(rows) - 1 else top + row_height
            if row_index:
                self._draw.line((rect[0], top, rect[2], top), fill=self.style.frame_color, width=1)
            for column_index, value in enumerate(row):
                left = round(rect[0] + column_index * column_width)
                right = round(rect[0] + (column_index + 1) * column_width)
                if column_index:
                    self._draw.line((left, top, left, bottom), fill=self.style.frame_color, width=1)
                self._draw_fitted_text(
                    value,
                    (left + 6, top, right - 6, bottom),
                    self.geometry.label_font_size,
                    self.style.muted_text_color if row_index else self.style.text_color,
                    bold=row_index == 0,
                )

    def _draw_text_lines(self, lines: Sequence[str], rect: Rect) -> None:
        font = self.fonts.get(self.geometry.label_font_size)
        line_height = _line_height(font)
        width = max(1, rect[2] - rect[0])
        approximate_chars = max(24, int(width // max(6, font.getlength("M"))))
        y = rect[1]
        for source in lines:
            for line in textwrap.wrap(str(source), width=approximate_chars) or [""]:
                if y + line_height > rect[3]:
                    return
                self._draw.text((rect[0], y), line, fill=self.style.text_color, font=font)
                y += line_height

    def _draw_frame(self, row: int, column: int, rect: Rect) -> None:
        color = self.style.accent_colors[column % len(self.style.accent_colors)]
        self._draw.rectangle(
            _inclusive(rect),
            outline=color,
            width=max(1, self.style.cell_frame_width),
        )

    def _draw_artist_label(self, row: int, column: int) -> None:
        rect = self.geometry.artist_mixer_label(row, column)
        if rect is None:
            raise RuntimeError("Artist Mixer label was not reserved for this XY matrix")
        self._draw.rectangle(_inclusive(rect), fill=self.style.panel_color)
        accent_width = max(3, min(12, self.compositor.options.image_width // 64))
        self._draw.rectangle(
            (rect[0], rect[1], rect[0] + accent_width - 1, rect[3] - 1),
            fill=self.style.accent_colors[0],
        )
        self._draw_fitted_text(
            "ANIMA ARTIST MIXER",
            (rect[0] + accent_width + 8, rect[1], rect[2] - 8, rect[3]),
            self.geometry.label_font_size,
            self.style.text_color,
            bold=True,
        )

    def _draw_fitted_text(
        self,
        text: str,
        rect: Rect,
        font_size: int,
        color: RGBColor,
        *,
        bold: bool = False,
        align: str = "left",
    ) -> None:
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        if width <= 0 or height <= 0:
            return
        value = str(text)
        selected = self.fonts.get(max(7, int(font_size)), bold=bold)
        for size in range(max(7, int(font_size)), 6, -1):
            candidate = self.fonts.get(size, bold=bold)
            box = self._draw.textbbox((0, 0), value, font=candidate)
            if box[2] - box[0] <= width and box[3] - box[1] <= height:
                selected = candidate
                break
        while value and self._draw.textbbox((0, 0), value, font=selected)[2] > width:
            value = value[:-1]
        if value != str(text) and width >= self._draw.textlength("...", font=selected):
            value = value.rstrip() + "..."
        box = self._draw.textbbox((0, 0), value, font=selected)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        if align == "center":
            x = rect[0] + (width - text_width) // 2 - box[0]
        elif align == "right":
            x = rect[2] - text_width - box[0]
        else:
            x = rect[0] - box[0]
        y = rect[1] + (height - text_height) // 2 - box[1]
        self._draw.text((x, y), value, fill=color, font=selected)

    def _prepare_image(self, image: ImageInput) -> Image.Image:
        converted = image_to_pil(image, alpha_background=self.style.placeholder_color)
        target = (self.compositor.options.image_width, self.compositor.options.image_height)
        if converted.size == target:
            return converted
        try:
            if self.compositor.options.image_fit == "strict":
                raise ValueError(f"Expected image size {target}, received {converted.size}")
            if self.compositor.options.image_fit == "stretch":
                return converted.resize(target, Image.Resampling.LANCZOS)
            if self.compositor.options.image_fit == "cover":
                return ImageOps.fit(converted, target, method=Image.Resampling.LANCZOS)
            contained = ImageOps.contain(converted, target, method=Image.Resampling.LANCZOS)
            result = Image.new("RGB", target, self.style.placeholder_color)
            try:
                result.paste(
                    contained,
                    ((target[0] - contained.width) // 2, (target[1] - contained.height) // 2),
                )
            finally:
                contained.close()
            return result
        finally:
            converted.close()


__all__ = [
    "DetailGeometry",
    "XYMatrixCompositor",
    "XYMatrixGeometry",
    "XYMatrixOptions",
    "XYMatrixSession",
]
