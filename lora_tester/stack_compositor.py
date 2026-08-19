from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .compositor import FontResolver, image_to_pil
from .stack import LoraStack, LoraStackItem
from .styles import DecorationContext, RGBColor, StyleConfig, get_style_decorator


Rect: TypeAlias = tuple[int, int, int, int]
Coordinate: TypeAlias = tuple[int, int]
ImageInput: TypeAlias = Image.Image | Any
DEFAULT_CONTROL_GAP_RATIO = 0.125


def _inclusive(rect: Rect) -> Rect:
    left, top, right, bottom = rect
    return (left, top, max(left, right - 1), max(top, bottom - 1))


def _font_line_height(font: ImageFont.ImageFont) -> int:
    box = font.getbbox("Ag")
    return max(1, box[3] - box[1] + 2)


def _axis_token(index: int) -> str:
    """Return spreadsheet-style labels: A, B, ..., Z, AA, AB, ..."""

    value = int(index) + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _stack_mix_labels(stacks: Sequence[LoraStack]) -> tuple[str, ...]:
    """Name columns by each complete LoRA configuration in the mix."""

    tokens: dict[LoraStackItem, str] = {}
    for stack in stacks:
        for item in stack.items:
            tokens.setdefault(item, _axis_token(len(tokens)))
    return tuple("+".join(tokens[item] for item in stack.items) for stack in stacks)


def _original_lora_items(stacks: Sequence[LoraStack]) -> tuple[LoraStackItem, ...]:
    """Return each distinct file, trigger-word, and strength configuration once."""

    items: list[LoraStackItem] = []
    seen: set[LoraStackItem] = set()
    for stack in stacks:
        for item in stack.items:
            if item not in seen:
                seen.add(item)
                items.append(item)
    return tuple(items)


def _prompt_axis_label(index: int) -> str:
    return f"Prompt {int(index) + 1}"


@dataclass(frozen=True, slots=True)
class StackMatrixOptions:
    image_width: int
    image_height: int
    image_fit: str = "strict"
    show_stack_details: bool = True
    max_canvas_pixels: int | None = 150_000_000
    control_gap: int | None = None
    reserve_artist_mixer_labels: bool = False

    def __post_init__(self) -> None:
        if int(self.image_width) <= 0 or int(self.image_height) <= 0:
            raise ValueError("image_width and image_height must be positive")
        if self.image_fit not in {"strict", "contain", "cover", "stretch"}:
            raise ValueError("image_fit must be strict, contain, cover, or stretch")
        if self.max_canvas_pixels is not None and int(self.max_canvas_pixels) <= 0:
            raise ValueError("max_canvas_pixels must be positive or None")


@dataclass(frozen=True, slots=True)
class StackMatrixGeometry:
    image_width: int
    image_height: int
    canvas_size: tuple[int, int]
    grid_bounds: Rect
    header_rect: Rect
    row_label_rects: Mapping[int, Rect]
    cells: Mapping[Coordinate, Rect]
    control_gap: int
    control_separator_x: int
    header_font_size: int
    row_font_size: int
    artist_mixer_label_height: int
    artist_mixer_label_rects: Mapping[Coordinate, Rect]

    def cell(self, row: int, column: int) -> Rect:
        return self.cells[(int(row), int(column))]

    def artist_mixer_label(self, row: int, column: int) -> Rect | None:
        return self.artist_mixer_label_rects.get((int(row), int(column)))


class LoraStackMatrixCompositor:
    """Compose prompt rows and LoRA stack columns into one labeled image."""

    def __init__(
        self,
        stacks: Sequence[LoraStack],
        prompts: Sequence[str],
        image_width: int,
        image_height: int,
        *,
        style: StyleConfig | None = None,
        show_stack_details: bool = True,
        image_fit: str = "strict",
        max_canvas_pixels: int | None = 150_000_000,
        control_gap: int | None = None,
        reserve_artist_mixer_labels: bool = False,
    ) -> None:
        self.stacks = tuple(stacks)
        self.prompts = tuple(str(prompt).strip() for prompt in prompts)
        if any(not isinstance(stack, LoraStack) for stack in self.stacks):
            raise TypeError("stacks must contain LoraStack values")
        if not self.prompts or any(not prompt for prompt in self.prompts):
            raise ValueError("At least one non-empty positive prompt is required")
        self.style = style or StyleConfig.black()
        self.options = StackMatrixOptions(
            image_width=int(image_width),
            image_height=int(image_height),
            image_fit=image_fit,
            show_stack_details=bool(show_stack_details),
            max_canvas_pixels=max_canvas_pixels,
            control_gap=control_gap,
            reserve_artist_mixer_labels=bool(reserve_artist_mixer_labels),
        )
        self.fonts = FontResolver(self.style.font_path)
        self.geometry = self._build_geometry()
        pixels = self.geometry.canvas_size[0] * self.geometry.canvas_size[1]
        if self.options.max_canvas_pixels is not None and pixels > self.options.max_canvas_pixels:
            raise MemoryError(
                f"Composite canvas would contain {pixels:,} pixels; "
                "reduce the image size, prompt count, or stack count."
            )

    @property
    def column_count(self) -> int:
        return len(self.stacks) + 1

    @property
    def row_count(self) -> int:
        return len(self.prompts)

    @property
    def original_lora_items(self) -> tuple[LoraStackItem, ...]:
        return _original_lora_items(self.stacks)

    def start(self) -> "LoraStackMatrixSession":
        return LoraStackMatrixSession(self)

    def render_template(self) -> Image.Image:
        return self.start().finalize(strict=False)

    def compose(self, images: Sequence[ImageInput] | Mapping[Coordinate, ImageInput] | Iterable[ImageInput], *, strict: bool = True) -> Image.Image:
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
                raise ValueError("Image sequence contains more items than the matrix")
        return session.finalize(strict=strict)

    def _build_geometry(self) -> StackMatrixGeometry:
        style = self.style
        options = self.options
        header_font_size = int(style.font_size or max(14, min(54, round(min(options.image_width, options.image_height) * 0.08))))
        row_font_size = int(style.small_font_size or max(11, min(28, round(header_font_size * 0.58))))
        header_font = self.fonts.get(header_font_size, bold=True)
        row_font = self.fonts.get(row_font_size, bold=True)
        # Prompt labels are rotated into a narrow side axis, so their length
        # must not expand the whole canvas horizontally.
        row_label_width = max(56, min(96, _font_line_height(row_font) + style.footer_padding * 2))
        label_gap = min(style.footer_padding, max(4, row_label_width // 4))
        header_height = max(48, _font_line_height(header_font) + style.footer_padding * 2)
        cell_gap = max(0, int(style.cell_gap))
        automatic_control_gap = max(
            int(style.region_gap),
            round(options.image_width * DEFAULT_CONTROL_GAP_RATIO),
        )
        control_gap = max(
            cell_gap,
            int(options.control_gap if options.control_gap is not None else automatic_control_gap),
        )
        artist_mixer_label_height = (
            _font_line_height(row_font) + style.footer_padding * 2
            if options.reserve_artist_mixer_labels
            else 0
        )
        row_step = options.image_height + artist_mixer_label_height + cell_gap
        left = style.outer_margin + row_label_width
        top = style.outer_margin + header_height
        cells: dict[Coordinate, Rect] = {}
        artist_mixer_label_rects: dict[Coordinate, Rect] = {}
        current_x = left
        for column in range(self.column_count):
            if column:
                current_x += control_gap if column == 1 else cell_gap
            for row in range(self.row_count):
                y = top + row * row_step
                cells[(row, column)] = (current_x, y, current_x + options.image_width, y + options.image_height)
                if artist_mixer_label_height:
                    artist_mixer_label_rects[(row, column)] = (
                        current_x,
                        y + options.image_height,
                        current_x + options.image_width,
                        y + options.image_height + artist_mixer_label_height,
                    )
            current_x += options.image_width
        grid_bounds = (
            left,
            top,
            current_x,
            top + self.row_count * (options.image_height + artist_mixer_label_height)
            + max(0, self.row_count - 1) * cell_gap,
        )
        row_label_rects = {
            row: (
                style.outer_margin,
                top + row * row_step,
                left - label_gap,
                top + row * row_step + options.image_height,
            )
            for row in range(self.row_count)
        }
        header_rect = (left, style.outer_margin, current_x, top)
        footer_height = 0
        if options.show_stack_details and self.original_lora_items:
            footer_height = max(48, style.footer_padding * 2 + _font_line_height(row_font) * 2)
        footer_top = grid_bounds[3] + (style.footer_gap if footer_height else 0)
        canvas_bottom = footer_top + footer_height + style.outer_margin
        canvas_size = (current_x + style.outer_margin, canvas_bottom)
        separator_x = left + options.image_width + control_gap // 2
        return StackMatrixGeometry(
            image_width=options.image_width,
            image_height=options.image_height,
            canvas_size=canvas_size,
            grid_bounds=grid_bounds,
            header_rect=header_rect,
            row_label_rects=MappingProxyType(row_label_rects),
            cells=MappingProxyType(cells),
            control_gap=control_gap,
            control_separator_x=separator_x,
            header_font_size=header_font_size,
            row_font_size=row_font_size,
            artist_mixer_label_height=artist_mixer_label_height,
            artist_mixer_label_rects=MappingProxyType(artist_mixer_label_rects),
        )


class LoraStackMatrixSession:
    def __init__(self, compositor: LoraStackMatrixCompositor) -> None:
        self.compositor = compositor
        self.geometry = compositor.geometry
        self.style = compositor.style
        self.fonts = compositor.fonts
        self._submitted: set[Coordinate] = set()
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
        row, column = int(coordinate[0]), int(coordinate[1])
        if not (0 <= row < self.compositor.row_count and 0 <= column < self.compositor.column_count):
            raise ValueError(f"Matrix coordinate is outside the layout: {(row, column)}")
        key = (row, column)
        if key in self._submitted and not replace_existing:
            raise ValueError(f"Matrix cell has already been submitted: {key}")
        prepared = self._prepare_image(image)
        try:
            rect = self.geometry.cell(row, column)
            self._canvas.paste(prepared, (rect[0], rect[1]))
            self._draw_image_frame(row, column, rect)
            if artist_mixer:
                self._draw_artist_mixer_label(row, column)
            self._submitted.add(key)
        finally:
            prepared.close()

    def finalize(self, *, strict: bool = True) -> Image.Image:
        if strict and len(self._submitted) != self.expected_count:
            raise ValueError(f"Composition is missing {self.expected_count - len(self._submitted)} image(s)")
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
        fitted = None
        mask = None
        try:
            if self.style.background_fit == "stretch":
                fitted = image.resize(target, Image.Resampling.LANCZOS)
            elif self.style.background_fit == "cover":
                fitted = ImageOps.fit(image, target, method=Image.Resampling.LANCZOS)
            elif self.style.background_fit == "contain":
                fitted = Image.new("RGB", target, self.style.background_color)
                contained = ImageOps.contain(image, target, method=Image.Resampling.LANCZOS)
                try:
                    fitted.paste(contained, ((target[0] - contained.width) // 2, (target[1] - contained.height) // 2))
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
            image.close()
            if fitted is not None:
                fitted.close()
            if mask is not None:
                mask.close()

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
        self._draw_headers()
        for row in range(self.compositor.row_count):
            rect = self.geometry.row_label_rects[row]
            self._draw_vertical_text(
                _prompt_axis_label(row),
                rect,
                self.geometry.row_font_size,
                self.style.text_color,
                bold=True,
            )
            for column in range(self.compositor.column_count):
                self._draw.rectangle(_inclusive(self.geometry.cell(row, column)), fill=self.style.placeholder_color)
                self._draw_image_frame(row, column, self.geometry.cell(row, column))
                label_rect = self.geometry.artist_mixer_label(row, column)
                if label_rect is not None:
                    self._draw.rectangle(_inclusive(label_rect), fill=self.style.panel_color)
        if self.style.show_region_frames:
            self._draw.line(
                (self.geometry.control_separator_x, self.geometry.grid_bounds[1], self.geometry.control_separator_x, self.geometry.grid_bounds[3]),
                fill=self.style.frame_color,
                width=max(1, self.style.frame_width),
            )
        if self.compositor.options.show_stack_details and self.compositor.original_lora_items:
            self._draw_footer()

    def _draw_headers(self) -> None:
        labels = ["BASE"] + list(_stack_mix_labels(self.compositor.stacks))
        for column, label in enumerate(labels):
            first = self.geometry.cell(0, column)
            last = self.geometry.cell(self.compositor.row_count - 1, column)
            rect = (first[0], self.geometry.header_rect[1], last[2], self.geometry.header_rect[3])
            color = self.style.frame_color if column == 0 else self.style.accent_colors[(column - 1) % len(self.style.accent_colors)]
            self._draw_fitted_text(label, rect, self.geometry.header_font_size, color, bold=True, align="center")

    def _draw_footer(self) -> None:
        top = self.geometry.grid_bounds[3] + self.style.footer_gap
        rect = (self.geometry.grid_bounds[0], top, self.geometry.grid_bounds[2], self.geometry.canvas_size[1] - self.style.outer_margin)
        self._draw.rectangle(_inclusive(rect), fill=self.style.panel_color, outline=self.style.frame_color, width=max(1, self.style.frame_width))
        width = max(1, rect[2] - rect[0])
        items = self.compositor.original_lora_items
        column_width = width / max(1, len(items))
        for index, item in enumerate(items):
            left = round(rect[0] + index * column_width)
            right = round(rect[0] + (index + 1) * column_width)
            if index:
                self._draw.line((left, rect[1], left, rect[3]), fill=self.style.frame_color, width=1)
            accent = self.style.accent_colors[index % len(self.style.accent_colors)]
            if item.is_artist_tag:
                title = f"Artist tag: {item.display_name}"
                details = f"Weight: {item.strength:g}"
            else:
                title = f"LoRA: {item.display_name}"
                details = f"Strength: {item.strength:g}"
                if item.trigger_word.strip():
                    details += f" / Trigger: {item.trigger_word.strip()}"
            self._draw_fitted_text(title, (left + 8, rect[1] + 4, right - 8, rect[1] + 28), self.geometry.row_font_size, accent, bold=True)
            self._draw_fitted_text(details, (left + 8, rect[1] + 28, right - 8, rect[3] - 4), max(9, self.geometry.row_font_size - 3), self.style.text_color)

    def _draw_image_frame(self, row: int, column: int, rect: Rect) -> None:
        color = self.style.frame_color if column == 0 else self.style.accent_colors[(column - 1) % len(self.style.accent_colors)]
        width = self.style.frame_width if column == 0 else self.style.cell_frame_width
        self._draw.rectangle(_inclusive(rect), outline=color, width=max(1, width))

    def _draw_artist_mixer_label(self, row: int, column: int) -> None:
        label_rect = self.geometry.artist_mixer_label(row, column)
        if label_rect is None:
            raise RuntimeError(
                "Anima Artist Mixer label was submitted without reserving label geometry"
            )
        self._draw.rectangle(_inclusive(label_rect), fill=self.style.panel_color)
        accent_width = max(3, min(12, self.compositor.options.image_width // 64))
        self._draw.rectangle(
            (label_rect[0], label_rect[1], label_rect[0] + accent_width - 1, label_rect[3] - 1),
            fill=self.style.accent_colors[0],
        )
        padding = max(8, self.style.footer_padding)
        self._draw_fitted_text(
            "Anima Artist Mixer",
            (
                label_rect[0] + accent_width + padding,
                label_rect[1],
                label_rect[2] - padding,
                label_rect[3],
            ),
            self.geometry.row_font_size,
            self.style.text_color,
            bold=True,
        )

    def _draw_vertical_text(
        self,
        text: str,
        rect: Rect,
        font_size: int,
        color: RGBColor,
        *,
        bold: bool = False,
    ) -> None:
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        if width <= 0 or height <= 0:
            return

        max_text_width = max(1, height - 8)
        selected = self.fonts.get(max(7, int(font_size)), bold=bold)
        for size in range(max(7, int(font_size)), 6, -1):
            candidate = self.fonts.get(size, bold=bold)
            if self._draw.textbbox((0, 0), str(text), font=candidate)[2] <= max_text_width:
                selected = candidate
                break
        value = str(text)
        while value and self._draw.textbbox((0, 0), value, font=selected)[2] > max_text_width:
            value = value[:-1]
        if value != str(text) and max_text_width >= self._draw.textlength("...", font=selected):
            value = value.rstrip() + "..."

        box = self._draw.textbbox((0, 0), value, font=selected)
        text_width = max(1, box[2] - box[0])
        text_height = max(1, box[3] - box[1])
        padding = 4
        layer = Image.new("RGBA", (text_width + padding * 2, text_height + padding * 2), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.text((padding - box[0], padding - box[1]), value, fill=(*color, 255), font=selected)
        rotated = layer.rotate(270, expand=True, resample=Image.Resampling.BICUBIC)
        x = rect[0] + max(0, (width - rotated.width) // 2)
        y = rect[1] + max(0, (height - rotated.height) // 2)
        self._canvas.paste(rotated, (x, y), rotated)

    def _draw_fitted_text(self, text: str, rect: Rect, font_size: int, color: RGBColor, *, bold: bool = False, align: str = "left") -> None:
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        if width <= 0 or height <= 0:
            return
        selected = self.fonts.get(max(7, int(font_size)), bold=bold)
        for size in range(max(7, int(font_size)), 6, -1):
            candidate = self.fonts.get(size, bold=bold)
            if self._draw.textbbox((0, 0), str(text), font=candidate)[2] <= width:
                selected = candidate
                break
        value = str(text)
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
        mode = self.compositor.options.image_fit
        if mode == "strict":
            raise ValueError(f"Expected image size {target}, received {converted.size}")
        if mode == "stretch":
            return converted.resize(target, Image.Resampling.LANCZOS)
        if mode == "cover":
            return ImageOps.fit(converted, target, method=Image.Resampling.LANCZOS)
        contained = ImageOps.contain(converted, target, method=Image.Resampling.LANCZOS)
        result = Image.new("RGB", target, self.style.placeholder_color)
        result.paste(contained, ((target[0] - contained.width) // 2, (target[1] - contained.height) // 2))
        return result


__all__ = [
    "DEFAULT_CONTROL_GAP_RATIO",
    "LoraStackMatrixCompositor",
    "LoraStackMatrixSession",
    "StackMatrixGeometry",
    "StackMatrixOptions",
]
