from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable

from PIL import Image, ImageDraw


RGBColor: TypeAlias = tuple[int, int, int]
ColorInput: TypeAlias = str | int | tuple[int, int, int] | tuple[int, int, int, int]
BackgroundSource: TypeAlias = object


def parse_color(value: ColorInput) -> RGBColor:
    if isinstance(value, int):
        if not 0 <= value <= 0xFFFFFF:
            raise ValueError("Integer colors must be between 0x000000 and 0xFFFFFF")
        return ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if len(text) == 3:
            text = "".join(character * 2 for character in text)
        if len(text) not in {6, 8}:
            raise ValueError(f"Unsupported color value: {value!r}")
        try:
            channels = tuple(int(text[index : index + 2], 16) for index in range(0, 6, 2))
        except ValueError as exc:
            raise ValueError(f"Unsupported color value: {value!r}") from exc
        return channels  # type: ignore[return-value]
    if isinstance(value, tuple) and len(value) in {3, 4}:
        channels = tuple(int(channel) for channel in value[:3])
        if any(channel < 0 or channel > 255 for channel in channels):
            raise ValueError("Color channels must be between 0 and 255")
        return channels  # type: ignore[return-value]
    raise TypeError(f"Unsupported color type: {type(value).__name__}")


def mix_color(first: RGBColor, second: RGBColor, amount: float) -> RGBColor:
    amount = max(0.0, min(1.0, float(amount)))
    return tuple(
        round(first[index] * (1.0 - amount) + second[index] * amount) for index in range(3)
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class StyleConfig:
    mode: str
    background_color: ColorInput
    panel_color: ColorInput
    placeholder_color: ColorInput
    text_color: ColorInput
    muted_text_color: ColorInput
    frame_color: ColorInput
    accent_colors: tuple[ColorInput, ColorInput, ColorInput]
    background_image: BackgroundSource | None = None
    background_fit: str = "cover"
    background_opacity: float = 1.0
    decorator: str = "technical"
    outer_margin: int = 24
    cell_gap: int = 4
    region_gap: int = 36
    caption_height: int | None = None
    footer_gap: int = 18
    footer_padding: int = 14
    frame_width: int = 2
    cell_frame_width: int = 1
    font_path: str | Path | None = None
    font_size: int | None = None
    small_font_size: int | None = None
    show_axis_labels: bool = True
    show_cell_captions: bool = False
    show_special_cell_labels: bool = True
    show_region_frames: bool = True
    show_region_labels: bool = False
    show_coordinates: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"black", "white", "custom"}:
            raise ValueError("Style mode must be 'black', 'white', or 'custom'")
        if self.background_fit not in {"cover", "contain", "stretch", "tile"}:
            raise ValueError("background_fit must be cover, contain, stretch, or tile")
        if not 0.0 <= float(self.background_opacity) <= 1.0:
            raise ValueError("background_opacity must be between 0 and 1")
        if len(self.accent_colors) != 3:
            raise ValueError("accent_colors must contain colors for A, B, and C")
        for attribute in (
            "outer_margin",
            "cell_gap",
            "region_gap",
            "footer_gap",
            "footer_padding",
            "frame_width",
            "cell_frame_width",
        ):
            if int(getattr(self, attribute)) < 0:
                raise ValueError(f"{attribute} cannot be negative")
        for attribute in ("caption_height", "font_size", "small_font_size"):
            value = getattr(self, attribute)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{attribute} must be positive when supplied")

        for attribute in (
            "background_color",
            "panel_color",
            "placeholder_color",
            "text_color",
            "muted_text_color",
            "frame_color",
        ):
            object.__setattr__(self, attribute, parse_color(getattr(self, attribute)))
        object.__setattr__(
            self,
            "accent_colors",
            tuple(parse_color(color) for color in self.accent_colors),
        )

    @classmethod
    def black(cls, **overrides: Any) -> "StyleConfig":
        return cls(
            mode="black",
            background_color="#111416",
            panel_color="#1B2022",
            placeholder_color="#252A2C",
            text_color="#F2F4F3",
            muted_text_color="#9AA3A6",
            frame_color="#687175",
            accent_colors=("#C8F04B", "#31BFC4", "#F0785A"),
            **overrides,
        )

    @classmethod
    def white(cls, **overrides: Any) -> "StyleConfig":
        return cls(
            mode="white",
            background_color="#F3F2EE",
            panel_color="#E4E6E2",
            placeholder_color="#FCFCFA",
            text_color="#171A1B",
            muted_text_color="#606A6D",
            frame_color="#798184",
            accent_colors=("#6E8500", "#087E85", "#B7442F"),
            **overrides,
        )

    @classmethod
    def custom(cls, **values: Any) -> "StyleConfig":
        base = cls.black(decorator="none")
        allowed = {item.name for item in fields(cls)} - {"mode"}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise TypeError(f"Unknown style option(s): {', '.join(unknown)}")
        return replace(base, mode="custom", **values)

    @classmethod
    def from_mode(cls, mode: str, **overrides: Any) -> "StyleConfig":
        normalized = str(mode).strip().lower()
        if normalized == "black":
            return cls.black(**overrides)
        if normalized == "white":
            return cls.white(**overrides)
        if normalized == "custom":
            return cls.custom(**overrides)
        raise ValueError(f"Unknown style mode: {mode!r}")


@dataclass(slots=True)
class DecorationContext:
    canvas: Image.Image
    draw: ImageDraw.ImageDraw
    plan: Any
    geometry: Any
    style: StyleConfig


@runtime_checkable
class StyleDecorator(Protocol):
    def draw_background(self, context: DecorationContext) -> None: ...

    def draw_foreground(self, context: DecorationContext) -> None: ...


class NullDecorator:
    def draw_background(self, context: DecorationContext) -> None:
        return None

    def draw_foreground(self, context: DecorationContext) -> None:
        return None


class TechnicalDecorator:
    """A restrained survey-grid treatment around the comparison matrix."""

    def draw_background(self, context: DecorationContext) -> None:
        style = context.style
        bounds = context.geometry.grid_bounds
        subtle = mix_color(style.background_color, style.frame_color, 0.24)
        step = max(24, min(context.geometry.image_width, context.geometry.image_height) // 4)
        for x in range(style.outer_margin, context.canvas.width - style.outer_margin + 1, step):
            context.draw.line((x, style.outer_margin, x, context.canvas.height - style.outer_margin), fill=subtle)
        for y in range(style.outer_margin, context.canvas.height - style.outer_margin + 1, step):
            context.draw.line((style.outer_margin, y, context.canvas.width - style.outer_margin, y), fill=subtle)

        accent = style.accent_colors[0]
        x0, _, x1, _ = bounds
        marker = max(12, style.region_gap)
        marker_y = max(1, style.outer_margin // 2)
        context.draw.line((x0, marker_y, x0 + marker * 3, marker_y), fill=accent, width=max(1, style.frame_width))
        context.draw.line((x1 - marker, marker_y, x1, marker_y), fill=style.frame_color, width=1)

    def draw_foreground(self, context: DecorationContext) -> None:
        x0, y0, x1, y1 = context.geometry.grid_bounds
        size = max(8, context.style.region_gap // 2)
        color = context.style.frame_color
        width = max(1, context.style.frame_width)
        for x, y, sx, sy in (
            (x0, y0, 1, 1),
            (x1, y0, -1, 1),
            (x0, y1, 1, -1),
            (x1, y1, -1, -1),
        ):
            context.draw.line((x, y, x + sx * size, y), fill=color, width=width)
            context.draw.line((x, y, x, y + sy * size), fill=color, width=width)


_DECORATORS: dict[str, StyleDecorator] = {
    "none": NullDecorator(),
    "technical": TechnicalDecorator(),
}


def register_style_decorator(name: str, decorator: StyleDecorator, *, replace_existing: bool = False) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Decorator name cannot be empty")
    if not isinstance(decorator, StyleDecorator):
        raise TypeError("Decorator must implement draw_background and draw_foreground")
    if key in _DECORATORS and not replace_existing:
        raise ValueError(f"Style decorator is already registered: {key}")
    _DECORATORS[key] = decorator


def get_style_decorator(name: str) -> StyleDecorator:
    key = str(name).strip().lower()
    try:
        return _DECORATORS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_DECORATORS))
        raise ValueError(f"Unknown style decorator {name!r}; available: {available}") from exc


def available_style_decorators() -> tuple[str, ...]:
    return tuple(sorted(_DECORATORS))


__all__ = [
    "BackgroundSource",
    "ColorInput",
    "DecorationContext",
    "RGBColor",
    "StyleConfig",
    "StyleDecorator",
    "available_style_decorators",
    "get_style_decorator",
    "mix_color",
    "parse_color",
    "register_style_decorator",
]
