from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass, replace
from typing import Iterable

from .artist import ARTIST_TAG_MODE, ArtistTagTemplate, parse_artist_tag_entries, split_artist_tags


def _display_name(value: str) -> str:
    normalized = value.replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    stem, _ = os.path.splitext(filename)
    return stem or filename or value


def _normalize_stack_name(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Style stack name must be text or None")
    if any(character in value for character in ("\r", "\n", "\t")):
        raise ValueError("Style stack name must be a single line without tabs")
    return value.strip() or None


@dataclass(frozen=True, slots=True)
class LoraStackItem:
    """One LoRA entry stored in a stack."""

    name: str
    trigger_word: str = ""
    strength: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("LoRA file cannot be empty")
        if not math.isfinite(float(self.strength)):
            raise ValueError("LoRA strength must be finite")
        if self.is_artist_tag and not split_artist_tags(self.trigger_word):
            raise ValueError("Artist tag cannot be empty")

    @property
    def is_artist_tag(self) -> bool:
        return str(self.name).strip() == ARTIST_TAG_MODE

    @property
    def artist_tags(self) -> tuple[str, ...]:
        return split_artist_tags(self.trigger_word) if self.is_artist_tag else ()

    @property
    def display_name(self) -> str:
        if self.is_artist_tag:
            return ", ".join(self.artist_tags)
        return _display_name(str(self.name))


@dataclass(frozen=True, slots=True)
class LoraStack:
    """An ordered collection of LoRA entries applied together."""

    items: tuple[LoraStackItem, ...]
    artist_template: ArtistTagTemplate | None = None
    custom_name: str | None = None

    def __post_init__(self) -> None:
        normalized = tuple(self.items)
        if not normalized:
            raise ValueError("LoRA stack cannot be empty")
        if any(not isinstance(item, LoraStackItem) for item in normalized):
            raise TypeError("LoRA stack items must be LoraStackItem instances")
        if self.artist_template is not None and not isinstance(
            self.artist_template, ArtistTagTemplate
        ):
            raise TypeError("artist_template must come from an Artist Tag Template node")
        object.__setattr__(self, "items", normalized)
        object.__setattr__(self, "custom_name", _normalize_stack_name(self.custom_name))

    @classmethod
    def from_values(cls, values: Iterable[tuple[str, str, float]]) -> "LoraStack":
        return cls(tuple(LoraStackItem(name, trigger, strength) for name, trigger, strength in values))

    @property
    def label(self) -> str:
        return " + ".join(item.display_name for item in self.items)

    @property
    def trigger_words(self) -> tuple[str, ...]:
        return tuple(
            item.trigger_word.strip()
            for item in self.items
            if not item.is_artist_tag and item.trigger_word.strip()
        )

    @property
    def artist_entries(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (tag, float(item.strength))
            for item in self.items
            if item.is_artist_tag
            for tag in item.artist_tags
        )

    def signature(self) -> tuple[tuple[str, str, float], ...]:
        return tuple((item.name, item.trigger_word, float(item.strength)) for item in self.items)


@dataclass(frozen=True, slots=True)
class LoraStackList:
    """An ordered list of complete LoRA stacks."""

    stacks: tuple[LoraStack, ...]

    def __post_init__(self) -> None:
        normalized = tuple(self.stacks)
        if any(not isinstance(stack, LoraStack) for stack in normalized):
            raise TypeError("LoRA stack lists must contain LoraStack instances")
        object.__setattr__(self, "stacks", normalized)

    @classmethod
    def merge(cls, values: Iterable[LoraStack | "LoraStackList"]) -> "LoraStackList":
        merged: list[LoraStack] = []
        for value in values:
            source = value.stacks if isinstance(value, LoraStackList) else (value,)
            for stack in source:
                if not isinstance(stack, LoraStack):
                    raise TypeError("Only LoraStack and LoraStackList values can be merged")
                merged.append(stack)
        return cls(tuple(merged))


def rename_lora_stack(stack: LoraStack, name: str | None) -> LoraStack:
    if not isinstance(stack, LoraStack):
        raise TypeError("rename_lora_stack expects a LoraStack")
    normalized = _normalize_stack_name(name)
    return stack if stack.custom_name == normalized else replace(stack, custom_name=normalized)


def _expand_stack_name(template: str, index: int) -> str:
    result: list[str] = []
    position = 0
    while position < len(template):
        character = template[position]
        if character == "\\" and position + 1 < len(template) and template[position + 1] in "\\{}":
            result.append(template[position + 1])
            position += 2
        elif template.startswith("{i}", position):
            result.append(str(index))
            position += 3
        else:
            result.append(character)
            position += 1
    return "".join(result)


def rename_lora_stack_list(stacks: LoraStackList, index: int, name: str) -> LoraStackList:
    if not isinstance(stacks, LoraStackList):
        raise TypeError("rename_lora_stack_list expects a LoraStackList")
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("Style stack index must be an integer")
    if not stacks.stacks or index >= len(stacks.stacks):
        return stacks
    template = _normalize_stack_name(name) or ""
    updated = tuple(
        rename_lora_stack(stack, _expand_stack_name(template, position))
        if index < 0 or position == index else stack
        for position, stack in enumerate(stacks.stacks)
    )
    return stacks if all(first is second for first, second in zip(updated, stacks.stacks)) else LoraStackList(updated)


def parse_artist_stack(
    text: str, artist_template: ArtistTagTemplate | None = None, custom_name: str | None = None
) -> LoraStack:
    """Parse artist-only prompt text into one weighted stack entry per tag."""

    entries = parse_artist_tag_entries(text)
    if not entries:
        raise ValueError("Artist tag text cannot be empty")
    for tag, _weight in entries:
        if ":" in tag or tag.count("(") != tag.count(")") or not tag.strip("@ "):
            raise ValueError(f"Invalid artist tag or weight syntax: {tag}")
    return LoraStack(
        tuple(LoraStackItem(ARTIST_TAG_MODE, tag, weight) for tag, weight in entries),
        artist_template=artist_template,
        custom_name=custom_name,
    )


def replace_stack_artist(
    stack: LoraStack,
    match_tag: str,
    replacement_name: str,
    replacement_trigger: str,
    strength: float,
    strength_mode: str = "replace",
) -> LoraStack:
    """Replace exact artist matches without changing ordinary LoRA triggers."""

    if not isinstance(stack, LoraStack):
        raise TypeError("replace_stack_artist expects a LoraStack")
    matches = parse_artist_stack(match_tag).items
    if len(matches) != 1:
        raise ValueError("Match exactly one artist tag")
    target = matches[0].trigger_word
    if strength_mode not in {"replace", "multiply"}:
        raise ValueError("strength_mode must be replace or multiply")
    replacement = LoraStackItem(str(replacement_name).strip(), str(replacement_trigger), float(strength))
    items: list[LoraStackItem] = []
    changed = False
    for item in stack.items:
        if not item.is_artist_tag or target not in item.artist_tags:
            items.append(item)
            continue
        changed = True
        effective_strength = replacement.strength if strength_mode == "replace" else item.strength * replacement.strength
        for tag in item.artist_tags:
            if tag == target:
                items.append(LoraStackItem(replacement.name, replacement.trigger_word, effective_strength))
            else:
                items.append(LoraStackItem(ARTIST_TAG_MODE, tag, item.strength))
    return replace(stack, items=tuple(items)) if changed else stack


def split_lora_stack(stack: LoraStack) -> LoraStackList:
    """Return all non-empty combinations in singles, pairs, ... order."""

    if not isinstance(stack, LoraStack):
        raise TypeError("split_lora_stack expects a LoraStack")
    combinations: list[LoraStack] = []
    for size in range(1, len(stack.items) + 1):
        for indexes in itertools.combinations(range(len(stack.items)), size):
            combinations.append(
                stack if size == len(stack.items) else LoraStack(
                    tuple(stack.items[index] for index in indexes),
                    artist_template=stack.artist_template,
                )
            )
    return LoraStackList(tuple(combinations))


FLATTEN_WEIGHT_MODES = ("inherit", "normalize", "dual")


def flatten_lora_stack(
    stack: LoraStack,
    include_original: bool = False,
    weight_mode: str = "inherit",
) -> LoraStackList:
    """Return single-entry stacks with the requested child weight policy."""

    if not isinstance(stack, LoraStack):
        raise TypeError("flatten_lora_stack expects a LoraStack")
    if weight_mode not in FLATTEN_WEIGHT_MODES:
        raise ValueError(
            f"weight_mode must be one of {', '.join(FLATTEN_WEIGHT_MODES)}"
        )
    if len(stack.items) == 1:
        item = stack.items[0]
        if weight_mode == "inherit" or float(item.strength) == 1.0:
            return LoraStackList((stack,))
        normalized = replace(item, strength=1.0)
        if weight_mode == "normalize":
            child = replace(stack, items=(normalized,))
            return LoraStackList((stack, child) if include_original else (child,))
        return LoraStackList((stack, LoraStack((normalized,), artist_template=stack.artist_template)))
    stacks = [stack] if include_original else []
    for item in stack.items:
        inherited = item
        if weight_mode == "normalize":
            children = (LoraStackItem(item.name, item.trigger_word, 1.0),)
        elif weight_mode == "dual" and float(item.strength) != 1.0:
            children = (
                LoraStackItem(item.name, item.trigger_word, 1.0),
                inherited,
            )
        else:
            children = (inherited,)
        stacks.extend(
            LoraStack((child,), artist_template=stack.artist_template)
            for child in children
        )
    return LoraStackList(tuple(stacks))


__all__ = [
    "LoraStack", "LoraStackItem", "LoraStackList", "split_lora_stack",
    "flatten_lora_stack", "FLATTEN_WEIGHT_MODES", "parse_artist_stack", "replace_stack_artist",
    "rename_lora_stack", "rename_lora_stack_list",
]
