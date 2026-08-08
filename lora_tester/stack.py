from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass
from typing import Iterable


def _display_name(value: str) -> str:
    normalized = value.replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    stem, _ = os.path.splitext(filename)
    return stem or filename or value


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

    @property
    def display_name(self) -> str:
        return _display_name(str(self.name))


@dataclass(frozen=True, slots=True)
class LoraStack:
    """An ordered collection of LoRA entries applied together."""

    items: tuple[LoraStackItem, ...]

    def __post_init__(self) -> None:
        normalized = tuple(self.items)
        if not normalized:
            raise ValueError("LoRA stack cannot be empty")
        if any(not isinstance(item, LoraStackItem) for item in normalized):
            raise TypeError("LoRA stack items must be LoraStackItem instances")
        object.__setattr__(self, "items", normalized)

    @classmethod
    def from_values(cls, values: Iterable[tuple[str, str, float]]) -> "LoraStack":
        return cls(tuple(LoraStackItem(name, trigger, strength) for name, trigger, strength in values))

    @property
    def label(self) -> str:
        return " + ".join(item.display_name for item in self.items)

    @property
    def trigger_words(self) -> tuple[str, ...]:
        return tuple(item.trigger_word.strip() for item in self.items if item.trigger_word.strip())

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


def split_lora_stack(stack: LoraStack) -> LoraStackList:
    """Return all non-empty combinations in singles, pairs, ... order."""

    if not isinstance(stack, LoraStack):
        raise TypeError("split_lora_stack expects a LoraStack")
    combinations: list[LoraStack] = []
    for size in range(1, len(stack.items) + 1):
        for indexes in itertools.combinations(range(len(stack.items)), size):
            combinations.append(LoraStack(tuple(stack.items[index] for index in indexes)))
    return LoraStackList(tuple(combinations))


__all__ = ["LoraStack", "LoraStackItem", "LoraStackList", "split_lora_stack"]
