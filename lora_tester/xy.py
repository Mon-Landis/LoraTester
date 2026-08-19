from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .stack import LoraStack, LoraStackItem, LoraStackList


MAX_AXIS_ENTRIES = 64
MAX_SEED = 0xFFFFFFFFFFFFFFFF


def axis_token(index: int) -> str:
    """Return spreadsheet-style identifiers: A, B, ..., Z, AA, AB, ..."""

    value = int(index) + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _display_name(value: str) -> str:
    normalized = str(value).replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    stem, _ = os.path.splitext(filename)
    return stem or filename or str(value)


def _format_number(value: float) -> str:
    return f"{float(value):g}"


@dataclass(frozen=True, slots=True)
class PromptEntry:
    prompt: str
    prefix: str = ""
    suffix: str = ""
    independent_artist_tags: str = ""

    def __post_init__(self) -> None:
        if not str(self.prompt).strip():
            raise ValueError("Prompt entries cannot be empty")

    @property
    def full_prompt(self) -> str:
        return ", ".join(
            part
            for part in (
                str(self.prefix).strip(),
                str(self.prompt).strip(),
                str(self.suffix).strip(),
            )
            if part
        )


@dataclass(frozen=True, slots=True)
class PromptList:
    entries: tuple[PromptEntry, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("Prompt list cannot be empty")
        if len(entries) > MAX_AXIS_ENTRIES:
            raise ValueError(f"Prompt list cannot contain more than {MAX_AXIS_ENTRIES} entries")
        if any(not isinstance(entry, PromptEntry) for entry in entries):
            raise TypeError("Prompt lists must contain PromptEntry values")
        object.__setattr__(self, "entries", entries)

    @classmethod
    def parse(
        cls,
        text: str,
        *,
        separator_mode: str = "blank_lines",
        custom_separator: str = "---",
    ) -> "PromptList":
        source = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
        if not source:
            raise ValueError("Multi-prompt text cannot be empty")
        if separator_mode == "blank_lines":
            values = re.split(r"\n[ \t]*\n+", source)
        elif separator_mode == "lines":
            values = source.split("\n")
        elif separator_mode == "custom":
            separator = str(custom_separator)
            if not separator:
                raise ValueError("Custom prompt separator cannot be empty")
            values = source.split(separator)
        else:
            raise ValueError(
                "separator_mode must be blank_lines, lines, or custom"
            )
        prompts = tuple(PromptEntry(value.strip()) for value in values if value.strip())
        return cls(prompts)

    def append_global(
        self,
        text: str,
        *,
        position: str,
        independent_artist_tags: str = "",
    ) -> "PromptList":
        addition = str(text).strip()
        artist_tags = str(independent_artist_tags).strip()
        if position not in {"before", "after"}:
            raise ValueError("position must be before or after")
        entries = []
        for entry in self.entries:
            prefix = entry.prefix
            suffix = entry.suffix
            if addition:
                if position == "before":
                    prefix = ", ".join(part for part in (prefix.strip(), addition) if part)
                else:
                    suffix = ", ".join(part for part in (suffix.strip(), addition) if part)
            combined_artists = "\n".join(
                part
                for part in (entry.independent_artist_tags.strip(), artist_tags)
                if part
            )
            entries.append(
                PromptEntry(
                    prompt=entry.prompt,
                    prefix=prefix,
                    suffix=suffix,
                    independent_artist_tags=combined_artists,
                )
            )
        return PromptList(tuple(entries))


@dataclass(frozen=True, slots=True)
class SeedList:
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds:
            raise ValueError("Seed list cannot be empty")
        if len(seeds) > MAX_AXIS_ENTRIES:
            raise ValueError(f"Seed list cannot contain more than {MAX_AXIS_ENTRIES} entries")
        if any(seed < 0 or seed > MAX_SEED for seed in seeds):
            raise ValueError(f"Seeds must be between 0 and {MAX_SEED}")
        object.__setattr__(self, "seeds", seeds)

    @classmethod
    def parse(cls, text: str) -> "SeedList":
        values = [value for value in re.split(r"[,，;；\s]+", str(text).strip()) if value]
        if not values:
            raise ValueError("Seed input cannot be empty")
        try:
            return cls(tuple(int(value, 10) for value in values))
        except ValueError as exc:
            raise ValueError("Seed input must contain decimal integers") from exc

    @classmethod
    def random(cls, count: int, source_seed: int) -> "SeedList":
        normalized_count = int(count)
        if not 1 <= normalized_count <= MAX_AXIS_ENTRIES:
            raise ValueError(
                f"Random seed count must be between 1 and {MAX_AXIS_ENTRIES}"
            )
        generator = random.Random(int(source_seed))
        return cls(tuple(generator.randrange(MAX_SEED + 1) for _ in range(normalized_count)))


@dataclass(frozen=True, slots=True)
class AxisParameter:
    name: str
    value: Any

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("Axis parameter name cannot be empty")


@dataclass(frozen=True, slots=True)
class DetailBlock:
    """A categorized footer block rendered as either a table or plain text."""

    title: str
    mode: str
    headers: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    text: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.title).strip():
            raise ValueError("Detail block title cannot be empty")
        if self.mode not in {"table", "text"}:
            raise ValueError("Detail block mode must be table or text")
        headers = tuple(str(value) for value in self.headers)
        rows = tuple(tuple(str(value) for value in row) for row in self.rows)
        text = tuple(str(value) for value in self.text)
        if self.mode == "table":
            if not headers:
                raise ValueError("Table detail blocks require headers")
            if any(len(row) != len(headers) for row in rows):
                raise ValueError("Every detail table row must match the header width")
        elif not text:
            raise ValueError("Text detail blocks require at least one line")
        object.__setattr__(self, "headers", headers)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "text", text)


@dataclass(frozen=True, slots=True)
class AxisEntry:
    label: str
    parameters: tuple[AxisParameter, ...]
    detail_label: str = ""

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise ValueError("Axis entry label cannot be empty")
        parameters = tuple(self.parameters)
        if any(not isinstance(parameter, AxisParameter) for parameter in parameters):
            raise TypeError("Axis entries must contain AxisParameter values")
        names = [parameter.name for parameter in parameters]
        if len(set(names)) != len(names):
            raise ValueError("One axis entry cannot assign the same parameter twice")
        object.__setattr__(self, "parameters", parameters)

    @property
    def parameter_map(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {parameter.name: parameter.value for parameter in self.parameters}
        )


@dataclass(frozen=True, slots=True)
class XYAxis:
    """Orientation-independent, grouped axis data for an XY test."""

    title: str
    groups: tuple[tuple[AxisEntry, ...], ...]
    detail_blocks: tuple[DetailBlock, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.title).strip():
            raise ValueError("Axis title cannot be empty")
        groups = tuple(tuple(group) for group in self.groups)
        if not groups or any(not group for group in groups):
            raise ValueError("Axis groups must be a non-empty two-dimensional sequence")
        if any(not isinstance(entry, AxisEntry) for group in groups for entry in group):
            raise TypeError("Axis groups must contain AxisEntry values")
        if sum(len(group) for group in groups) > MAX_AXIS_ENTRIES:
            raise ValueError(f"An axis cannot contain more than {MAX_AXIS_ENTRIES} entries")
        details = tuple(self.detail_blocks)
        if any(not isinstance(block, DetailBlock) for block in details):
            raise TypeError("Axis detail blocks must contain DetailBlock values")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "detail_blocks", details)

    @property
    def entries(self) -> tuple[AxisEntry, ...]:
        return tuple(entry for group in self.groups for entry in group)

    @property
    def parameter_names(self) -> frozenset[str]:
        """Names assigned by any entry; used to reject cross-axis conflicts early."""

        return frozenset(
            parameter.name
            for entry in self.entries
            for parameter in entry.parameters
        )

    @property
    def data(self) -> tuple[tuple[tuple[AxisParameter, ...], ...], ...]:
        """The explicit three-level queue: groups -> entries -> parameters."""

        return tuple(
            tuple(entry.parameters for entry in group)
            for group in self.groups
        )

    @property
    def group_breaks(self) -> tuple[int, ...]:
        breaks: list[int] = []
        count = 0
        for group in self.groups[:-1]:
            count += len(group)
            breaks.append(count)
        return tuple(breaks)


def build_prompt_axis(prompts: PromptList, *, title: str = "PROMPT") -> XYAxis:
    if not isinstance(prompts, PromptList):
        raise TypeError("prompts must come from a Multi Prompt Input node")
    entries = tuple(
        AxisEntry(
            label=f"P{index:02d}",
            parameters=(AxisParameter("prompt", entry),),
            detail_label=entry.full_prompt,
        )
        for index, entry in enumerate(prompts.entries, start=1)
    )
    detail_lines = tuple(
        f"P{index:02d}  {entry.full_prompt}"
        + (
            f"  |  ARTISTS: {entry.independent_artist_tags.strip()}"
            if entry.independent_artist_tags.strip()
            else ""
        )
        for index, entry in enumerate(prompts.entries, start=1)
    )
    return XYAxis(
        title=title,
        groups=(entries,),
        detail_blocks=(DetailBlock("PROMPTS", "text", text=detail_lines),),
    )


def _source_key(item: LoraStackItem, artist_tag: str | None = None) -> tuple[str, str]:
    if artist_tag is not None:
        return ("artist", str(artist_tag).strip().lstrip("@").casefold())
    return ("lora", os.path.normcase(str(item.name).replace("\\", "/")))


def build_lora_stack_axis(
    stacks: LoraStackList,
    *,
    include_base: bool = True,
    title: str = "STYLE",
) -> XYAxis:
    if not isinstance(stacks, LoraStackList):
        raise TypeError("stacks must come from a LoRA Stack List node")
    if not stacks.stacks and not include_base:
        raise ValueError("LoRA Stack axis requires at least one stack or BASE")

    source_tokens: dict[tuple[str, str], str] = {}
    source_rows: dict[tuple[str, str], tuple[str, str, str]] = {}
    trigger_rows: list[tuple[str, str, str]] = []

    def token_for(item: LoraStackItem, artist_tag: str | None = None) -> str:
        key = _source_key(item, artist_tag)
        token = source_tokens.setdefault(key, axis_token(len(source_tokens)))
        if key not in source_rows:
            if artist_tag is not None:
                source_rows[key] = (token, "ARTIST", str(artist_tag).strip().lstrip("@"))
            else:
                source_rows[key] = (token, "LORA", _display_name(item.name))
        return token

    stack_entries: list[AxisEntry] = []
    configuration_rows: list[tuple[str, str]] = []
    for stack in stacks.stacks:
        components: list[str] = []
        for item in stack.items:
            if item.is_artist_tag:
                for artist_tag in item.artist_tags:
                    token = token_for(item, artist_tag)
                    components.append(f"{token}-{_format_number(item.strength)}")
                continue
            token = token_for(item)
            components.append(f"{token}-{_format_number(item.strength)}")
            if item.trigger_word.strip():
                row = (token, _format_number(item.strength), item.trigger_word.strip())
                if row not in trigger_rows:
                    trigger_rows.append(row)
        label = "+".join(components) or "EMPTY"
        stack_entries.append(
            AxisEntry(
                label=label,
                parameters=(AxisParameter("lora_stack", stack),),
                detail_label=stack.label,
            )
        )
        configuration_rows.append((label, stack.label))

    groups: list[tuple[AxisEntry, ...]] = []
    if include_base:
        groups.append((AxisEntry("BASE", ()),))
    if stack_entries:
        groups.append(tuple(stack_entries))

    detail_blocks: list[DetailBlock] = []
    if source_rows:
        detail_blocks.append(
            DetailBlock(
                "STYLE SOURCES",
                "table",
                headers=("CODE", "TYPE", "SOURCE"),
                rows=tuple(source_rows.values()),
            )
        )
    if configuration_rows:
        detail_blocks.append(
            DetailBlock(
                "STYLE CONFIGURATIONS",
                "table",
                headers=("COLUMN", "COMPONENTS"),
                rows=tuple(configuration_rows),
            )
        )
    if trigger_rows:
        detail_blocks.append(
            DetailBlock(
                "LORA TRIGGERS",
                "table",
                headers=("CODE", "WEIGHT", "TRIGGER"),
                rows=tuple(trigger_rows),
            )
        )
    return XYAxis(title=title, groups=tuple(groups), detail_blocks=tuple(detail_blocks))


def build_seed_axis(seeds: SeedList, *, title: str = "SEED") -> XYAxis:
    if not isinstance(seeds, SeedList):
        raise TypeError("seeds must come from a Seed List node")
    entries = tuple(
        AxisEntry(
            label=str(seed),
            parameters=(AxisParameter("seed", int(seed)),),
            detail_label=str(seed),
        )
        for seed in seeds.seeds
    )
    return XYAxis(
        title=title,
        groups=(entries,),
        detail_blocks=(
            DetailBlock(
                "SEEDS",
                "table",
                headers=("INDEX", "SEED"),
                rows=tuple((f"S{index:02d}", str(seed)) for index, seed in enumerate(seeds.seeds, start=1)),
            ),
        ),
    )


def merge_axis_parameters(*entries: AxisEntry) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for entry in entries:
        if not isinstance(entry, AxisEntry):
            raise TypeError("XY parameters can only be merged from AxisEntry values")
        for parameter in entry.parameters:
            if parameter.name in merged:
                raise ValueError(
                    f"Both axes assign parameter {parameter.name!r}; combine it on one axis instead"
                )
            merged[parameter.name] = parameter.value
    return merged


__all__ = [
    "AxisEntry",
    "AxisParameter",
    "DetailBlock",
    "MAX_AXIS_ENTRIES",
    "MAX_SEED",
    "PromptEntry",
    "PromptList",
    "SeedList",
    "XYAxis",
    "axis_token",
    "build_lora_stack_axis",
    "build_prompt_axis",
    "build_seed_axis",
    "merge_axis_parameters",
]
