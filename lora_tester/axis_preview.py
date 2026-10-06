from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pprint import pformat
from typing import Any

from .stack import LoraStack
from .xy import PromptEntry, XYAxis


LABELS = {
    "axis": ("轴内容预览", "Axis Content Preview"),
    "title": ("轴标题", "Axis title"),
    "groups": ("分组数", "Groups"),
    "entries": ("总项数", "Entries"),
    "parameters": ("参数", "Parameters"),
    "group": ("分组", "Group"),
    "entry": ("项", "Entry"),
    "label": ("标签", "Label"),
    "detail_label": ("详情标签", "Detail label"),
    "base": ("无参数覆盖（BASE / 沿用采样器基础设置）", "No parameter overrides (BASE / sampler defaults)"),
    "empty": ("（空）", "(empty)"),
    "prompt": ("提示词正文", "Prompt body"),
    "prefix": ("前置文本", "Prefix"),
    "suffix": ("后置文本", "Suffix"),
    "full_prompt": ("合并后的提示词", "Combined prompt"),
    "independent_artist_tags": ("独立画师 Tag", "Independent artist tags"),
    "stack": ("风格组合", "Style stack"),
    "file": ("文件", "File"),
    "trigger": ("触发词", "Trigger words"),
    "artist": ("画师 Tag", "Artist tags"),
    "artist_text": ("画师原始文本", "Original artist text"),
    "weight": ("权重", "Weight"),
    "template": ("画师 Tag 模板", "Artist tag template"),
    "builtin": ("内置（由采样器按模型选择）", "Built-in (selected by the sampler for its model)"),
    "details": ("轴详情", "Axis details"),
    "table": ("表格", "Table"),
    "text": ("文本", "Text"),
    "row": ("行", "Row"),
}


def format_axis_preview(axis: XYAxis, language: str = "zh") -> str:
    if not isinstance(axis, XYAxis):
        raise TypeError("Axis Preview expects an XYAxis from an axis node")
    if language not in {"zh", "en"}:
        raise ValueError("language must be zh or en")
    label_index = 0 if language == "zh" else 1
    lines: list[str] = []

    def label(key: str) -> str:
        return LABELS[key][label_index]

    def field(name: str, value: Any, indent: int) -> None:
        prefix = " " * indent
        if isinstance(value, str):
            parts = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if len(parts) == 1:
                lines.append(f"{prefix}{name}: {parts[0] or label('empty')}")
            else:
                lines.append(f"{prefix}{name}:")
                lines.extend(f"{prefix}  {part}" for part in parts)
        elif isinstance(value, PromptEntry):
            lines.append(f"{prefix}{name}:")
            for key in ("prompt", "prefix", "suffix", "full_prompt", "independent_artist_tags"):
                field(label(key), getattr(value, key), indent + 2)
        elif isinstance(value, LoraStack):
            lines.append(f"{prefix}{name}: {label('stack')} ({len(value.items)})")
            for item_index, item in enumerate(value.items, start=1):
                item_prefix = " " * (indent + 2)
                kind = label("artist") if item.is_artist_tag else "LoRA"
                lines.append(f"{item_prefix}[{item_index:02d}] {kind}")
                if item.is_artist_tag:
                    field(label("artist_text"), item.trigger_word, indent + 4)
                    field(label("artist"), item.artist_tags, indent + 4)
                else:
                    field(label("file"), item.name, indent + 4)
                    field(label("trigger"), item.trigger_word, indent + 4)
                field(label("weight"), item.strength, indent + 4)
            field(label("template"), value.artist_template or label("builtin"), indent + 2)
        elif isinstance(value, Mapping):
            lines.append(f"{prefix}{name}:")
            if not value:
                lines.append(f"{prefix}  {{}}")
            for key, nested in value.items():
                field(str(key), nested, indent + 2)
        elif isinstance(value, (tuple, list)):
            lines.append(f"{prefix}{name}:")
            if not value:
                lines.append(f"{prefix}  []")
            for value_index, nested in enumerate(value, start=1):
                field(f"[{value_index:02d}]", nested, indent + 2)
        elif is_dataclass(value) and not isinstance(value, type):
            lines.append(f"{prefix}{name}: {type(value).__name__}")
            for data_field in fields(value):
                field(data_field.name, getattr(value, data_field.name), indent + 2)
        elif value is None or isinstance(value, (bool, int, float)):
            lines.append(f"{prefix}{name}: {value}")
        else:
            field(name, pformat(value, width=88, sort_dicts=False), indent)

    lines.append(label("axis"))
    lines.append("=" * 48)
    field(label("title"), axis.title, 0)
    field(label("groups"), len(axis.groups), 0)
    field(label("entries"), len(axis.entries), 0)
    field(label("parameters"), ", ".join(sorted(axis.parameter_names)) or label("empty"), 0)
    global_index = 0
    for group_index, group in enumerate(axis.groups, start=1):
        lines.extend(("", "-" * 48, f"{label('group')} {group_index:02d} ({len(group)} {label('entry')})"))
        for local_index, entry in enumerate(group, start=1):
            global_index += 1
            lines.extend(("", f"  [{global_index:02d}] {label('entry')} {local_index:02d}"))
            field(label("label"), entry.label, 4)
            if entry.detail_label:
                field(label("detail_label"), entry.detail_label, 4)
            if not entry.parameters:
                lines.append(f"    {label('base')}")
            for parameter in entry.parameters:
                field(parameter.name, parameter.value, 4)
    if axis.detail_blocks:
        lines.extend(("", "=" * 48, label("details")))
        for block_index, block in enumerate(axis.detail_blocks, start=1):
            lines.extend(("", f"  [{block_index:02d}] {block.title} ({label(block.mode)})"))
            if block.mode == "table":
                for row_index, row in enumerate(block.rows, start=1):
                    lines.append(f"    {label('row')} {row_index:02d}")
                    for header, cell in zip(block.headers, row):
                        field(header, cell, 6)
                if not block.rows:
                    field(label("table"), " | ".join(block.headers), 4)
            else:
                for text in block.text:
                    lines.extend("    " + part for part in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    return "\n".join(lines)


__all__ = ["format_axis_preview"]
