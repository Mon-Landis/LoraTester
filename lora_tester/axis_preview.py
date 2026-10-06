from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pprint import pformat
from typing import Any

from .stack import LoraStack
from .xy import PromptEntry, XYAxis


LABELS = {
    "axis": ("轴内容预览", "Axis Content Preview"),
    "groups": ("分组数", "Groups"),
    "entries": ("总项数", "Entries"),
    "parameters": ("参数", "Parameters"),
    "group": ("分组", "Group"),
    "entry": ("项", "Entry"),
    "detail_label": ("说明", "Note"),
    "base": ("BASE / 沿用基础设置", "BASE / sampler defaults"),
    "empty": ("（空）", "(empty)"),
    "prompt": ("正文", "Body"),
    "prefix": ("前置", "Prefix"),
    "suffix": ("后置", "Suffix"),
    "independent_artist_tags": ("独立画师", "Independent artists"),
    "stack": ("风格组合", "Style stack"),
    "file": ("文件", "File"),
    "trigger": ("触发词", "Trigger"),
    "artist": ("画师", "Artists"),
    "weight": ("权重", "Weight"),
    "template": ("画师模板", "Artist template"),
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
    active_containers: set[int] = set()

    def label(key: str) -> str:
        return LABELS[key][label_index]

    def text(value: Any) -> str:
        return str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ⏎ ").replace("\t", " ⇥ ")

    def inline(value: Any) -> str:
        if isinstance(value, str):
            return text(value) or label("empty")
        if value is None or isinstance(value, (bool, int, float)):
            return str(value)
        identity = id(value)
        if identity in active_containers:
            return "<cycle>"
        active_containers.add(identity)
        try:
            if isinstance(value, PromptEntry):
                parts = [f"{label('prompt')}: {inline(value.prompt)}"]
                for key in ("prefix", "suffix", "independent_artist_tags"):
                    field_value = getattr(value, key)
                    if field_value:
                        parts.append(f"{label(key)}: {inline(field_value)}")
                return "; ".join(parts)
            if isinstance(value, LoraStack):
                items = []
                for item in value.items:
                    if item.is_artist_tag:
                        parts = [f"{label('artist')}: {inline(item.trigger_word)}"]
                    else:
                        parts = [f"{label('file')}: {inline(item.name)}"]
                        if item.trigger_word:
                            parts.append(f"{label('trigger')}: {inline(item.trigger_word)}")
                    parts.append(f"{label('weight')}: {inline(item.strength)}")
                    items.append(("Artist" if item.is_artist_tag else "LoRA") + "(" + "; ".join(parts) + ")")
                result = label("stack") + "[" + ", ".join(items) + "]"
                if value.artist_template is not None:
                    result += f"; {label('template')}: {inline(value.artist_template)}"
                return result
            if isinstance(value, Mapping):
                return "{" + ", ".join(f"{text(key)}: {inline(nested)}" for key, nested in value.items()) + "}"
            if isinstance(value, (tuple, list)):
                return "[" + ", ".join(inline(nested) for nested in value) + "]"
            if is_dataclass(value) and not isinstance(value, type):
                return type(value).__name__ + "(" + ", ".join(
                    f"{data_field.name}: {inline(getattr(value, data_field.name))}"
                    for data_field in fields(value)
                ) + ")"
            return text(pformat(value, width=120, sort_dicts=False))
        finally:
            active_containers.remove(identity)

    lines = [
        f"{label('axis')}: {text(axis.title)} | {label('groups')}: {len(axis.groups)} | "
        f"{label('entries')}: {len(axis.entries)} | {label('parameters')}: "
        + (", ".join(text(name) for name in sorted(axis.parameter_names)) or label("empty"))
    ]
    global_index = 0
    for group_index, group in enumerate(axis.groups, start=1):
        branch = "└─" if group_index == len(axis.groups) and not axis.detail_blocks else "├─"
        lines.append(f"{branch} {label('group')} {group_index:02d} ({len(group)} {label('entry')})")
        for local_index, entry in enumerate(group, start=1):
            global_index += 1
            branch = "└─" if local_index == len(group) else "├─"
            parts = [f"[{global_index:02d}] {text(entry.label)}"]
            if not entry.parameters:
                parts.append(label("base"))
            else:
                parts.extend(f"{text(parameter.name)}: {inline(parameter.value)}" for parameter in entry.parameters)
            if entry.detail_label and entry.detail_label != entry.label and not any(
                isinstance(parameter.value, (PromptEntry, LoraStack)) for parameter in entry.parameters
            ):
                parts.append(f"{label('detail_label')}: {text(entry.detail_label)}")
            lines.append("\t" + branch + " " + " | ".join(parts))
    if axis.detail_blocks:
        lines.append("└─ " + label("details"))
        for block_index, block in enumerate(axis.detail_blocks, start=1):
            branch = "└─" if block_index == len(axis.detail_blocks) else "├─"
            lines.append(f"\t{branch} {text(block.title)} ({label(block.mode)})")
            rows = [
                f"{label('row')} {row_index:02d} | " + " | ".join(f"{text(header)}: {inline(cell)}" for header, cell in zip(block.headers, row))
                for row_index, row in enumerate(block.rows, start=1)
            ] if block.mode == "table" else [inline(value) for value in block.text]
            if not rows and block.mode == "table":
                rows = [" | ".join(text(header) for header in block.headers)]
            for row_index, row in enumerate(rows, start=1):
                branch = "└─" if row_index == len(rows) else "├─"
                lines.append("\t\t" + branch + " " + row)
    return "\n".join(lines)


__all__ = ["format_axis_preview"]
