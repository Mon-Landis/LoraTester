from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence


LEVELS = (0.25, 0.5, 0.75, 1.0)
SLOTS = ("A", "B", "C")
Coordinate = tuple[int, int]


def _format_weight(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        value = 0.0
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _display_name(value: str) -> str:
    normalized = value.replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    stem, _ = os.path.splitext(filename)
    return stem or filename or value


@dataclass(frozen=True, slots=True)
class LoraSpec:
    name: str
    max_weight: float = 1.0
    trigger_word: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("LoRA name cannot be empty")
        if not math.isfinite(float(self.max_weight)):
            raise ValueError("LoRA max_weight must be finite")

    @property
    def display_name(self) -> str:
        return _display_name(str(self.name))


@dataclass(frozen=True, slots=True)
class RenderTask:
    task_id: str
    sequence_index: int
    multipliers: tuple[float, ...]
    weights: tuple[float, ...]
    active_slots: tuple[str, ...]
    prompt_additions: tuple[str, ...]
    caption: str
    kind: str

    def multiplier_for(self, slot: str) -> float:
        index = SLOTS.index(slot)
        return self.multipliers[index] if index < len(self.multipliers) else 0.0

    def weight_for(self, slot: str) -> float:
        index = SLOTS.index(slot)
        return self.weights[index] if index < len(self.weights) else 0.0


@dataclass(frozen=True, slots=True)
class CellSpec:
    coordinate: Coordinate
    task_id: str | None
    region_key: str | None = None

    @property
    def occupied(self) -> bool:
        return self.task_id is not None


@dataclass(frozen=True, slots=True)
class RegionSpec:
    key: str
    label: str
    coordinates: tuple[Coordinate, ...]
    slots: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AxisSpec:
    key: str
    slot: str
    side: str
    positions: tuple[int, ...]
    multipliers: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.slot not in SLOTS:
            raise ValueError(f"Unknown LoRA slot: {self.slot}")
        if self.side not in {"top", "bottom", "left", "right"}:
            raise ValueError(f"Unknown axis side: {self.side}")
        if len(self.positions) != len(self.multipliers) or not self.positions:
            raise ValueError("Axis positions and multipliers must be non-empty and have equal length")


@dataclass(frozen=True, slots=True)
class LayoutPlan:
    loras: tuple[LoraSpec, ...]
    x_positions: tuple[int, ...]
    y_positions: tuple[int, ...]
    tasks: tuple[RenderTask, ...]
    cells: tuple[CellSpec, ...]
    regions: tuple[RegionSpec, ...]
    axes: tuple[AxisSpec, ...]
    major_column_breaks: tuple[int, ...] = ()
    major_row_breaks: tuple[int, ...] = ()
    _task_map: Mapping[str, RenderTask] = field(init=False, repr=False, compare=False)
    _cell_map: Mapping[Coordinate, CellSpec] = field(init=False, repr=False, compare=False)
    _placements: Mapping[str, tuple[Coordinate, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        task_map = {task.task_id: task for task in self.tasks}
        cell_map = {cell.coordinate: cell for cell in self.cells}
        if len(task_map) != len(self.tasks):
            raise ValueError("Layout contains duplicate task IDs")
        if len(cell_map) != len(self.cells):
            raise ValueError("Layout contains duplicate cell coordinates")
        axis_keys = {axis.key for axis in self.axes}
        if len(axis_keys) != len(self.axes):
            raise ValueError("Layout contains duplicate axis keys")
        for axis in self.axes:
            if SLOTS.index(axis.slot) >= len(self.loras):
                raise ValueError(f"Axis references unavailable LoRA slot: {axis.slot}")
            available = self.x_positions if axis.side in {"top", "bottom"} else self.y_positions
            if any(position not in available for position in axis.positions):
                raise ValueError(f"Axis {axis.key} contains positions outside the layout")

        expected = {(x, y) for y in self.y_positions for x in self.x_positions}
        if set(cell_map) != expected:
            raise ValueError("Layout cells do not cover the declared coordinate grid")

        placements: dict[str, list[Coordinate]] = {task_id: [] for task_id in task_map}
        for cell in self.cells:
            if cell.task_id is None:
                continue
            if cell.task_id not in task_map:
                raise ValueError(f"Cell references unknown task: {cell.task_id}")
            placements[cell.task_id].append(cell.coordinate)
        missing = [task_id for task_id, coords in placements.items() if not coords]
        if missing:
            raise ValueError(f"Tasks without a placement: {', '.join(missing)}")

        object.__setattr__(self, "_task_map", MappingProxyType(task_map))
        object.__setattr__(self, "_cell_map", MappingProxyType(cell_map))
        object.__setattr__(
            self,
            "_placements",
            MappingProxyType({key: tuple(value) for key, value in placements.items()}),
        )

    @property
    def lora_count(self) -> int:
        return len(self.loras)

    @property
    def unique_task_count(self) -> int:
        return len(self.tasks)

    @property
    def occupied_cell_count(self) -> int:
        return sum(cell.occupied for cell in self.cells)

    @property
    def blank_cell_count(self) -> int:
        return len(self.cells) - self.occupied_cell_count

    @property
    def duplicate_placement_count(self) -> int:
        return self.occupied_cell_count - self.unique_task_count

    def task(self, task_id: str) -> RenderTask:
        try:
            return self._task_map[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown render task: {task_id}") from exc

    def cell_at(self, coordinate: Coordinate) -> CellSpec:
        try:
            return self._cell_map[coordinate]
        except KeyError as exc:
            raise KeyError(f"Coordinate is outside this layout: {coordinate}") from exc

    def placements_for(self, task_id: str) -> tuple[Coordinate, ...]:
        try:
            return self._placements[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown render task: {task_id}") from exc


def build_layout(loras: Sequence[LoraSpec]) -> LayoutPlan:
    normalized = tuple(loras)
    if not 1 <= len(normalized) <= 3:
        raise ValueError("The compositor requires between one and three LoRAs")
    if len(normalized) == 1:
        return _build_one_lora_layout(normalized)
    if len(normalized) == 2:
        return _build_two_lora_layout(normalized)
    return _build_three_lora_layout(normalized)


def _make_task(
    loras: tuple[LoraSpec, ...], multipliers: tuple[float, ...], sequence_index: int
) -> RenderTask:
    weights = tuple(lora.max_weight * multiplier for lora, multiplier in zip(loras, multipliers))
    active_indexes = tuple(index for index, value in enumerate(multipliers) if value > 0.0)
    active_slots = tuple(SLOTS[index] for index in active_indexes)
    prompt_additions = tuple(
        loras[index].trigger_word.strip()
        for index in active_indexes
        if loras[index].trigger_word.strip()
    )
    if not active_indexes:
        task_id = "base"
        caption = "BASE"
        kind = "base"
    else:
        task_id = "+".join(
            f"{SLOTS[index]}{int(round(multipliers[index] * 100)):03d}"
            for index in active_indexes
        )
        caption = " + ".join(
            f"{SLOTS[index]} {_format_weight(weights[index])}" for index in active_indexes
        )
        kind = {1: "single", 2: "pair", 3: "triple"}[len(active_indexes)]
    return RenderTask(
        task_id=task_id,
        sequence_index=sequence_index,
        multipliers=multipliers,
        weights=weights,
        active_slots=active_slots,
        prompt_additions=prompt_additions,
        caption=caption,
        kind=kind,
    )


def _build_tasks(loras: tuple[LoraSpec, ...]) -> tuple[RenderTask, ...]:
    count = len(loras)
    multiplier_sets: list[tuple[float, ...]] = [tuple(0.0 for _ in range(count))]

    for index in range(count):
        for level in LEVELS:
            values = [0.0] * count
            values[index] = level
            multiplier_sets.append(tuple(values))

    for first, second in ((0, 1), (0, 2), (1, 2)):
        if second >= count:
            continue
        for first_level in LEVELS:
            for second_level in LEVELS:
                values = [0.0] * count
                values[first] = first_level
                values[second] = second_level
                multiplier_sets.append(tuple(values))

    if count == 3:
        multiplier_sets.extend(
            (
                (0.5, 0.5, 0.5),
                (1.0, 0.5, 0.5),
                (0.5, 1.0, 0.5),
                (0.5, 0.5, 1.0),
                (1.0, 1.0, 0.5),
                (1.0, 0.5, 1.0),
                (0.5, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            )
        )

    tasks = tuple(_make_task(loras, values, index) for index, values in enumerate(multiplier_sets))
    if len({task.multipliers for task in tasks}) != len(tasks):
        raise AssertionError("Internal error: duplicate multiplier set")
    return tasks


def _task_lookup(tasks: Iterable[RenderTask]) -> dict[tuple[float, ...], str]:
    return {task.multipliers: task.task_id for task in tasks}


def _cell(
    coordinate: Coordinate,
    values: tuple[float, ...] | None,
    lookup: Mapping[tuple[float, ...], str],
    region: str | None,
) -> CellSpec:
    return CellSpec(
        coordinate=coordinate,
        task_id=None if values is None else lookup[values],
        region_key=region,
    )


def _region(
    key: str, label: str, coordinates: Iterable[Coordinate], slots: tuple[str, ...]
) -> RegionSpec:
    return RegionSpec(key=key, label=label, coordinates=tuple(coordinates), slots=slots)


def _axis(
    key: str,
    slot: str,
    side: str,
    positions: Iterable[int],
    multipliers: Iterable[float],
) -> AxisSpec:
    return AxisSpec(
        key=key,
        slot=slot,
        side=side,
        positions=tuple(positions),
        multipliers=tuple(multipliers),
    )


def _build_one_lora_layout(loras: tuple[LoraSpec, ...]) -> LayoutPlan:
    tasks = _build_tasks(loras)
    lookup = _task_lookup(tasks)
    x_positions = (0, 1, 2, 3, 4)
    y_positions = (0,)
    cells = [_cell((0, 0), (0.0,), lookup, "BASE")]
    cells.extend(
        _cell((index, 0), (level,), lookup, "A")
        for index, level in enumerate(LEVELS, start=1)
    )
    return LayoutPlan(
        loras=loras,
        x_positions=x_positions,
        y_positions=y_positions,
        tasks=tasks,
        cells=tuple(cells),
        regions=(
            _region("A", "A", ((x, 0) for x in range(1, 5)), ("A",)),
        ),
        axes=(
            _axis("A_TOP", "A", "top", range(1, 5), LEVELS),
        ),
    )


def _build_two_lora_layout(loras: tuple[LoraSpec, ...]) -> LayoutPlan:
    tasks = _build_tasks(loras)
    lookup = _task_lookup(tasks)
    x_positions = (-4, -3, -2, -1, 0)
    y_positions = (4, 3, 2, 1, 0)
    cells: list[CellSpec] = []
    for y in y_positions:
        for x in x_positions:
            if x == 0 and y == 0:
                cells.append(_cell((x, y), (0.0, 0.0), lookup, "BASE"))
            elif y == 0:
                cells.append(_cell((x, y), (-x / 4.0, 0.0), lookup, "A"))
            elif x == 0:
                cells.append(_cell((x, y), (0.0, y / 4.0), lookup, "B"))
            else:
                cells.append(_cell((x, y), (-x / 4.0, y / 4.0), lookup, "AB"))

    return LayoutPlan(
        loras=loras,
        x_positions=x_positions,
        y_positions=y_positions,
        tasks=tasks,
        cells=tuple(cells),
        regions=(
            _region("AB", "AB", ((x, y) for y in range(1, 5) for x in range(-4, 0)), ("A", "B")),
            _region("A", "A", ((x, 0) for x in range(-4, 0)), ("A",)),
            _region("B", "B", ((0, y) for y in range(1, 5)), ("B",)),
        ),
        axes=(
            _axis("A_TOP", "A", "top", range(-4, 0), reversed(LEVELS)),
            _axis("B_LEFT", "B", "left", range(4, 0, -1), reversed(LEVELS)),
        ),
        major_column_breaks=(3,),
        major_row_breaks=(3,),
    )


def _build_three_lora_layout(loras: tuple[LoraSpec, ...]) -> LayoutPlan:
    tasks = _build_tasks(loras)
    lookup = _task_lookup(tasks)
    x_positions = tuple(range(-4, 5))
    y_positions = tuple(range(4, -5, -1))
    triple_cells: dict[Coordinate, tuple[float, float, float]] = {
        (-3, -1): (1.0, 0.5, 0.5),
        (-1, -1): (0.5, 0.5, 0.5),
        (-4, -2): (1.0, 1.0, 0.5),
        (-2, -2): (0.5, 1.0, 0.5),
        (-3, -3): (1.0, 0.5, 1.0),
        (-1, -3): (0.5, 0.5, 1.0),
        (-4, -4): (1.0, 1.0, 1.0),
        (-2, -4): (0.5, 1.0, 1.0),
    }

    cells: list[CellSpec] = []
    for y in y_positions:
        for x in x_positions:
            coordinate = (x, y)
            if coordinate == (0, 0):
                cells.append(_cell(coordinate, (0.0, 0.0, 0.0), lookup, "BASE"))
            elif x < 0 and y > 0:
                cells.append(_cell(coordinate, (y / 4.0, -x / 4.0, 0.0), lookup, "AB"))
            elif x > 0 and y > 0:
                cells.append(_cell(coordinate, (y / 4.0, 0.0, x / 4.0), lookup, "AC"))
            elif x > 0 and y < 0:
                cells.append(_cell(coordinate, (0.0, -y / 4.0, x / 4.0), lookup, "BC"))
            elif x == 0 and y > 0:
                cells.append(_cell(coordinate, (y / 4.0, 0.0, 0.0), lookup, "A"))
            elif x > 0 and y == 0:
                cells.append(_cell(coordinate, (0.0, 0.0, x / 4.0), lookup, "C"))
            elif (x < 0 and y == 0) or (x == 0 and y < 0):
                level = (-x if x < 0 else -y) / 4.0
                region = "B_X" if x < 0 else "B_Y"
                cells.append(_cell(coordinate, (0.0, level, 0.0), lookup, region))
            elif x < 0 and y < 0:
                cells.append(_cell(coordinate, triple_cells.get(coordinate), lookup, "ABC"))
            else:
                raise AssertionError(f"Unhandled coordinate: {coordinate}")

    return LayoutPlan(
        loras=loras,
        x_positions=x_positions,
        y_positions=y_positions,
        tasks=tasks,
        cells=tuple(cells),
        regions=(
            _region("AB", "AB", ((x, y) for y in range(1, 5) for x in range(-4, 0)), ("A", "B")),
            _region("AC", "AC", ((x, y) for y in range(1, 5) for x in range(1, 5)), ("A", "C")),
            _region("BC", "BC", ((x, y) for y in range(-4, 0) for x in range(1, 5)), ("B", "C")),
            _region("ABC", "ABC", ((x, y) for y in range(-4, 0) for x in range(-4, 0)), ("A", "B", "C")),
            _region("A", "A", ((0, y) for y in range(1, 5)), ("A",)),
            _region("B_X", "B", ((x, 0) for x in range(-4, 0)), ("B",)),
            _region("B_Y", "B", ((0, y) for y in range(-4, 0)), ("B",)),
            _region("C", "C", ((x, 0) for x in range(1, 5)), ("C",)),
        ),
        axes=(
            _axis("B_TOP", "B", "top", range(-4, 0), reversed(LEVELS)),
            _axis("C_TOP", "C", "top", range(1, 5), LEVELS),
            _axis("A_LEFT", "A", "left", range(4, 0, -1), reversed(LEVELS)),
            _axis("A_RIGHT", "A", "right", range(4, 0, -1), reversed(LEVELS)),
            _axis("B_RIGHT", "B", "right", range(-1, -5, -1), LEVELS),
            _axis("C_BOTTOM", "C", "bottom", range(1, 5), LEVELS),
        ),
        major_column_breaks=(3, 4),
        major_row_breaks=(3, 4),
    )


__all__ = [
    "AxisSpec",
    "CellSpec",
    "Coordinate",
    "LayoutPlan",
    "LEVELS",
    "LoraSpec",
    "RegionSpec",
    "RenderTask",
    "SLOTS",
    "build_layout",
]
