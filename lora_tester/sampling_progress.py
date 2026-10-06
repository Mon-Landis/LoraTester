from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from typing import Any


@dataclass
class _CellProgress:
    progress: Any
    completed_tasks: int
    total_tasks: int
    fraction: float = 0.0

    def update(self, value: float, total: float, preview: Any) -> None:
        denominator = max(1.0, float(total))
        fraction = max(0.0, min(float(value), denominator)) / (denominator + 1.0)
        self.fraction = max(self.fraction, fraction)
        self.progress.update_absolute(self.completed_tasks + self.fraction, self.total_tasks, preview)


_ACTIVE_CELL: ContextVar[_CellProgress | None] = ContextVar("lora_tester_cell_progress", default=None)
_INSTALL_LOCK = Lock()


def _install_dispatcher(progress_class: type) -> None:
    with _INSTALL_LOCK:
        original = progress_class.update_absolute
        if getattr(original, "_lora_tester_context_dispatcher", False):
            return

        @wraps(original)
        def update_absolute(bar, value, total=None, preview=None):
            cell = _ACTIVE_CELL.get()
            bar_node = getattr(bar, "node_id", None)
            owner_node = getattr(cell.progress, "node_id", None) if cell is not None else None
            if cell is None or bar is cell.progress or (bar_node is not None and bar_node != owner_node):
                return original(bar, value, total, preview)
            if total is not None:
                bar.total = total
            bar.current = min(value, bar.total)
            cell.update(bar.current, bar.total, preview)

        update_absolute._lora_tester_context_dispatcher = True
        progress_class.update_absolute = update_absolute


@contextmanager
def aggregate_cell_progress(progress: Any, completed_tasks: int, total_tasks: int):
    module = sys.modules.get("comfy.utils")
    progress_class = getattr(module, "ProgressBar", None)
    if progress is None or progress_class is None:
        yield
        return
    if total_tasks <= 0 or not 0 <= completed_tasks < total_tasks:
        raise ValueError("Cell progress requires a valid XY task index and total")
    _install_dispatcher(progress_class)
    token = _ACTIVE_CELL.set(_CellProgress(progress, completed_tasks, total_tasks))
    try:
        yield
    finally:
        _ACTIVE_CELL.reset(token)
