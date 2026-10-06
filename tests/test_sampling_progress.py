from __future__ import annotations

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tester.sampling_progress import aggregate_cell_progress


class SamplingProgressTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        events = self.events

        class ProgressBar:
            def __init__(self, total, node_id=None):
                self.total = total
                self.current = 0
                self.node_id = node_id

            def update_absolute(self, value, total=None, preview=None):
                if total is not None:
                    self.total = total
                self.current = min(value, self.total)
                events.append((self.current, self.total, preview, self.node_id))

            def update(self, value):
                self.update_absolute(self.current + value)

        self.progress_class = ProgressBar
        self.overall = ProgressBar(3, node_id="xy")
        self.module_patch = patch.dict(sys.modules, {"comfy.utils": SimpleNamespace(ProgressBar=ProgressBar)})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_per_cell_events_become_monotonic_total_with_previews(self):
        for completed, steps in enumerate((2, 5, 3)):
            with aggregate_cell_progress(self.overall, completed, 3):
                inner = self.progress_class(steps)
                for step in range(1, steps + 1):
                    inner.update_absolute(step, steps, f"preview-{completed}-{step}")
                self.assertEqual(inner.current, steps)
            self.overall.update_absolute(completed + 1, 3)
        values = [value for value, *_ in self.events]
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(total == 3 and node == "xy" for _, total, _, node in self.events))
        self.assertTrue(self.events[0][2].startswith("preview-"))
        self.assertLess(self.events[-2][0], 3)
        self.assertEqual(self.events[-1][:2], (3, 3))

    def test_final_sampling_step_reserves_room_for_decode_and_composition(self):
        with aggregate_cell_progress(self.overall, 2, 3):
            self.progress_class(10).update_absolute(10)
        self.assertGreater(self.events[-1][0], 2)
        self.assertLess(self.events[-1][0], 3)

    def test_nested_progress_bars_do_not_move_overall_progress_backwards(self):
        with aggregate_cell_progress(self.overall, 1, 3):
            self.progress_class(10).update_absolute(9)
            self.progress_class(20).update_absolute(1)
            self.progress_class(2).update(2)
        values = [event[0] for event in self.events]
        self.assertEqual(values, sorted(values))

    def test_failed_cell_resets_context_and_native_progress_is_unchanged(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with aggregate_cell_progress(self.overall, 0, 3):
                self.progress_class(4).update_absolute(1)
                raise RuntimeError("interrupted")
        self.progress_class(8).update_absolute(2)
        self.assertEqual(self.events[-1], (2, 8, None, None))

    def test_other_threads_and_independent_contexts_are_not_redirected(self):
        with aggregate_cell_progress(self.overall, 0, 3):
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(self.progress_class(8).update_absolute, 2).result()
            Context().run(self.progress_class(9).update_absolute, 3)
        self.assertEqual(self.events, [(2, 8, None, None), (3, 9, None, None)])

    def test_explicit_other_node_is_not_redirected(self):
        with aggregate_cell_progress(self.overall, 0, 3):
            self.progress_class(10, node_id="other").update_absolute(4)
        self.assertEqual(self.events, [(4, 10, None, "other")])

    def test_dispatcher_is_idempotent_and_owner_bar_does_not_recurse(self):
        with aggregate_cell_progress(self.overall, 0, 3):
            function = self.progress_class.update_absolute
            self.overall.update_absolute(0, 3)
        with aggregate_cell_progress(self.overall, 1, 3):
            self.assertIs(self.progress_class.update_absolute, function)
            self.overall.update_absolute(1, 3)
        self.assertEqual(self.events, [(0, 3, None, "xy"), (1, 3, None, "xy")])

    def test_missing_progress_runtime_leaves_sampler_call_usable(self):
        with patch.dict(sys.modules, {"comfy.utils": SimpleNamespace()}):
            with aggregate_cell_progress(self.overall, 0, 3):
                self.progress_class(10).update_absolute(4)
        with aggregate_cell_progress(None, 0, 3):
            self.progress_class(11).update_absolute(5)
        self.assertEqual(self.events, [(4, 10, None, None), (5, 11, None, None)])

    def test_invalid_task_indices_are_rejected(self):
        for completed, total in ((0, 0), (-1, 3), (3, 3)):
            with self.subTest(completed=completed, total=total), self.assertRaises(ValueError):
                with aggregate_cell_progress(self.overall, completed, total):
                    pass


if __name__ == "__main__":
    unittest.main()
