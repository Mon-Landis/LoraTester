from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_artist_audit import build_audit, render_markdown  # noqa: E402


class ArtistAuditTests(unittest.TestCase):
    def test_audit_exercises_real_routing_branches(self) -> None:
        audit = build_audit()
        cases = {item["case"]: item for item in audit["routing_cases"]}
        self.assertEqual(cases["anima_lora_only_is_native"]["route"], "native_prompt")
        self.assertIsNone(cases["anima_lora_only_is_native"]["artist_chain"])
        self.assertEqual(
            cases["anima_single_artist_plus_lora_is_native"]["route"],
            "native_prompt",
        )
        self.assertIsNone(
            cases["anima_single_artist_plus_lora_is_native"]["artist_chain"]
        )
        self.assertEqual(cases["anima_prompt_tag_is_base_text"]["route"], "native_prompt")
        self.assertEqual(
            cases["anima_independent_tags_use_mixer"]["artist_chain"],
            "@test_artist\n@independent\n(@second:0.5)",
        )
        self.assertEqual(
            cases["anima_independent_tags_use_mixer"]["mixer_base_prompt"],
            "@prompt_artist, portrait",
        )
        self.assertIn("@trigger_artist", cases["anima_lora_trigger_is_not_extracted"]["mixer_base_prompt"])
        self.assertEqual(
            cases["anima_multi_artist_switch_disabled"]["route"],
            "native_prompt_mixer_disabled",
        )
        self.assertIsNone(
            cases["anima_multi_artist_switch_disabled"]["artist_chain"]
        )

        preflight = {item["case"]: item for item in audit["preflight_cases"]}
        self.assertTrue(
            preflight["anima_independent_one_plus_stack_artist"]["mixer_active"]
        )
        self.assertEqual(
            preflight["anima_independent_one_plus_stack_artist"]["mixer_combinations"],
            ["@first + @independent"],
        )
        self.assertEqual(
            preflight["anima_independent_two_cover_base_and_stacks"]["mixer_combinations"],
            [
                "@independent + (@second:0.5)",
                "@first + @independent + (@second:0.5)",
            ],
        )

    def test_checked_in_audit_is_deterministic(self) -> None:
        audit = build_audit()
        checked_in = json.loads(
            (ROOT / "audit" / "artist_routing_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(checked_in, audit)
        report = render_markdown(audit)
        self.assertIn("## Combination Preflight", report)
        self.assertIn("anima_mixer_present", report)


if __name__ == "__main__":
    unittest.main()
