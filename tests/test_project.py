"""Behavioral tests for Remine's number contract, methodology critic, and LangGraph retry loop."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remine.agents import OpenAICompatibleModel, ReplayModel  # noqa: E402
from remine.evaluation import evaluate  # noqa: E402
from remine.models import (  # noqa: E402
    BindError,
    InvalidDecision,
    Verdict,
    bind_story,
    validate_angle,
)
from remine.tools import RemineTools  # noqa: E402
from remine.workflow import run_snapshot  # noqa: E402


def fixtures() -> tuple[RemineTools, dict]:
    tools = RemineTools(ROOT / "data/lfs_260807_snapshot.json")
    responses = json.loads(
        (ROOT / "data/replay_responses.json").read_text(encoding="utf-8")
    )
    return tools, responses


class RemineCourseTests(unittest.TestCase):
    def test_replay_matches_hand_labels_and_recovers_on_retry(self) -> None:
        tools, responses = fixtures()
        states = run_snapshot(tools, ReplayModel(responses))
        labels = json.loads(
            (ROOT / "data/teaching_labels.json").read_text(encoding="utf-8")
        )
        result = evaluate(states, labels)
        self.assertEqual(
            (
                result.correct_verdicts,
                result.total,
                result.correct_rejections,
                result.recovered_via_retry,
            ),
            (5, 5, 2, 1),
        )
        # LFS-202 failed attempt 1 ('double') and succeeded on attempt 2
        self.assertEqual(states[1]["decision"].attempts, 2)
        self.assertIn("quantity word(s) ['double']", states[1]["feedback_log"][0])
        # LFS-205 was blocked by deterministic gate without calling any LLM
        self.assertEqual(states[4]["decision"].verdict, Verdict.UNRELIABLE)
        self.assertEqual(states[4]["decision"].attempts, 0)

    def test_unconfirmed_methodology_rule_never_enters_context(self) -> None:
        tools, _ = fixtures()
        self.assertEqual(
            {m.id for m in tools.confirmed_methodology()},
            {"M1", "M2", "M3", "M4"},
        )
        with self.assertRaises(InvalidDecision):
            validate_angle(
                {
                    "verdict": "publish",
                    "reason": "Citing unconfirmed note",
                    "rejection_category": None,
                    "citations": ["R:dq260807a", "M:M5"],
                },
                "LFS-201",
                {"R:dq260807a", "F:F201_1", "M:M2"},
            )

    def test_bare_numerals_hyphenated_quantities_and_superlatives_fail_bind(self) -> None:
        tools, responses = fixtures()
        brief = tools.briefs[0]
        allowed = {"R:dq260807a", "F:F201_1", "F:F201_2", "M:M2", "M:M3"}
        base = dict(responses["drafter:LFS-201"])

        with_digit = {**base, "body": "The rate rose by 33.8 points: {{F201_1.human}}."}
        with self.assertRaisesRegex(BindError, "bare numeral"):
            bind_story(with_digit, brief, allowed, tools.release["table_url"])

        with_hyphen_qty = {
            **base,
            "body": "Roughly one-third of the shift is shown here: {{F201_1.human}}.",
        }
        with self.assertRaisesRegex(BindError, "quantity word"):
            bind_story(with_hyphen_qty, brief, allowed, tools.release["table_url"])

        with_superlative = {
            **base,
            "body": "Women recorded the largest increase: {{F201_1.human}}.",
        }
        with self.assertRaisesRegex(BindError, "superlative"):
            bind_story(with_superlative, brief, allowed, tools.release["table_url"])

    def test_false_not_discussed_claim_caught_by_mention_map(self) -> None:
        tools, responses = fixtures()
        brief = tools.briefs[0]
        allowed = {"R:dq260807a", "F:F201_1", "F:F201_2", "M:M2", "M:M3"}
        bad_novelty = {
            **responses["drafter:LFS-201"],
            "differs_from_daily": "not discussed",
        }
        with self.assertRaisesRegex(BindError, "claims 'not discussed'"):
            bind_story(bad_novelty, brief, allowed, tools.release["table_url"])

    def test_persistent_bind_error_routes_to_needs_editorial_review(self) -> None:
        tools, responses = fixtures()
        responses["drafter:LFS-202:attempt2"] = responses["drafter:LFS-202:attempt1"]
        states = run_snapshot(tools, ReplayModel(responses), max_attempts=2)
        self.assertEqual(states[1]["decision"].verdict, Verdict.REVIEW)
        self.assertEqual(states[1]["decision"].attempts, 2)
        self.assertIn("after 2 attempt(s)", states[1]["decision"].critic_note)

    def test_critic_rejection_triggers_retry_and_routes_to_review_if_unfixed(self) -> None:
        tools, responses = fixtures()
        responses["critic:LFS-201"] = {
            "approved": False,
            "note": "Story mislabels core-aged women as all women (violates M:M2).",
        }
        states = run_snapshot(tools, ReplayModel(responses), max_attempts=2)
        self.assertEqual(states[0]["decision"].verdict, Verdict.REVIEW)
        self.assertEqual(states[0]["decision"].attempts, 2)
        self.assertIn("critic rejected", states[0]["decision"].critic_note)

    def test_pure_metadata_check_identifies_age_confound(self) -> None:
        tools, _ = fixtures()
        from remine.models import check_semantic_comparability
        is_valid, reason = check_semantic_comparability(tools.briefs[2], tools.methodology)
        self.assertFalse(is_valid)
        self.assertIn("Semantic confound metadata check", reason or "")

    def test_metadata_semantic_standardization_gate_blocks_confound_at_gate_zero(self) -> None:
        tools, responses = fixtures()
        from remine.models import RejectionCategory
        states = run_snapshot(
            tools, ReplayModel(responses), enforce_semantic_standardization=True
        )
        lfs_203 = states[2]["decision"]
        self.assertEqual(lfs_203.verdict, Verdict.SKIP)
        self.assertEqual(lfs_203.rejection_category, RejectionCategory.CONFOUNDED_COMPARISON)
        self.assertEqual(lfs_203.attempts, 0)
        self.assertIn("metadata semantic comparability gate", lfs_203.reason)

    def test_local_model_uses_reasoning_content_and_fenced_json(self) -> None:
        model = object.__new__(OpenAICompatibleModel)
        model.model = "test-model"
        message = SimpleNamespace(
            content="",
            reasoning_content='```json\n{"approved": true, "note": "ok"}\n```',
        )
        create = lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )
        model.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        self.assertEqual(
            model.complete("critic", "LFS-201", "system", "user"),
            {"approved": True, "note": "ok"},
        )


    def test_ai_adoption_snapshot_replay_and_retry_recovery(self) -> None:
        tools = RemineTools(ROOT / "data/ai_adoption_snapshot.json")
        responses = json.loads(
            (ROOT / "data/replay_responses.json").read_text(encoding="utf-8")
        )
        labels = json.loads(
            (ROOT / "data/teaching_labels.json").read_text(encoding="utf-8")
        )
        states = run_snapshot(tools, ReplayModel(responses))
        result = evaluate(states, labels)
        self.assertEqual(
            (
                result.correct_verdicts,
                result.total,
                result.correct_rejections,
                result.recovered_via_retry,
            ),
            (8, 8, 3, 1),
        )
        self.assertEqual(states[1]["decision"].attempts, 2)
        self.assertEqual(states[4]["decision"].verdict, Verdict.UNRELIABLE)
        self.assertEqual(states[4]["decision"].attempts, 0)
        # AI-306 is the explanatory decomposition probe busting the firm-size confound
        self.assertEqual(states[5]["decision"].verdict, Verdict.PUBLISH)
        self.assertEqual(states[5]["decision"].bound_story.stance, "challenges")
        self.assertIn("majority of the apparent provincial gap", states[5]["decision"].bound_story.body)
        # AI-307 is the deep frontier probe uncovering the micro-enterprise uncertainty barrier
        self.assertEqual(states[6]["decision"].verdict, Verdict.PUBLISH)
        self.assertEqual(states[6]["decision"].bound_story.stance, "challenges")
        self.assertIn("informational uncertainty", states[6]["decision"].bound_story.body)
        # AI-308 is the analytical paper echo skipped via two-tier coverage subtraction
        from remine.models import RejectionCategory
        self.assertEqual(states[7]["decision"].verdict, Verdict.SKIP)
        self.assertEqual(
            states[7]["decision"].rejection_category,
            RejectionCategory.ALREADY_IN_ANALYTICAL_PAPER,
        )

    def test_ai_adoption_semantic_comparability_gate(self) -> None:
        tools = RemineTools(ROOT / "data/ai_adoption_snapshot.json")
        responses = json.loads(
            (ROOT / "data/replay_responses.json").read_text(encoding="utf-8")
        )
        from remine.models import RejectionCategory
        states = run_snapshot(
            tools, ReplayModel(responses), enforce_semantic_standardization=True
        )
        ai_303 = states[2]["decision"]
        self.assertEqual(ai_303.verdict, Verdict.SKIP)
        self.assertEqual(ai_303.rejection_category, RejectionCategory.CONFOUNDED_COMPARISON)
        self.assertEqual(ai_303.attempts, 0)
        self.assertIn("metadata semantic comparability gate", ai_303.reason)


if __name__ == "__main__":
    unittest.main()
