"""Transparent evaluation, Gate 0 ablation metrics, and run serialization."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from .models import Verdict, render_bound_segments
from .tools import RemineTools
from .workflow import State

_LLM_TRACE_NODES = frozenset({"angle", "draft", "critic"})


@dataclass(frozen=True)
class Evaluation:
    total: int
    correct_verdicts: int
    correct_rejections: int
    expected_skips: int
    recovered_via_retry: int
    confusion: dict[str, dict[str, int]]

    @property
    def verdict_accuracy(self) -> float:
        return self.correct_verdicts / self.total if self.total else 0.0

    @property
    def rejection_accuracy(self) -> float:
        return (
            self.correct_rejections / self.expected_skips
            if self.expected_skips
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct_verdicts": self.correct_verdicts,
            "correct_rejections": self.correct_rejections,
            "expected_skips": self.expected_skips,
            "recovered_via_retry": self.recovered_via_retry,
            "verdict_accuracy": round(self.verdict_accuracy, 4),
            "rejection_accuracy": round(self.rejection_accuracy, 4),
            "confusion": self.confusion,
        }


def evaluate(
    states: tuple[State, ...], labels: dict[str, dict[str, str | None]]
) -> Evaluation:
    state_brief_ids = {s["brief"].id for s in states}
    if not state_brief_ids.issubset(labels.keys()):
        missing = state_brief_ids - set(labels)
        raise ValueError(f"labels missing candidate briefs: {missing}")
    counts: dict[str, Counter[str]] = {}
    correct = rejections = skips = recovered = 0
    for state in states:
        decision = state["decision"]
        label = labels[decision.brief_id]
        expected = Verdict(label["verdict"])
        counts.setdefault(expected.value, Counter())[decision.verdict.value] += 1
        correct += decision.verdict == expected
        if expected == Verdict.SKIP:
            skips += 1
            rejections += decision.rejection_category == label["rejection_category"]
        if decision.verdict == Verdict.PUBLISH and decision.attempts > 1:
            recovered += 1
    return Evaluation(
        total=len(states),
        correct_verdicts=correct,
        correct_rejections=rejections,
        expected_skips=skips,
        recovered_via_retry=recovered,
        confusion={k: dict(v) for k, v in counts.items()},
    )


def count_llm_calls(states: tuple[State, ...], model: Any = None) -> int:
    """Count total agent LLM invocations via model telemetry or workflow trace."""
    call_log = getattr(model, "call_log", None)
    if call_log:
        return len(call_log)
    return sum(
        1
        for state in states
        for event in state.get("trace", ())
        if event.get("node") in _LLM_TRACE_NODES
    )


def ablation_comparison(
    baseline_states: tuple[State, ...],
    gated_states: tuple[State, ...],
    labels: dict[str, dict[str, str | None]] | None = None,
) -> dict[str, Any]:
    """Compare standard vs. Gate 0 semantic-standardization pipeline runs."""
    base_calls = count_llm_calls(baseline_states)
    gated_calls = count_llm_calls(gated_states)
    base_verdicts = Counter(s["decision"].verdict.value for s in baseline_states)
    gated_verdicts = Counter(s["decision"].verdict.value for s in gated_states)

    report: dict[str, Any] = {
        "baseline_llm_calls": base_calls,
        "gated_llm_calls": gated_calls,
        "llm_calls_saved": base_calls - gated_calls,
        "baseline_verdicts": dict(base_verdicts),
        "gated_verdicts": dict(gated_verdicts),
    }
    if labels:
        base_ids = {s["brief"].id for s in baseline_states}
        gated_ids = {s["brief"].id for s in gated_states}
        if base_ids <= labels.keys() and gated_ids <= labels.keys():
            base_eval = evaluate(baseline_states, labels)
            gated_eval = evaluate(gated_states, labels)
            report["baseline_eval"] = base_eval.to_dict()
            report["gated_eval"] = gated_eval.to_dict()
    return report


def _serialize_brief_facts(state: State) -> list[dict[str, Any]]:
    brief = state["brief"]
    return [
        {
            "id": f.id,
            "probe_type": f.probe_type,
            "human": f.human,
            "cut": dict(f.cut),
            "held_at": dict(f.held_at),
            "vectors": list(f.vectors),
            "periods": list(f.periods),
        }
        for f in brief.facts
    ]


def serialize_run(
    tools: RemineTools,
    states: tuple[State, ...],
    *,
    model_name: str = "replay",
    enforce_semantic_standardization: bool = False,
    labels: dict[str, dict[str, str | None]] | None = None,
    model: Any = None,
) -> dict[str, Any]:
    """Serialize workflow outputs, token bindings, traces, and audit data for UI/JSON."""
    counts: Counter[str] = Counter()
    stories: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    total_mcp_calls = 0
    total_duration_ms = 0.0

    for state in states:
        brief = state["brief"]
        decision = state["decision"]
        verdict = decision.verdict.value
        counts[verdict] += 1

        trace = list(state.get("trace", ()))
        mcp_calls = list(state.get("mcp_calls", ()))
        overlaps = list(state.get("coverage_overlaps", ()))
        total_mcp_calls += len(mcp_calls)
        total_duration_ms += sum(float(ev.get("duration_ms", 0.0)) for ev in trace)

        facts_by_id = {f.id: f for f in brief.facts}
        serialized_facts = _serialize_brief_facts(state)

        row: dict[str, Any] = {
            "brief_id": brief.id,
            "title": brief.title,
            "summary": brief.summary,
            "quality_flag": brief.quality_flag,
            "reliable": brief.reliable,
            "standardized": brief.standardized,
            "is_decomposition": brief.is_decomposition,
            "confound_risk": brief.confound_risk,
            "mentions": dict(brief.mentions),
            "verdict": verdict,
            "rejection_category": (
                decision.rejection_category.value
                if decision.rejection_category
                else None
            ),
            "reason": decision.reason,
            "critic_note": decision.critic_note,
            "attempts": decision.attempts,
            "citations": list(decision.citations),
            "rules": [
                {"id": r.id, "title": r.title, "rule": r.rule, "source": r.source}
                for r in state.get("rules", ())
            ],
            "memory": list(state.get("memory", ())),
            "coverage_overlaps": overlaps,
            "mcp_calls": mcp_calls,
            "trace": trace,
            "feedback_log": list(state.get("feedback_log", ())),
            "error": state.get("error", ""),
            "facts": serialized_facts,
        }
        ledger.append(row)

        if decision.bound_story is not None:
            bs = decision.bound_story
            raw_body = bs.raw_body or bs.body
            raw_headline = bs.raw_headline or bs.headline
            stories.append(
                {
                    "brief_id": brief.id,
                    "title": brief.title,
                    "headline": bs.headline,
                    "raw_headline": raw_headline,
                    "assumption": bs.assumption,
                    "stance": bs.stance,
                    "differs_from_daily": bs.differs_from_daily,
                    "body": bs.body,
                    "raw_body": raw_body,
                    "body_segments": render_bound_segments(raw_body, facts_by_id),
                    "fact_ids": list(bs.fact_ids),
                    "citations": list(bs.citations),
                    "table_url": tools.release.get("table_url", ""),
                    "provenance": list(bs.provenance),
                    "token_bindings": [tb.to_dict() for tb in bs.token_bindings],
                    "facts": serialized_facts,
                    "critic_note": decision.critic_note,
                    "attempts": decision.attempts,
                    "recovered_via_retry": decision.attempts > 1,
                }
            )

        if decision.verdict in (Verdict.REVIEW, Verdict.UNRELIABLE):
            review_queue.append(row)

    eval_summary: dict[str, Any] | None = None
    if labels:
        brief_ids = {s["brief"].id for s in states}
        if brief_ids <= labels.keys():
            eval_summary = evaluate(states, labels).to_dict()

    manifest_dict: dict[str, Any] | None = None
    if tools.coverage_manifest is not None:
        manifest_dict = {
            "release_id": tools.coverage_manifest.release_id,
            "paper_catalogue": tools.coverage_manifest.paper_catalogue,
            "claimed_coordinates": [
                asdict(coord)
                for coord in tools.coverage_manifest.claimed_coordinates
            ],
        }

    return {
        "release": dict(tools.release),
        "config": {
            "model": model_name,
            "enforce_semantic_standardization": enforce_semantic_standardization,
        },
        "summary": {
            "total_briefs": len(states),
            "publish_count": counts.get(Verdict.PUBLISH.value, 0),
            "skip_count": counts.get(Verdict.SKIP.value, 0),
            "review_count": counts.get(Verdict.REVIEW.value, 0),
            "unreliable_count": counts.get(Verdict.UNRELIABLE.value, 0),
            "llm_calls": count_llm_calls(states, model=model),
            "mcp_calls": total_mcp_calls,
            "total_duration_ms": round(total_duration_ms, 2),
            "evaluation": eval_summary,
        },
        "coverage_manifest": manifest_dict,
        "stories": stories,
        "ledger": ledger,
        "review_queue": review_queue,
    }
