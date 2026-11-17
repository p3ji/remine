"""LangGraph state machine with MCP retrieval, RetryPolicy, Command routing, and trace reducers."""
from __future__ import annotations

import operator
import time
import urllib.error
from dataclasses import replace
from typing import Annotated, Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

from .agents import (
    Model,
    allowed_source_ids,
    critique_story,
    draft_story,
    select_angle,
)
from .mcp_server import MCPToolClient, RemineMCPServer
from .models import (
    BindError,
    BoundStory,
    CandidateBrief,
    EditorialDecision,
    MethodRule,
    RejectionCategory,
    TraceEvent,
    Verdict,
    bind_story,
    check_semantic_comparability,
)
from .tools import RemineTools


class State(TypedDict, total=False):
    brief: CandidateBrief
    rules: tuple[MethodRule, ...]
    memory: tuple[dict, ...]
    coverage_overlaps: tuple[dict, ...]
    mcp_calls: tuple[dict, ...]
    trace: Annotated[tuple[dict, ...], operator.add]
    decision: EditorialDecision
    raw_draft: dict
    bound_story: BoundStory
    attempts: int
    feedback: str
    feedback_log: Annotated[tuple[str, ...], operator.add]
    error: str


def _is_transient_error(exc: BaseException) -> bool:
    """Identify momentary network, timeout, or server errors for LangGraph RetryPolicy."""
    if isinstance(exc, (ConnectionError, TimeoutError, urllib.error.URLError)):
        return True
    try:
        import openai

        if isinstance(
            exc,
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
            ),
        ):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code >= 500 or exc.status_code in (408, 429)
    except ImportError:
        pass
    return False


def _is_model_failure(exc: BaseException) -> bool:
    """Identify malformed model responses or non-transient API errors that escalate fail-closed."""
    if _is_transient_error(exc):
        return False
    # ValueError also covers json.JSONDecodeError, InvalidDecision, and BindError.
    if isinstance(exc, ValueError):
        return True
    try:
        import openai

        if isinstance(exc, openai.APIStatusError):
            return True
    except ImportError:
        pass
    return False


def build_graph(
    tools: RemineTools,
    model: Model,
    max_attempts: int = 2,
    enforce_semantic_standardization: bool = False,
    use_mcp: bool = True,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    retry_policy: RetryPolicy | None = None,
):
    """Compile once; invoke per candidate brief so state never leaks across briefs."""
    release = tools.release
    mcp_server = RemineMCPServer(tools) if use_mcp else None
    rule_index = {r.id: r for r in tools.confirmed_methodology()}
    transient_retry = retry_policy or RetryPolicy(
        max_attempts=3,
        initial_interval=0.25,
        backoff_factor=2.0,
        retry_on=_is_transient_error,
    )

    def _emit_trace(
        state: State,
        node: str,
        status: str,
        attempt: int,
        summary: str,
        started_at: float,
        details: dict[str, Any] | None = None,
    ) -> tuple[dict, ...]:
        event = TraceEvent(
            node=node,
            status=status,
            attempt=attempt,
            summary=summary,
            duration_ms=(time.perf_counter() - started_at) * 1000.0,
            details=details or {},
        ).to_dict()
        if on_event is not None:
            on_event({"brief_id": state["brief"].id, **event})
        # Single-element tuple appended atomically by State.trace operator.add reducer.
        return (event,)

    def gate(state: State) -> Command[Literal["retrieve", "__end__"]]:
        t0 = time.perf_counter()
        brief = state["brief"]
        if not brief.reliable or brief.quality_flag.upper() != "A":
            decision = EditorialDecision(
                brief_id=brief.id,
                verdict=Verdict.UNRELIABLE,
                reason=(
                    f"Blocked by deterministic reliability gate "
                    f"(quality_flag={brief.quality_flag!r}, reliable={brief.reliable})."
                ),
                rejection_category=None,
                citations=(f"R:{release['id']}",),
                attempts=0,
            )
            return Command(
                update={
                    "decision": decision,
                    "trace": _emit_trace(
                        state,
                        "gate",
                        "blocked",
                        0,
                        f"Reliability gate blocked {brief.id} (flag={brief.quality_flag})",
                        t0,
                        {"verdict": decision.verdict.value, "quality_flag": brief.quality_flag},
                    ),
                },
                goto=END,
            )

        if enforce_semantic_standardization:
            is_valid, reason = check_semantic_comparability(brief, tools.methodology)
            if not is_valid:
                rule_citations = tuple(
                    f"M:{r.id}" for r in tools.confirmed_methodology() if r.semantic_rule
                )
                decision = EditorialDecision(
                    brief_id=brief.id,
                    verdict=Verdict.SKIP,
                    reason=f"Blocked by metadata semantic comparability gate: {reason}",
                    rejection_category=RejectionCategory.CONFOUNDED_COMPARISON,
                    citations=(f"R:{release['id']}", *rule_citations),
                    attempts=0,
                )
                return Command(
                    update={
                        "decision": decision,
                        "trace": _emit_trace(
                            state,
                            "gate",
                            "blocked",
                            0,
                            f"Gate 0 semantic check skipped {brief.id}",
                            t0,
                            {"verdict": decision.verdict.value, "reason": reason},
                        ),
                    },
                    goto=END,
                )

        return Command(
            update={
                "attempts": 0,
                "feedback": "",
                "feedback_log": (),
                "trace": _emit_trace(
                    state,
                    "gate",
                    "passed",
                    0,
                    f"Passed Gate 0 ({'strict' if enforce_semantic_standardization else 'advisory'} semantic check)",
                    t0,
                    {"quality_flag": brief.quality_flag, "standardized": brief.standardized},
                ),
            },
            goto="retrieve",
        )

    def retrieve(state: State) -> State:
        t0 = time.perf_counter()
        brief = state["brief"]
        context = f"{brief.title} {brief.summary}"

        if mcp_server is not None:
            client = MCPToolClient(mcp_server)
            rule_rows = client.call_tool(
                "search_methodology_rules",
                {"brief_id": brief.id, "query": context, "limit": 3},
            )
            rules = tuple(rule_index[row["id"]] for row in rule_rows if row["id"] in rule_index)
            memory = tuple(
                client.call_tool("recall_editorial_memory", {"query": context, "limit": 2})
            )
            manifest_report = client.call_tool(
                "inspect_coverage_manifest", {"brief_id": brief.id}
            )
            overlaps = tuple(manifest_report.get("overlaps", ()))
            mcp_calls = tuple(client.call_log)
        else:
            rules = tools.search_methodology(brief, context)
            memory = tools.recall_editorial_memory(context)
            overlaps = tuple(tools.check_coverage_overlaps(brief))
            mcp_calls = ()

        return {
            "rules": rules,
            "memory": memory,
            "coverage_overlaps": overlaps,
            "mcp_calls": mcp_calls,
            "trace": _emit_trace(
                state,
                "retrieve",
                "ok",
                0,
                f"Retrieved {len(rules)} rule(s), {len(memory)} precedent(s), {len(overlaps)} overlap(s)",
                t0,
                {
                    "rule_ids": [r.id for r in rules],
                    "memory_ids": [m.get("id") for m in memory],
                    "overlap_tiers": sorted({o.get("tier", "") for o in overlaps}),
                    "mcp_tool_count": len(mcp_calls),
                },
            ),
        }

    def angle(state: State) -> Command[Literal["draft", "review", "__end__"]]:
        t0 = time.perf_counter()
        brief = state["brief"]
        overlaps = list(state.get("coverage_overlaps", ()))
        try:
            decision = select_angle(
                model,
                release,
                brief,
                state["rules"],
                state["memory"],
                coverage_overlaps=overlaps,
            )
        except Exception as exc:
            if not _is_model_failure(exc):
                raise
            err = f"angle failed: {type(exc).__name__}: {exc}"
            return Command(
                update={
                    "error": err,
                    "trace": _emit_trace(state, "angle", "error", 0, err, t0),
                },
                goto="review",
            )

        trace = _emit_trace(
            state,
            "angle",
            decision.verdict.value,
            0,
            f"Angle Verdict: {decision.verdict.value}"
            + (
                f" ({decision.rejection_category.value})"
                if decision.rejection_category
                else ""
            ),
            t0,
            {
                "verdict": decision.verdict.value,
                "rejection_category": (
                    decision.rejection_category.value
                    if decision.rejection_category
                    else None
                ),
            },
        )
        return Command(
            update={"decision": decision, "trace": trace},
            goto=END if decision.verdict == Verdict.SKIP else "draft",
        )

    def draft(state: State) -> Command[Literal["bind_check", "review"]]:
        t0 = time.perf_counter()
        attempt = state.get("attempts", 0) + 1
        overlaps = list(state.get("coverage_overlaps", ()))
        try:
            raw = draft_story(
                model,
                release,
                state["brief"],
                state["rules"],
                state["memory"],
                attempt=attempt,
                feedback=state.get("feedback", ""),
                previous_draft=state.get("raw_draft"),
                coverage_overlaps=overlaps,
            )
            return Command(
                update={
                    "raw_draft": raw,
                    "attempts": attempt,
                    "trace": _emit_trace(
                        state,
                        "draft",
                        "ok",
                        attempt,
                        f"Draft attempt {attempt} generated ({raw.get('stance', 'draft')})",
                        t0,
                        {"fact_ids": raw.get("fact_ids", [])},
                    ),
                },
                goto="bind_check",
            )
        except Exception as exc:
            if not _is_model_failure(exc):
                raise
            err = f"drafter failed: {type(exc).__name__}: {exc}"
            return Command(
                update={
                    "attempts": attempt,
                    "error": err,
                    "trace": _emit_trace(state, "draft", "error", attempt, err, t0),
                },
                goto="review",
            )

    def bind_check(state: State) -> Command[Literal["critic", "draft", "review"]]:
        t0 = time.perf_counter()
        brief = state["brief"]
        attempt = state.get("attempts", 1)
        allowed = set(allowed_source_ids(release, brief, state["rules"]))
        try:
            bound = bind_story(
                state["raw_draft"], brief, allowed, release["table_url"]
            )
            return Command(
                update={
                    "bound_story": bound,
                    "trace": _emit_trace(
                        state,
                        "bind_check",
                        "passed",
                        attempt,
                        f"Bound {len(bound.token_bindings)} symbolic token(s) with 0 contract violations",
                        t0,
                        {"token_count": len(bound.token_bindings), "fact_ids": list(bound.fact_ids)},
                    ),
                },
                goto="critic",
            )
        except BindError as exc:
            msg = f"bind failed ({type(exc).__name__}): {exc}"
            if attempt < max_attempts:
                return Command(
                    update={
                        "feedback": msg,
                        "feedback_log": (msg,),
                        "trace": _emit_trace(
                            state,
                            "bind_check",
                            "retry",
                            attempt,
                            f"Attempt {attempt} failed binding -> retry ({exc})",
                            t0,
                            {"error": msg},
                        ),
                    },
                    goto="draft",
                )
            return Command(
                update={
                    "feedback_log": (msg,),
                    "error": f"after {attempt} attempt(s), {msg}",
                    "trace": _emit_trace(
                        state,
                        "bind_check",
                        "escalated",
                        attempt,
                        f"Attempt {attempt} failed binding -> human review",
                        t0,
                        {"error": msg},
                    ),
                },
                goto="review",
            )

    def critic(state: State) -> Command[Literal["__end__", "draft", "review"]]:
        t0 = time.perf_counter()
        attempt = state.get("attempts", 1)
        overlaps = list(state.get("coverage_overlaps", ()))
        try:
            approved, note = critique_story(
                model,
                release,
                state["brief"],
                state["rules"],
                state["bound_story"],
                attempt=attempt,
                coverage_overlaps=overlaps,
            )
        except Exception as exc:
            if not _is_model_failure(exc):
                raise
            err = f"critic failed: {type(exc).__name__}: {exc}"
            return Command(
                update={
                    "error": err,
                    "trace": _emit_trace(state, "critic", "error", attempt, err, t0),
                },
                goto="review",
            )

        if approved:
            bound = state["bound_story"]
            merged_citations = tuple(
                dict.fromkeys((*state["decision"].citations, *bound.citations))
            )
            return Command(
                update={
                    "decision": replace(
                        state["decision"],
                        citations=merged_citations,
                        bound_story=bound,
                        critic_note=note,
                        attempts=attempt,
                    ),
                    "trace": _emit_trace(
                        state,
                        "critic",
                        "approved",
                        attempt,
                        f"Critic approved on attempt {attempt}",
                        t0,
                        {"note": note},
                    ),
                },
                goto=END,
            )

        msg = f"critic rejected: {note}"
        if attempt < max_attempts:
            return Command(
                update={
                    "feedback": msg,
                    "feedback_log": (msg,),
                    "trace": _emit_trace(
                        state,
                        "critic",
                        "retry",
                        attempt,
                        f"Critic rejected attempt {attempt} -> retry",
                        t0,
                        {"note": note},
                    ),
                },
                goto="draft",
            )
        return Command(
            update={
                "feedback_log": (msg,),
                "error": f"after {attempt} attempt(s), {msg}",
                "trace": _emit_trace(
                    state,
                    "critic",
                    "escalated",
                    attempt,
                    f"Critic rejected attempt {attempt} -> human review",
                    t0,
                    {"note": note},
                ),
            },
            goto="review",
        )

    def review(state: State) -> State:
        t0 = time.perf_counter()
        brief = state["brief"]
        attempt = state.get("attempts", 1)
        decision = EditorialDecision(
            brief_id=brief.id,
            verdict=Verdict.REVIEW,
            reason="A human editor must review this candidate brief.",
            rejection_category=None,
            citations=(f"R:{release['id']}",),
            critic_note=state.get("error", ""),
            attempts=attempt,
        )
        return {
            "decision": decision,
            "trace": _emit_trace(
                state,
                "review",
                "escalated",
                attempt,
                "Escalated to Human Review Queue",
                t0,
                {"error": state.get("error", "")},
            ),
        }

    graph = StateGraph(State)
    graph.add_node("gate", gate)
    graph.add_node("retrieve", retrieve, retry_policy=transient_retry)
    graph.add_node("angle", angle, retry_policy=transient_retry)
    graph.add_node("draft", draft, retry_policy=transient_retry)
    graph.add_node("bind_check", bind_check)
    graph.add_node("critic", critic, retry_policy=transient_retry)
    graph.add_node("review", review)

    graph.add_edge(START, "gate")
    graph.add_edge("retrieve", "angle")
    graph.add_edge("review", END)
    return graph.compile()


def run_snapshot(
    tools: RemineTools,
    model: Model,
    max_attempts: int = 2,
    enforce_semantic_standardization: bool = False,
    use_mcp: bool = True,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[State, ...]:
    graph = build_graph(
        tools,
        model,
        max_attempts=max_attempts,
        enforce_semantic_standardization=enforce_semantic_standardization,
        use_mcp=use_mcp,
        on_event=on_event,
    )
    return tuple(graph.invoke({"brief": brief}) for brief in tools.list_briefs())
