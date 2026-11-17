"""Typed domain contracts, Gate 0 semantic checks, and Python token-binding compiler."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

TOKEN = re.compile(r"\{\{([^{}]+)\}\}")
NUMERAL = re.compile(
    r"(?<![\w])(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:st|nd|rd|th)?(?![\w])"
)
_QUANTITY_WORDS = re.compile(
    r"(?<![\w])("
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|"
    r"half|halves|third|thirds|quarter|quarters|double|triple|twice|thrice|dozen"
    r")(?:s)?(?![\w])",
    re.I,
)
_SUPERLATIVES = re.compile(
    r"(?<![\w])("
    r"largest|biggest|greatest|smallest|highest|lowest|most|least|"
    r"fastest|slowest|strongest|weakest|best|worst|"
    r"record|unprecedented|unmatched"
    r")(?![\w])",
    re.I,
)
_CALENDAR_QUARTER = re.compile(
    r"\b(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarters?\b",
    re.I,
)
YEAR = re.compile(r"^(19|20)\d{2}$")
ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
STANCES = {"concretizes", "challenges"}


class Verdict(StrEnum):
    """Terminal or editorial outcomes produced by the Remine state machine."""

    PUBLISH = "publish"
    SKIP = "skip"
    UNRELIABLE = "unreliable_data"
    REVIEW = "needs_editorial_review"


class RejectionCategory(StrEnum):
    """Structured reasons explaining why a candidate brief was skipped."""

    ALREADY_IN_DAILY = "already_in_daily"
    ALREADY_IN_ANALYTICAL_PAPER = "already_in_analytical_paper"
    CONFOUNDED_COMPARISON = "confounded_comparison"
    LOW_RELIABILITY = "low_reliability"
    OTHER = "other"


class DimensionRole(StrEnum):
    """Analytical roles played by dimensions inside an official statistics cube."""

    COMPARATIVE = "comparative"
    COMPOSITIONAL_CONFOUNDER = "compositional_confounder"
    MEASURE = "measure"
    AGGREGATE = "aggregate"


@dataclass(frozen=True)
class ClaimCoordinate:
    """Structured representation of a claim already covered in Tier 1 or Tier 2 prose."""

    tier: str  # "daily" or "analytical_paper"
    prose_quote: str
    dimensions: dict[str, str]
    value: float | None = None
    period: str | None = None


@dataclass(frozen=True)
class CoverageManifest:
    """Two-tier manifest recording figures already reported in The Daily and papers."""

    release_id: str
    paper_catalogue: str | None
    claimed_coordinates: tuple[ClaimCoordinate, ...]


@dataclass(frozen=True)
class SemanticRule:
    """Metadata invariant identifying when a comparative slice leaves a confounder unadjusted."""

    comparative_dim: str
    confounder_dim: str
    required_benchmark: str
    unstandardized_aggregate: str
    explanation: str


@dataclass(frozen=True)
class MethodRule:
    """Official survey methodology rule retrieved via RAG when confirmed=True."""

    id: str
    title: str
    rule: str
    source: str
    confirmed: bool
    semantic_rule: SemanticRule | None = None


@dataclass(frozen=True)
class Fact:
    """Deterministic statistical computation packaged as a bindable symbolic token."""

    id: str
    probe_type: str
    human: str
    cut: dict[str, str]
    held_at: dict[str, str]
    vectors: tuple[str, ...]
    periods: tuple[str, ...]


@dataclass(frozen=True)
class CandidateBrief:
    """Candidate story bundle extracted from a Statistics Canada release cube."""

    id: str
    release_id: str
    title: str
    summary: str
    quality_flag: str
    reliable: bool
    facts: tuple[Fact, ...]
    mentions: dict[str, bool]
    methodology_ids: tuple[str, ...]
    standardized: bool = True
    confound_risk: str | None = None
    is_decomposition: bool = False


@dataclass(frozen=True)
class TokenBinding:
    """Audit record for a single {{fact_id.human}} substitution inside bound prose."""

    token: str
    fact_id: str
    field_name: str
    resolved_text: str
    vectors: tuple[str, ...]
    periods: tuple[str, ...]
    cut: dict[str, str]
    held_at: dict[str, str]
    table_url: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize token binding metadata for JSON APIs and UI inspection."""
        return {
            "token": self.token,
            "fact_id": self.fact_id,
            "field_name": self.field_name,
            "resolved_text": self.resolved_text,
            "vectors": list(self.vectors),
            "periods": list(self.periods),
            "cut": dict(self.cut),
            "held_at": dict(self.held_at),
            "table_url": self.table_url,
        }


@dataclass(frozen=True)
class TraceEvent:
    """Single node transition emitted during LangGraph execution."""

    node: str
    status: str
    attempt: int
    summary: str
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the transition event for dashboard visualization."""
        return {
            "node": self.node,
            "status": self.status,
            "attempt": self.attempt,
            "summary": self.summary,
            "duration_ms": round(self.duration_ms, 2),
            "details": self.details,
        }


@dataclass(frozen=True)
class StoryDraft:
    """Unbound symbolic draft produced by the Token-Bound Drafter agent."""

    headline: str
    assumption: str
    stance: str
    differs_from_daily: str
    body: str
    fact_ids: tuple[str, ...]
    citations: tuple[str, ...]


@dataclass(frozen=True)
class BoundStory:
    """Validated story with symbolic tokens resolved to computed Statistics Canada text."""

    headline: str
    assumption: str
    stance: str
    differs_from_daily: str
    body: str
    fact_ids: tuple[str, ...]
    citations: tuple[str, ...]
    provenance: tuple[dict[str, Any], ...]
    token_bindings: tuple[TokenBinding, ...] = ()
    raw_headline: str = ""
    raw_body: str = ""


@dataclass(frozen=True)
class EditorialDecision:
    """Final editorial ruling and provenance bundle for one candidate brief."""

    brief_id: str
    verdict: Verdict
    reason: str
    rejection_category: RejectionCategory | None
    citations: tuple[str, ...]
    bound_story: BoundStory | None = None
    critic_note: str = ""
    attempts: int = 1


class InvalidDecision(ValueError):
    """Raised when an LLM response fails structural, citation, or angle validation."""


class BindError(ValueError):
    """Raised when a story draft violates the deterministic Python number or novelty contract."""


def check_semantic_comparability(
    brief: CandidateBrief, rules: tuple[MethodRule, ...]
) -> tuple[bool, str | None]:
    """Determine deterministically if a cross-member comparison leaves a confounder unadjusted."""
    if brief.is_decomposition:
        return True, None

    if not brief.standardized:
        msg = brief.confound_risk or "Candidate brief is flagged as unstandardized in cube metadata."
        return False, f"Semantic confound metadata check: {msg}"

    semantic_rules = [
        rule.semantic_rule
        for rule in rules
        if rule.confirmed and rule.semantic_rule and rule.id in brief.methodology_ids
    ]
    for fact in brief.facts:
        cut_keys = set(fact.cut.keys())
        is_cross_member = fact.probe_type == "gap_between_members" or any(
            key.endswith(("_high", "_low")) for key in cut_keys
        )
        if not is_cross_member:
            continue
        cut_dims = cut_keys | {key.split("_")[0] for key in cut_keys}
        for s_rule in semantic_rules:
            if s_rule.comparative_dim not in cut_dims:
                continue
            held_val = fact.held_at.get(s_rule.confounder_dim)
            if held_val and held_val.strip() == s_rule.unstandardized_aggregate:
                return False, (
                    f"Semantic confound metadata check: comparison across {s_rule.comparative_dim!r} "
                    f"leaves {s_rule.confounder_dim!r} at unstandardized aggregate {held_val!r}. "
                    f"{s_rule.explanation}"
                )
    return True, None


def resolve_token(token: str, facts: dict[str, Fact]) -> str:
    """Resolve a single {{fact_id.human}} token against verified pre-computed facts."""
    cleaned = token.strip()
    fact_id, _, field_name = cleaned.partition(".")
    if fact_id not in facts:
        raise BindError(f"unknown fact {fact_id!r} in token {{{{{cleaned}}}}}")
    if field_name != "human":
        raise BindError(
            f"unsupported field {field_name!r} on {fact_id}; use {{{{{fact_id}.human}}}}"
        )
    return facts[fact_id].human


def check_bare_numerals(text: str) -> list[str]:
    """Return any numerals outside {{...}} tokens that are neither 4-digit years nor ordinals."""
    stripped = TOKEN.sub(" ", text or "")
    bad: list[str] = []
    for match in NUMERAL.finditer(stripped):
        raw = match.group(0)
        if YEAR.match(raw) or ORDINAL.match(raw):
            continue
        bad.append(raw)
    return bad


def _no_numerals(text: str, where: str) -> None:
    bare = check_bare_numerals(text)
    if bare:
        raise BindError(
            f"bare numeral(s) {bare} in {where} — every number must be a "
            f"{{{{fact_id.human}}}} token so it comes from computed data"
        )


def _no_quantity_words(text: str, where: str) -> None:
    stripped = _CALENDAR_QUARTER.sub(" ", TOKEN.sub(" ", text or ""))
    found = sorted({match.group(0).lower() for match in _QUANTITY_WORDS.finditer(stripped)})
    if found:
        raise BindError(
            f"quantity word(s) {found} in {where} — a number spelled in words is still a "
            f"number and is not checked against the data; use a {{{{fact_id.human}}}} token "
            f"or rewrite without the comparison"
        )


def _no_superlatives(text: str, where: str) -> None:
    stripped = TOKEN.sub(" ", text or "")
    found = sorted({match.group(0).lower() for match in _SUPERLATIVES.finditer(stripped)})
    if found:
        raise BindError(
            f"superlative(s) {found} in {where} — a superlative is a claim about every "
            f"other series in the table and nothing here verifies it; cite a fact token "
            f"or rewrite without the claim"
        )


def _no_tokens(text: str, where: str) -> None:
    if TOKEN.search(text or ""):
        raise BindError(f"{where} is framing text and cannot carry fact tokens")


def _check_text_contract(text: str, where: str, *, allow_tokens: bool) -> None:
    _no_numerals(text, where)
    _no_quantity_words(text, where)
    _no_superlatives(text, where)
    if not allow_tokens:
        _no_tokens(text, where)


def check_differs_from_daily(
    draft: StoryDraft, mentions: dict[str, bool], facts: dict[str, Fact]
) -> None:
    """Reject empty novelty claims or false 'not discussed' claims contradicted by the mention map."""
    raw_claim = (draft.differs_from_daily or "").strip()
    if not raw_claim:
        raise BindError(
            "differs_from_daily is empty — state what The Daily said about this cut, "
            "or state 'not discussed'"
        )
    if raw_claim.lower() != "not discussed":
        return

    mentioned = {key.split("|", 1)[-1] for key, seen in (mentions or {}).items() if seen}
    for fact_id in draft.fact_ids:
        fact = facts.get(fact_id)
        if not fact:
            continue
        for val in fact.cut.values():
            for member in mentioned:
                if member and member in str(val):
                    raise BindError(
                        f"story claims 'not discussed' but The Daily names {member!r} "
                        f"— explain how this cut differs from The Daily's coverage"
                    )


def _parse_story_draft(raw: dict[str, Any]) -> StoryDraft:
    if not isinstance(raw, dict):
        raise BindError("story draft must be a JSON object")
    required_str = ("headline", "assumption", "stance", "differs_from_daily", "body")
    try:
        for key in required_str:
            if not isinstance(raw[key], str):
                raise TypeError(f"{key} must be a string")
        fact_ids = raw["fact_ids"]
        citations = raw["citations"]
        if not isinstance(fact_ids, (list, tuple)) or not isinstance(citations, (list, tuple)):
            raise TypeError("fact_ids and citations must be sequences")
        draft = StoryDraft(
            headline=raw["headline"].strip(),
            assumption=raw["assumption"].strip(),
            stance=raw["stance"].strip(),
            differs_from_daily=raw["differs_from_daily"].strip(),
            body=raw["body"].strip(),
            fact_ids=tuple(fact_ids),
            citations=tuple(citations),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BindError("missing or malformed story draft fields") from exc

    if not draft.headline or not draft.assumption or not draft.body:
        raise BindError("headline, assumption, and body must be nonempty")
    if draft.stance not in STANCES:
        raise BindError(f"stance must be one of {sorted(STANCES)}, got {draft.stance!r}")
    if not draft.fact_ids or any(not isinstance(fid, str) for fid in draft.fact_ids):
        raise BindError("fact_ids must be a nonempty list of fact IDs")
    return draft


def _validate_draft_sources(
    draft: StoryDraft, facts: dict[str, Fact], allowed_sources: set[str]
) -> None:
    if not set(draft.fact_ids) <= set(facts):
        raise BindError("draft references a fact_id not in this candidate brief")
    if (
        not draft.citations
        or any(not isinstance(cid, str) for cid in draft.citations)
        or len(set(draft.citations)) != len(draft.citations)
        or not set(draft.citations) <= allowed_sources
    ):
        raise BindError("citations must be unique IDs from the supplied sources")
    required_fact_citations = {f"F:{fid}" for fid in draft.fact_ids}
    if not required_fact_citations <= set(draft.citations):
        raise BindError("citations must include every F:<fact_id> used in fact_ids")


def _substitute_tokens(
    text: str,
    field_name: str,
    facts: dict[str, Fact],
    allowed_facts: set[str],
    table_url: str,
    bindings_out: list[TokenBinding],
) -> str:
    def _replace(match: re.Match[str]) -> str:
        tok = match.group(1).strip()
        fact_id = tok.partition(".")[0]
        if fact_id not in allowed_facts:
            raise BindError(
                f"story uses {{{{{tok}}}}} but does not list {fact_id} in fact_ids"
            )
        resolved = resolve_token(tok, facts)
        fact = facts[fact_id]
        bindings_out.append(
            TokenBinding(
                token=f"{{{{{tok}}}}}",
                fact_id=fact_id,
                field_name=field_name,
                resolved_text=resolved,
                vectors=fact.vectors,
                periods=fact.periods,
                cut=dict(fact.cut),
                held_at=dict(fact.held_at),
                table_url=table_url,
            )
        )
        return resolved

    return TOKEN.sub(_replace, text)


def render_bound_segments(raw_text: str, facts: dict[str, Fact]) -> list[dict[str, Any]]:
    """Split symbolic prose into ordered text and bound-token spans for UI provenance views."""
    segments: list[dict[str, Any]] = []
    cursor = 0
    for match in TOKEN.finditer(raw_text or ""):
        if match.start() > cursor:
            segments.append({"kind": "text", "text": raw_text[cursor : match.start()]})
        tok = match.group(1).strip()
        fact_id = tok.partition(".")[0]
        fact = facts.get(fact_id)
        resolved = fact.human if fact else match.group(0)
        segments.append(
            {
                "kind": "token",
                "token": f"{{{{{tok}}}}}",
                "fact_id": fact_id,
                "text": resolved,
                "vectors": list(fact.vectors) if fact else [],
                "periods": list(fact.periods) if fact else [],
            }
        )
        cursor = match.end()
    if cursor < len(raw_text or ""):
        segments.append({"kind": "text", "text": raw_text[cursor:]})
    return segments


def bind_story(
    raw: dict[str, Any], brief: CandidateBrief, allowed_sources: set[str], table_url: str
) -> BoundStory:
    """Enforce the zero-numeral Python contract and substitute {{fact_id.human}} tokens."""
    draft = _parse_story_draft(raw)
    facts = {fact.id: fact for fact in brief.facts}
    _validate_draft_sources(draft, facts, allowed_sources)

    for field_name in ("assumption", "differs_from_daily"):
        _check_text_contract(getattr(draft, field_name), f"story {field_name}", allow_tokens=False)
    for field_name in ("headline", "body"):
        _check_text_contract(getattr(draft, field_name), f"story {field_name}", allow_tokens=True)

    check_differs_from_daily(draft, brief.mentions, facts)

    used_tokens = [match.group(1).strip() for match in TOKEN.finditer(f"{draft.headline}\n{draft.body}")]
    if not used_tokens:
        raise BindError("story body must embed at least one {{fact_id.human}} token")

    allowed_facts = set(draft.fact_ids)
    bindings: list[TokenBinding] = []
    bound_headline = _substitute_tokens(
        draft.headline, "headline", facts, allowed_facts, table_url, bindings
    )
    bound_body = _substitute_tokens(
        draft.body, "body", facts, allowed_facts, table_url, bindings
    )
    provenance = tuple(
        {
            "id": fid,
            "vectors": list(facts[fid].vectors),
            "periods": list(facts[fid].periods),
            "cut": dict(facts[fid].cut),
            "held_at": dict(facts[fid].held_at),
            "table_url": table_url,
        }
        for fid in draft.fact_ids
    )
    return BoundStory(
        headline=bound_headline,
        assumption=draft.assumption,
        stance=draft.stance,
        differs_from_daily=draft.differs_from_daily,
        body=bound_body,
        fact_ids=draft.fact_ids,
        citations=draft.citations,
        provenance=provenance,
        token_bindings=tuple(bindings),
        raw_headline=draft.headline,
        raw_body=draft.body,
    )


def validate_angle(
    raw: dict[str, Any], brief_id: str, allowed_sources: set[str]
) -> EditorialDecision:
    """Validate Angle Selector JSON output before routing to Drafter or terminal SKIP."""
    try:
        verdict = Verdict(raw["verdict"])
        reason = raw["reason"]
        cat_raw = raw.get("rejection_category")
        citations = raw["citations"]
    except (KeyError, ValueError, TypeError) as exc:
        raise InvalidDecision("missing or invalid angle decision fields") from exc

    if verdict not in {Verdict.PUBLISH, Verdict.SKIP}:
        raise InvalidDecision("angle selector may only return publish or skip")
    if not isinstance(reason, str) or not reason.strip():
        raise InvalidDecision("reason must be nonempty")
    if (
        not isinstance(citations, list)
        or not citations
        or any(not isinstance(cid, str) for cid in citations)
        or len(set(citations)) != len(citations)
        or not set(citations) <= allowed_sources
    ):
        raise InvalidDecision("citations must be unique IDs from the supplied sources")
    if not any(cid.startswith(("F:", "R:")) for cid in citations):
        raise InvalidDecision(
            "angle decision must cite at least one fact (F:...) or release (R:...)"
        )

    try:
        category = RejectionCategory(cat_raw) if cat_raw is not None else None
    except ValueError as exc:
        raise InvalidDecision("unknown rejection_category") from exc

    if verdict == Verdict.SKIP and category is None:
        raise InvalidDecision("a skip verdict requires a rejection_category")
    if verdict == Verdict.PUBLISH and category is not None:
        raise InvalidDecision("a publish verdict cannot carry a rejection_category")

    return EditorialDecision(
        brief_id=brief_id,
        verdict=verdict,
        reason=reason.strip(),
        rejection_category=category,
        citations=tuple(citations),
    )
