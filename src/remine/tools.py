"""Read-only snapshot tools, two-tier coverage subtraction, and snapshot discovery."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import (
    CandidateBrief,
    ClaimCoordinate,
    CoverageManifest,
    Fact,
    MethodRule,
    SemanticRule,
)

_TOKEN = re.compile(r"[\wÀ-ÿ]{3,}", re.UNICODE)
_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "not",
    "that",
    "the",
    "this",
    "was",
    "were",
    "with",
}


def _terms(text: str) -> set[str]:
    return set(_TOKEN.findall(text.casefold())) - _STOPWORDS


def _parse_rule(item: dict[str, Any]) -> MethodRule:
    s_raw = item.get("semantic_rule")
    s_rule = SemanticRule(**s_raw) if s_raw else None
    return MethodRule(
        id=item["id"],
        title=item["title"],
        rule=item["rule"],
        source=item["source"],
        confirmed=bool(item["confirmed"]),
        semantic_rule=s_rule,
    )


def _parse_fact(item: dict[str, Any]) -> Fact:
    return Fact(
        id=item["id"],
        probe_type=item["probe_type"],
        human=item["human"],
        cut=dict(item["cut"]),
        held_at=dict(item["held_at"]),
        vectors=tuple(item["vectors"]),
        periods=tuple(item["periods"]),
        breakdown=item.get("breakdown"),
    )


def _parse_brief(item: dict[str, Any]) -> CandidateBrief:
    return CandidateBrief(
        id=item["id"],
        release_id=item["release_id"],
        title=item["title"],
        summary=item["summary"],
        quality_flag=item["quality_flag"],
        reliable=bool(item["reliable"]),
        facts=tuple(_parse_fact(f) for f in item["facts"]),
        mentions=dict(item["mentions"]),
        methodology_ids=tuple(item["methodology_ids"]),
        standardized=bool(item.get("standardized", True)),
        confound_risk=item.get("confound_risk"),
        is_decomposition=bool(item.get("is_decomposition", False)),
    )


def _parse_coverage_manifest(cov_raw: dict[str, Any] | None) -> CoverageManifest | None:
    if not cov_raw:
        return None
    coords = tuple(
        ClaimCoordinate(
            tier=coord["tier"],
            prose_quote=coord["prose_quote"],
            dimensions=dict(coord["dimensions"]),
            value=coord.get("value"),
            period=coord.get("period"),
        )
        for coord in cov_raw.get("claimed_coordinates", [])
    )
    return CoverageManifest(
        release_id=cov_raw["release_id"],
        paper_catalogue=cov_raw.get("paper_catalogue"),
        claimed_coordinates=coords,
    )


class RemineTools:
    """Read-only boundary over a Statistics Canada release snapshot."""

    def __init__(self, snapshot: Path | dict[str, Any]):
        if isinstance(snapshot, dict):
            data = snapshot
            self.snapshot_path: Path | None = None
        else:
            self.snapshot_path = Path(snapshot)
            data = json.loads(self.snapshot_path.read_text(encoding="utf-8"))

        self.release: dict[str, Any] = data["release"]
        self.methodology: tuple[MethodRule, ...] = tuple(
            _parse_rule(item) for item in data["methodology_rules"]
        )
        self.briefs: tuple[CandidateBrief, ...] = tuple(
            _parse_brief(item) for item in data["candidate_briefs"]
        )
        self.editorial_memory: tuple[dict[str, Any], ...] = tuple(
            data.get("editorial_memory", [])
        )
        self.coverage_manifest: CoverageManifest | None = _parse_coverage_manifest(
            data.get("coverage_manifest")
        )
        self._validate()

    def _validate(self) -> None:
        brief_ids = [b.id for b in self.briefs]
        if len(brief_ids) != len(set(brief_ids)):
            raise ValueError("candidate brief IDs must be unique")
        rule_ids = [m.id for m in self.methodology]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("methodology rule IDs must be unique")
        confirmed_ids = {m.id for m in self.confirmed_methodology()}
        for brief in self.briefs:
            if not set(brief.methodology_ids) <= confirmed_ids:
                raise ValueError(
                    f"{brief.id} links to an unconfirmed or unknown methodology rule"
                )
            fact_ids = [f.id for f in brief.facts]
            if not fact_ids or len(fact_ids) != len(set(fact_ids)):
                raise ValueError(f"{brief.id} must have unique nonempty fact IDs")

    def list_briefs(self) -> tuple[CandidateBrief, ...]:
        """Return all candidate briefs in snapshot order."""
        return self.briefs

    def get_brief(self, brief_id: str) -> CandidateBrief:
        """Look up a single candidate brief by ID."""
        for brief in self.briefs:
            if brief.id == brief_id:
                return brief
        raise KeyError(f"unknown candidate brief: {brief_id}")

    def confirmed_methodology(self) -> tuple[MethodRule, ...]:
        """Return only human-confirmed survey methodology rules."""
        return tuple(rule for rule in self.methodology if rule.confirmed)

    def search_methodology(
        self, brief: CandidateBrief, query: str, limit: int = 3
    ) -> tuple[MethodRule, ...]:
        """Retrieve confirmed methodology rules scoped to this brief's linked dimensions."""
        terms = _terms(query)
        if not terms:
            return ()
        candidates = (
            rule for rule in self.confirmed_methodology() if rule.id in brief.methodology_ids
        )
        ranked = sorted(
            candidates,
            key=lambda rule: (-len(_terms(f"{rule.title} {rule.rule}") & terms), rule.id),
        )
        return tuple(
            rule for rule in ranked if _terms(f"{rule.title} {rule.rule}") & terms
        )[:limit]

    def recall_editorial_memory(
        self, query: str, limit: int = 2
    ) -> tuple[dict[str, Any], ...]:
        """Retrieve similar prior editorial rulings as context only (never proof)."""
        terms = _terms(query)
        if not terms:
            return ()
        ranked = sorted(
            self.editorial_memory,
            key=lambda row: (-len(_terms(row["summary"]) & terms), row["id"]),
        )
        return tuple(row for row in ranked if _terms(row["summary"]) & terms)[:limit]

    def check_coverage_overlaps(self, brief: CandidateBrief) -> list[dict[str, Any]]:
        """Compare brief coordinates against Tier 1 (Daily) and Tier 2 (Analytical Paper) claims."""
        if not self.coverage_manifest or brief.is_decomposition:
            return []

        overlaps: list[dict[str, Any]] = []
        brief_dims: dict[str, set[str]] = {}
        brief_periods: set[str] = set()
        for fact in brief.facts:
            brief_periods.update(fact.periods)
            for key, val in {**fact.cut, **fact.held_at}.items():
                base_dim = key.split("_", 1)[0]
                brief_dims.setdefault(base_dim, set()).add(str(val))

        for coord in self.coverage_manifest.claimed_coordinates:
            matched: dict[str, str] = {}
            for dim_name, dim_val in coord.dimensions.items():
                if dim_val in brief_dims.get(dim_name, set()):
                    matched[dim_name] = dim_val
            if not matched:
                continue
            overlap_ratio = len(matched) / max(len(coord.dimensions), 1)
            period_match = bool(coord.period and coord.period in brief_periods)
            if overlap_ratio >= 0.75 or (overlap_ratio >= 0.65 and period_match and len(matched) >= 3):
                overlaps.append(
                    {
                        "tier": coord.tier,
                        "prose_quote": coord.prose_quote,
                        "matched_dimensions": matched,
                        "overlap_ratio": round(overlap_ratio, 2),
                        "period_match": period_match,
                        "paper_catalogue": self.coverage_manifest.paper_catalogue,
                    }
                )
        overlaps.sort(key=lambda item: (-item["overlap_ratio"], item["tier"]))
        return overlaps


def save_snapshot(snapshot: dict[str, Any], out_path: Path) -> Path:
    """Persist a release snapshot dictionary cleanly as UTF-8 JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path


def list_snapshots(data_dir: Path) -> list[dict[str, Any]]:
    """Discover frozen and live-harvested release snapshots in a data directory."""
    if not data_dir.exists():
        return []
    candidates = sorted(data_dir.glob("*_snapshot.json"))
    harvested_dir = data_dir / "harvested"
    if harvested_dir.exists():
        candidates.extend(sorted(harvested_dir.glob("*.json"), reverse=True))

    entries: list[dict[str, Any]] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            release = payload["release"]
            briefs = payload["candidate_briefs"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        entries.append(
            {
                "id": path.stem,
                "name": path.name,
                "path": str(path),
                "source": "harvested" if path.parent.name == "harvested" else "benchmark",
                "release_id": release.get("id", path.stem),
                "date": release.get("date", ""),
                "title": release.get("title", path.stem),
                "pid": release.get("pid"),
                "brief_count": len(briefs),
            }
        )
    return entries
