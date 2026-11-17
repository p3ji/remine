"""Deterministic statistical probes, unit humanization, and mention extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable

import pandas as pd

_SCALARS = {
    "units": 1.0,
    "": 1.0,
    "tens": 10.0,
    "hundreds": 100.0,
    "thousands": 1_000.0,
    "tens of thousands": 10_000.0,
    "hundreds of thousands": 100_000.0,
    "millions": 1_000_000.0,
    "billions": 1_000_000_000.0,
}

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

_UNRELIABLE_STATUS = {"F", "x", "E", "..", "..."}
_FULL_HISTORY_PROBES = {"long_run_compare"}
_CROSS_MEMBER_PROBES = {"gap_between_members", "gap_trend", "rank_order"}
_SHARE_PROBES = {"share_of_total"}
_RATE_ONLY_PROBES = {"long_run_compare"}
_CHANGE_PROBES = {"gap_trend", "long_run_compare", "share_of_total", "period_change"}


@dataclass
class ProbeFact:
    """Intermediate pre-computed statistical finding produced by a deterministic probe."""

    probe: str
    cut: dict[str, str]
    values: list[float]
    periods: list[str]
    vectors: list[str]
    uom: str = ""
    scalar: str = ""
    decimals: int = 1
    magnitude: float = 0.0
    se: float | None = None
    legibility: float = 0.5
    mentioned: bool = False
    human: str = ""
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


PROBES: dict[str, Callable[[pd.DataFrame, str, dict[str, Any]], list[ProbeFact]]] = {}


def _probe(name: str):
    def decorator(fn):
        PROBES[name] = fn
        return fn

    return decorator


def _scale(value: float, scalar_factor: str) -> float:
    return value * _SCALARS.get((scalar_factor or "").strip().lower(), 1.0)


def _round_people(count: float) -> int:
    magnitude = abs(count)
    step = 1_000 if magnitude >= 10_000 else 100 if magnitude >= 1_000 else 10
    return int(round(magnitude / step) * step)


def humanize_period(period: str) -> str:
    """Convert machine reference dates ('2026-10', '2026-10-20') into readable English."""
    text = str(period).strip()
    if len(text) == 7 and text[4] == "-" and text[:4].isdigit() and text[5:].isdigit():
        month = int(text[5:])
        if 1 <= month <= 12:
            return f"{_MONTHS[month - 1]} {text[:4]}"
    if (
        len(text) == 10
        and text[4] == "-"
        and text[7] == "-"
        and text[:4].isdigit()
        and text[5:7].isdigit()
        and text[8:].isdigit()
    ):
        month = int(text[5:7])
        day = int(text[8:])
        if 1 <= month <= 12:
            return f"{_MONTHS[month - 1]} {day}, {text[:4]}"
    return text


def humanize_value(value: float, uom: str, scalar: str, decimals: int = 1) -> str:
    """Render a level statistic in reader-friendly units."""
    u = (uom or "").strip().lower()
    if "person" in u:
        return f"about {_round_people(_scale(value, scalar)):,} people"
    if "percent" in u:
        return f"{value:.{decimals}f}%"
    if "dollar" in u:
        return f"${_scale(value, scalar):,.0f}"
    scaled = _scale(value, scalar)
    if abs(scaled) >= 1_000 and decimals == 0:
        return f"{scaled:,.0f}"
    unit_suffix = f" {uom.strip()}" if uom.strip() else ""
    return f"{scaled:,.{decimals}f}{unit_suffix}"


def humanize_delta(
    delta: float, uom: str, scalar: str, decimals: int = 1, *, include_direction: bool = True
) -> str:
    """Render a difference or change statistic cleanly with optional direction word."""
    u = (uom or "").strip().lower()
    if "person" in u:
        base = f"about {_round_people(_scale(delta, scalar)):,} people"
        if not include_direction:
            return base
        return base.replace(" people", " more people" if delta >= 0 else " fewer people")
    if "percent" in u:
        base = f"{abs(delta):.{decimals}f} percentage points"
        if not include_direction:
            return base
        word = "higher" if delta >= 0 else "lower"
        return f"{base} {word}"
    if "dollar" in u:
        base = f"${abs(_scale(delta, scalar)):,.0f}"
        if not include_direction:
            return base
        return f"{base} {'more' if delta >= 0 else 'less'}"
    scaled = abs(_scale(delta, scalar))
    unit_suffix = f" {uom.strip()}" if uom.strip() else ""
    base = f"{scaled:,.{decimals}f}{unit_suffix}"
    if not include_direction:
        return base
    return f"{base} {'higher' if delta >= 0 else 'lower'}"


def _is_mentioned(needle: str, prose_lower: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(needle.lower())}(?![\w-])", prose_lower) is not None


def extract_mentions(
    prose: str,
    dimensions: list[tuple[str, tuple[str, ...]]],
    synonyms: dict[str, str] | None = None,
) -> dict[str, bool]:
    """Identify which cube dimension members appear explicitly in The Daily article prose."""
    prose_lower = (prose or "").lower()
    hits: dict[str, bool] = {}
    for dim_name, members in dimensions:
        for member in members:
            hits[f"{dim_name}|{member}"] = _is_mentioned(member, prose_lower)
    for phrase, canonical_member in (synonyms or {}).items():
        if not _is_mentioned(phrase, prose_lower):
            continue
        for key in list(hits):
            if key.split("|", 1)[1] == canonical_member:
                hits[key] = True
    return hits


def prepare_cube_frame(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Filter unreliable StatCan cells out of active probing while returning sample suppressed rows."""
    frame = df.copy()
    if "GEO" in frame.columns and "Geography" not in frame.columns:
        frame = frame.rename(columns={"GEO": "Geography"})

    suppressed_samples: list[dict[str, Any]] = []
    if "STATUS" in frame.columns:
        status_clean = frame["STATUS"].fillna("").astype(str).str.strip()
        bad_mask = status_clean.isin(_UNRELIABLE_STATUS)
        if bad_mask.any():
            for _, row in frame[bad_mask].head(5).iterrows():
                suppressed_samples.append(row.to_dict())
            frame = frame[~bad_mask]

    frame["VALUE"] = pd.to_numeric(frame["VALUE"], errors="coerce")
    frame = frame[frame["VALUE"].notna()]
    if "DECIMALS" in frame.columns:
        frame["DECIMALS"] = pd.to_numeric(frame["DECIMALS"], errors="coerce").fillna(1).astype(int)
    else:
        frame["DECIMALS"] = 1

    for dim, member in (cfg.get("filters") or {}).items():
        if dim in frame.columns:
            frame = frame[frame[dim] == member]

    vdim = cfg.get("value_dimension")
    if not vdim or vdim not in frame.columns:
        return frame.assign(SE=pd.NA, SE_MOM=pd.NA, SE_YOY=pd.NA), suppressed_samples

    se_members = cfg.get("se_members") or {}
    meta_cols = {
        vdim,
        "VALUE",
        "VECTOR",
        "COORDINATE",
        "STATUS",
        "SYMBOL",
        "TERMINATED",
        "DECIMALS",
        "DGUID",
        "UOM",
        "UOM_ID",
        "SCALAR_FACTOR",
        "SCALAR_ID",
    }
    keys = [col for col in frame.columns if col not in meta_cols]
    est = frame[frame[vdim] == cfg["value_member"]].copy()
    for col, cfg_key in (("SE", "level"), ("SE_MOM", "month_over_month"), ("SE_YOY", "year_over_year")):
        se_member = se_members.get(cfg_key)
        if se_member and se_member in set(frame[vdim]):
            se_df = frame[frame[vdim] == se_member][keys + ["VALUE"]].rename(columns={"VALUE": col})
            est = est.merge(se_df, on=keys, how="left")
        else:
            est[col] = pd.NA
    return est, suppressed_samples


def _legibility(dim: str, cfg: dict[str, Any]) -> float:
    return float((cfg.get("legibility") or {}).get(dim, 0.5))


def _display_name(member: Any, cfg: dict[str, Any]) -> str:
    return str((cfg.get("display_names") or {}).get(str(member), member))


def _possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _change_se(rows: pd.DataFrame, periods_apart: int) -> float | None:
    last = rows.iloc[-1]
    if periods_apart == 1 and pd.notna(last.get("SE_MOM")):
        return float(last["SE_MOM"])
    if periods_apart == 12 and pd.notna(last.get("SE_YOY")):
        return float(last["SE_YOY"])
    first = rows.iloc[0]
    if pd.notna(first.get("SE")) and pd.notna(last.get("SE")):
        return float((float(first["SE"]) ** 2 + float(last["SE"]) ** 2) ** 0.5)
    return None


@_probe("gap_between_members")
def probe_gap_between_members(
    df: pd.DataFrame, dim: str, cfg: dict[str, Any]
) -> list[ProbeFact]:
    """Compute cross-member gaps in each period, propagating standard errors when present."""
    aggregates = set((cfg.get("aggregate_members") or {}).get(dim, []))
    facts: list[ProbeFact] = []
    for period, chunk in df.groupby("REF_DATE"):
        rows = {
            str(row[dim]): row
            for _, row in chunk.iterrows()
            if str(row[dim]) not in aggregates
        }
        for name_a, name_b in combinations(sorted(rows), 2):
            row_a, row_b = rows[name_a], rows[name_b]
            hi, lo = (row_a, row_b) if row_a["VALUE"] >= row_b["VALUE"] else (row_b, row_a)
            delta = float(hi["VALUE"] - lo["VALUE"])
            if delta == 0:
                continue
            se = None
            if pd.notna(hi.get("SE")) and pd.notna(lo.get("SE")):
                se = float((float(hi["SE"]) ** 2 + float(lo["SE"]) ** 2) ** 0.5)
            facts.append(
                ProbeFact(
                    probe="gap_between_members",
                    cut={f"{dim}_high": str(hi[dim]), f"{dim}_low": str(lo[dim])},
                    values=[delta, float(hi["VALUE"]), float(lo["VALUE"])],
                    periods=[str(period)],
                    vectors=[str(hi["VECTOR"]), str(lo["VECTOR"])],
                    uom=str(hi.get("UOM", "")),
                    scalar=str(hi.get("SCALAR_FACTOR", "")),
                    decimals=int(hi.get("DECIMALS", 1) or 1),
                    magnitude=abs(delta),
                    se=se,
                    legibility=_legibility(dim, cfg),
                    meta={"high": str(hi[dim]), "low": str(lo[dim]), "dimension": dim},
                )
            )
    return facts


@_probe("gap_trend")
def probe_gap_trend(df: pd.DataFrame, dim: str, cfg: dict[str, Any]) -> list[ProbeFact]:
    """Measure widening or narrowing gaps between unordered pairs over the active window."""
    gaps = probe_gap_between_members(df, dim, cfg)
    by_pair: dict[tuple[str, str], list[ProbeFact]] = {}
    for gap in gaps:
        pair = tuple(sorted((gap.meta["high"], gap.meta["low"])))
        by_pair.setdefault(pair, []).append(gap)

    facts: list[ProbeFact] = []
    for series in by_pair.values():
        series.sort(key=lambda f: f.periods[-1])
        if len(series) < 2:
            continue
        change = float(series[-1].values[0] - series[0].values[0])
        if change == 0:
            continue
        hi, lo = series[-1].meta["high"], series[-1].meta["low"]
        se = None
        if series[0].se is not None and series[-1].se is not None:
            se = float((series[0].se ** 2 + series[-1].se ** 2) ** 0.5)
        facts.append(
            ProbeFact(
                probe="gap_trend",
                cut={f"{dim}_high": hi, f"{dim}_low": lo},
                values=[change],
                periods=[series[0].periods[-1], series[-1].periods[-1]],
                vectors=series[-1].vectors,
                uom=series[-1].uom,
                scalar=series[-1].scalar,
                decimals=series[-1].decimals,
                magnitude=abs(change),
                se=se,
                legibility=_legibility(dim, cfg),
                meta={
                    "direction": "widening" if change > 0 else "narrowing",
                    "high": hi,
                    "low": lo,
                    "periods_observed": len(series),
                    "gap_start": float(series[0].values[0]),
                    "gap_end": float(series[-1].values[0]),
                    "hi_start": float(series[0].values[1]),
                    "lo_start": float(series[0].values[2]),
                    "hi_end": float(series[-1].values[1]),
                    "lo_end": float(series[-1].values[2]),
                },
            )
        )
    return facts


@_probe("long_run_compare")
def probe_long_run_compare(
    df: pd.DataFrame, dim: str, cfg: dict[str, Any]
) -> list[ProbeFact]:
    """Compare the latest period against the earliest historical observation for each member."""
    aggregates = set((cfg.get("aggregate_members") or {}).get(dim, []))
    facts: list[ProbeFact] = []
    for member, chunk in df.groupby(dim):
        if str(member) in aggregates:
            continue
        rows = chunk.sort_values("REF_DATE")
        if len(rows) < 6:
            continue
        first_val, last_val = float(rows["VALUE"].iloc[0]), float(rows["VALUE"].iloc[-1])
        change = last_val - first_val
        if change == 0:
            continue
        last_row = rows.iloc[-1]
        facts.append(
            ProbeFact(
                probe="long_run_compare",
                cut={dim: str(member)},
                values=[change, first_val, last_val],
                periods=[str(rows["REF_DATE"].iloc[0]), str(rows["REF_DATE"].iloc[-1])],
                vectors=[str(v) for v in rows["VECTOR"].unique()],
                uom=str(last_row.get("UOM", "")),
                scalar=str(last_row.get("SCALAR_FACTOR", "")),
                decimals=int(last_row.get("DECIMALS", 1) or 1),
                magnitude=abs(change),
                se=_change_se(rows, len(rows) - 1),
                legibility=_legibility(dim, cfg),
                meta={"periods_observed": len(rows)},
            )
        )
    return facts


@_probe("period_change")
def probe_period_change(
    df: pd.DataFrame, dim: str, cfg: dict[str, Any]
) -> list[ProbeFact]:
    """Measure latest single-period change for each non-aggregate member."""
    aggregates = set((cfg.get("aggregate_members") or {}).get(dim, []))
    facts: list[ProbeFact] = []
    for member, chunk in df.groupby(dim):
        if str(member) in aggregates:
            continue
        rows = chunk.sort_values("REF_DATE")
        if len(rows) < 2:
            continue
        prev_val, cur_val = float(rows["VALUE"].iloc[-2]), float(rows["VALUE"].iloc[-1])
        change = cur_val - prev_val
        if change == 0:
            continue
        last_row = rows.iloc[-1]
        facts.append(
            ProbeFact(
                probe="period_change",
                cut={dim: str(member)},
                values=[change, prev_val, cur_val],
                periods=[str(rows["REF_DATE"].iloc[-2]), str(rows["REF_DATE"].iloc[-1])],
                vectors=[str(last_row["VECTOR"])],
                uom=str(last_row.get("UOM", "")),
                scalar=str(last_row.get("SCALAR_FACTOR", "")),
                decimals=int(last_row.get("DECIMALS", 1) or 1),
                magnitude=abs(change),
                se=_change_se(rows.iloc[-2:], 1),
                legibility=_legibility(dim, cfg),
                meta={"periods_observed": len(rows)},
            )
        )
    return facts


@_probe("share_of_total")
def probe_share_of_total(
    df: pd.DataFrame, dim: str, cfg: dict[str, Any]
) -> list[ProbeFact]:
    """Compute change in each member's share of total across the window."""
    aggregates = set((cfg.get("aggregate_members") or {}).get(dim, []))
    periods = sorted(df["REF_DATE"].unique())
    if len(periods) < 2:
        return []

    def _shares(period: str) -> dict[str, tuple[float, pd.Series]] | None:
        chunk = df[(df["REF_DATE"] == period) & (~df[dim].isin(aggregates))]
        total = float(chunk["VALUE"].sum())
        if total <= 0 or len(chunk) < 2:
            return None
        return {
            str(row[dim]): (float(row["VALUE"]) / total * 100.0, row)
            for _, row in chunk.iterrows()
        }

    first, last = periods[0], periods[-1]
    then, now = _shares(first), _shares(last)
    if not then or not now:
        return []

    facts: list[ProbeFact] = []
    for member in sorted(set(then) & set(now)):
        share_then, _ = then[member]
        share_now, row = now[member]
        change = share_now - share_then
        if round(change, 1) == 0.0:
            continue
        facts.append(
            ProbeFact(
                probe="share_of_total",
                cut={dim: member},
                values=[change, share_now, share_then],
                periods=[str(first), str(last)],
                vectors=[str(row["VECTOR"])],
                uom=str(row.get("UOM", "")),
                scalar=str(row.get("SCALAR_FACTOR", "")),
                decimals=int(row.get("DECIMALS", 1) or 1),
                magnitude=abs(change),
                legibility=_legibility(dim, cfg),
                meta={
                    "share_now": share_now,
                    "share_then": share_then,
                    "direction": "rose" if change > 0 else "fell",
                    "periods_observed": len(periods),
                },
            )
        )
    return facts


@_probe("rank_order")
def probe_rank_order(df: pd.DataFrame, dim: str, cfg: dict[str, Any]) -> list[ProbeFact]:
    """Identify leader and trailer among comparable members in the latest period."""
    aggregates = set((cfg.get("aggregate_members") or {}).get(dim, []))
    periods = sorted(df["REF_DATE"].unique())
    if not periods:
        return []
    latest = periods[-1]
    chunk = df[(df["REF_DATE"] == latest) & (~df[dim].isin(aggregates))].sort_values(
        "VALUE", ascending=False
    )
    if len(chunk) < 2:
        return []
    top, bottom = chunk.iloc[0], chunk.iloc[-1]
    se = None
    if pd.notna(top.get("SE")) and pd.notna(bottom.get("SE")):
        se = float((float(top["SE"]) ** 2 + float(bottom["SE"]) ** 2) ** 0.5)
    return [
        ProbeFact(
            probe="rank_order",
            cut={f"{dim}_leader": str(top[dim]), f"{dim}_trailer": str(bottom[dim])},
            values=[float(top["VALUE"]), float(bottom["VALUE"])],
            periods=[str(latest)],
            vectors=[str(top["VECTOR"]), str(bottom["VECTOR"])],
            uom=str(top.get("UOM", "")),
            scalar=str(top.get("SCALAR_FACTOR", "")),
            decimals=int(top.get("DECIMALS", 1) or 1),
            magnitude=float(top["VALUE"] - bottom["VALUE"]),
            se=se,
            legibility=_legibility(dim, cfg),
            meta={"leader": str(top[dim]), "trailer": str(bottom[dim]), "periods_observed": len(periods)},
        )
    ]


def _describe_fact(
    fact: ProbeFact, measure: str | None, mdim: str | None, cfg: dict[str, Any]
) -> None:
    m_label = _display_name(measure, cfg).strip().lower() if measure else ""
    of_m = f" {m_label}" if m_label else ""
    aggregates = {
        name for names in (cfg.get("aggregate_members") or {}).values() for name in names
    }
    narrowed = [
        _display_name(val, cfg)
        for val in (fact.meta.get("held_at") or {}).values()
        if val not in aggregates
    ]
    among = f", among {' and '.join(narrowed)}" if narrowed else ""
    v, d = fact.values, fact.decimals

    subject = ""
    for key, val in fact.cut.items():
        if key != mdim:
            subject = _display_name(val, cfg)
            break

    if fact.probe == "gap_between_members":
        hi, lo = _display_name(fact.meta["high"], cfg), _display_name(fact.meta["low"], cfg)
        fact.human = (
            f"in {humanize_period(fact.periods[-1])}, the{of_m} gap between {hi} and {lo} "
            f"was {humanize_delta(v[0], fact.uom, fact.scalar, d, include_direction=False)} "
            f"({humanize_value(v[1], fact.uom, fact.scalar, d)} against "
            f"{humanize_value(v[2], fact.uom, fact.scalar, d)}){among}"
        )
    elif fact.probe == "gap_trend":
        hi, lo = _display_name(fact.meta["high"], cfg), _display_name(fact.meta["low"], cfg)
        verb = "widened" if fact.meta["direction"] == "widening" else "narrowed"
        fact.human = (
            f"between {humanize_period(fact.periods[0])} and {humanize_period(fact.periods[-1])}, "
            f"the{of_m} gap between {hi} and {lo} {verb} by "
            f"{humanize_delta(abs(v[0]), fact.uom, fact.scalar, d, include_direction=False)}{among}"
        )
        if "hi_start" in fact.meta and "hi_end" in fact.meta:
            p0, p1 = humanize_period(fact.periods[0]), humanize_period(fact.periods[-1])
            fact.meta["breakdown"] = {
                "columns": ["Series", p0, p1, "Change"],
                "rows": [
                    [
                        hi,
                        humanize_value(fact.meta["hi_start"], fact.uom, fact.scalar, d),
                        humanize_value(fact.meta["hi_end"], fact.uom, fact.scalar, d),
                        humanize_delta(fact.meta["hi_end"] - fact.meta["hi_start"], fact.uom, fact.scalar, d),
                    ],
                    [
                        lo,
                        humanize_value(fact.meta["lo_start"], fact.uom, fact.scalar, d),
                        humanize_value(fact.meta["lo_end"], fact.uom, fact.scalar, d),
                        humanize_delta(fact.meta["lo_end"] - fact.meta["lo_start"], fact.uom, fact.scalar, d),
                    ],
                    [
                        f"Gap ({hi} - {lo})",
                        humanize_value(fact.meta["gap_start"], fact.uom, fact.scalar, d),
                        humanize_value(fact.meta["gap_end"], fact.uom, fact.scalar, d),
                        humanize_delta(v[0], fact.uom, fact.scalar, d),
                    ],
                ],
            }
    elif fact.probe == "long_run_compare":
        fact.human = (
            f"as of {humanize_period(fact.periods[-1])}, {_possessive(subject)}{of_m} was "
            f"{humanize_delta(v[0], fact.uom, fact.scalar, d)} than in "
            f"{humanize_period(fact.periods[0])} "
            f"({humanize_value(v[2], fact.uom, fact.scalar, d)} against "
            f"{humanize_value(v[1], fact.uom, fact.scalar, d)}){among}"
        )
    elif fact.probe == "period_change":
        verb = "rose" if v[0] >= 0 else "fell"
        fact.human = (
            f"between {humanize_period(fact.periods[0])} and {humanize_period(fact.periods[-1])}, "
            f"{_possessive(subject)}{of_m} {verb} by "
            f"{humanize_delta(abs(v[0]), fact.uom, fact.scalar, d, include_direction=False)} "
            f"to {humanize_value(v[2], fact.uom, fact.scalar, d)}{among}"
        )
    elif fact.probe == "share_of_total":
        fact.human = (
            f"between {humanize_period(fact.periods[0])} and {humanize_period(fact.periods[-1])}, "
            f"{_possessive(subject)} share of{of_m} {fact.meta['direction']} "
            f"from {fact.meta['share_then']:.1f}% to {fact.meta['share_now']:.1f}%{among}"
        )
    elif fact.probe == "rank_order":
        top = _display_name(fact.meta["leader"], cfg)
        bottom = _display_name(fact.meta["trailer"], cfg)
        fact.human = (
            f"in {humanize_period(fact.periods[-1])}, {top} led{of_m} at "
            f"{humanize_value(v[0], fact.uom, fact.scalar, d)}, compared with "
            f"{humanize_value(v[1], fact.uom, fact.scalar, d)} for {bottom}{among}"
        )


def _slice_for_probe(
    df: pd.DataFrame, cfg: dict[str, Any], measure: str | None, probe_dim: str
) -> pd.DataFrame:
    out = df
    mdim = cfg.get("measure_dimension")
    if mdim and measure is not None and mdim in out.columns:
        out = out[out[mdim] == measure]
    for dim, member in (cfg.get("hold_at") or {}).items():
        if dim != probe_dim and dim in out.columns:
            out = out[out[dim] == member]
    return out


def _dedupe_latest(facts: list[ProbeFact]) -> list[ProbeFact]:
    latest: dict[tuple, ProbeFact] = {}
    for fact in sorted(facts, key=lambda f: f.periods[-1]):
        key = (fact.probe, tuple(sorted(fact.cut.items())))
        prior = latest.get(key)
        if prior and "periods_observed" not in fact.meta:
            fact.meta["periods_observed"] = prior.meta.get("periods_observed", 1) + 1
        elif "periods_observed" not in fact.meta:
            fact.meta["periods_observed"] = 1
        latest[key] = fact
    return list(latest.values())


def run_probes(
    df: pd.DataFrame, dimensions: list[str], cfg: dict[str, Any]
) -> list[ProbeFact]:
    """Execute deterministic statistical probes across sliced cube dimensions."""
    mdim = cfg.get("measure_dimension")
    measures = list(cfg.get("measures") or [None])
    hold = cfg.get("hold_at") or {}
    probe_dims = [dim for dim in dimensions if dim in hold] or [
        dim for dim in dimensions if dim != mdim
    ]
    rates = set(cfg.get("rate_measures") or [])
    counts = set(cfg.get("count_measures") or [])
    primary_count = cfg.get("primary_count_measure")
    window_n = int(cfg.get("window_periods") or 0)
    comparable = cfg.get("comparable_members") or {}

    facts: list[ProbeFact] = []
    for measure in measures:
        for dim in probe_dims:
            sliced = _slice_for_probe(df, cfg, measure, dim)
            if dim not in sliced.columns or sliced[dim].nunique() < 2:
                continue
            # Ensure every remaining dimension is pinned uniquely to avoid mixing series.
            non_target = [
                c
                for c in sliced.columns
                if c not in {dim, mdim, "REF_DATE", "VALUE", "VECTOR", "COORDINATE",
                             "STATUS", "SYMBOL", "TERMINATED", "DECIMALS", "DGUID",
                             "UOM", "UOM_ID", "SCALAR_FACTOR", "SCALAR_ID", "SE", "SE_MOM", "SE_YOY"}
            ]
            for extra_dim in non_target:
                if sliced[extra_dim].nunique() > 1:
                    first_val = sliced[extra_dim].iloc[0]
                    sliced = sliced[sliced[extra_dim] == first_val]

            for name, fn in PROBES.items():
                if name in _CROSS_MEMBER_PROBES and rates and measure not in rates:
                    continue
                if name in _SHARE_PROBES and counts and measure not in counts:
                    continue
                if name in _RATE_ONLY_PROBES and rates and measure not in rates:
                    continue
                if name in _SHARE_PROBES and primary_count and measure in counts and measure != primary_count:
                    continue

                frame = sliced
                if window_n and name not in _FULL_HISTORY_PROBES:
                    keep_periods = sorted(frame["REF_DATE"].unique())[-window_n:]
                    frame = frame[frame["REF_DATE"].isin(keep_periods)]
                if comparable.get(dim):
                    frame = frame[frame[dim].isin(comparable[dim])]
                if frame[dim].nunique() < 2:
                    continue

                for fact in fn(frame, dim, cfg):
                    fact.meta["measure"] = measure
                    fact.meta["held_at"] = {
                        k: val
                        for k, val in (cfg.get("hold_at") or {}).items()
                        if k != dim and k not in fact.cut
                    }
                    if mdim and measure is not None:
                        fact.cut = {**fact.cut, mdim: measure}
                    _describe_fact(fact, measure, mdim, cfg)
                    facts.append(fact)

    return _dedupe_latest(facts)


def gate_and_rank_facts(
    facts: list[ProbeFact], mentions: dict[str, bool], cfg: dict[str, Any]
) -> list[ProbeFact]:
    """Filter statistically unreliable findings and rank survivors by salience and novelty."""
    aggregates = {
        name for names in (cfg.get("aggregate_members") or {}).values() for name in names
    }
    min_mag = float(cfg.get("min_magnitude", 0.0))
    kept: list[ProbeFact] = []
    for fact in facts:
        cut_vals = {str(v) for v in fact.cut.values()}
        if cut_vals & aggregates:
            continue
        if fact.se is not None and fact.magnitude <= fact.se:
            continue
        if fact.se is None and fact.magnitude < min_mag:
            continue
        kept.append(fact)

    if not kept:
        return []

    mentioned_members = {k.split("|", 1)[1] for k, seen in (mentions or {}).items() if seen}
    peak: dict[str, float] = {}
    for fact in kept:
        peak[fact.probe] = max(peak.get(fact.probe, 0.0), fact.magnitude)

    weights = cfg.get("weights") or {
        "magnitude": 0.30,
        "persistence": 0.20,
        "legibility": 0.15,
        "reader_scale": 0.15,
        "change_story": 0.20,
    }
    demotion = float(cfg.get("mention_demotion", 0.5))

    for fact in kept:
        max_mag = peak.get(fact.probe, 0.0)
        mag_score = fact.magnitude / max_mag if max_mag else 0.0
        persistence = min(1.0, float(fact.meta.get("periods_observed", 1)) / 12.0)
        u = (fact.uom or "").lower()
        reader_scale = 1.0 if ("person" in u or "dollar" in u or "percent" in u) else 0.5
        change_score = 1.0 if fact.probe in _CHANGE_PROBES else 0.0
        fact.score = (
            weights["magnitude"] * mag_score
            + weights["persistence"] * persistence
            + weights["legibility"] * fact.legibility
            + weights["reader_scale"] * reader_scale
            + weights.get("change_story", 0.0) * change_score
        )
        fact.mentioned = any(
            member and member in str(val)
            for val in fact.cut.values()
            for member in mentioned_members
        )
        if fact.mentioned:
            fact.score *= demotion

    return sorted(kept, key=lambda f: f.score, reverse=True)
