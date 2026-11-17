"""Automated live Statistics Canada The Daily & WDS cube harvester."""
from __future__ import annotations

import html as html_mod
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from .probes import (
    ProbeFact,
    extract_mentions,
    gate_and_rank_facts,
    humanize_delta,
    humanize_period,
    humanize_value,
    prepare_cube_frame,
    run_probes,
)
from .tools import RemineTools, save_snapshot

STATCAN_BASE = "https://www150.statcan.gc.ca"
DAILY_INDEX_URL = f"{STATCAN_BASE}/n1/dai-quo/index-eng.htm"
WDS_BASE = f"{STATCAN_BASE}/t1/wds/rest"

_ARTICLE_HREF = re.compile(
    r'href="(/n1)?(/daily-quotidien/(\d{6})/(dq\d{6}[a-z])-eng\.htm)"'
)
_PID_LINK = re.compile(r"tv\.action\?pid=(\d{10})")
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
_SELECT_RE = re.compile(r"<select\b.*?</select>", re.S | re.I)
_TABLEBTN_RE = re.compile(
    r'<div\b[^>]*\bclass="[^"]*\btablebtn\b[^"]*"[^>]*>.*?</div>', re.S | re.I
)
_MAIN_RE = re.compile(r"<main\b[^>]*>(.*?)</main>", re.S | re.I)
_WB_CONT_RE = re.compile(r'<[^>]+\bid="wb-cont"[^>]*>(.*)', re.S | re.I)
_MWSBODYTEXT_RE = re.compile(
    r'<[^>]+\bclass="[^"]*\bmwsbodytext\b[^"]*"[^>]*>(.*?)</div>', re.S | re.I
)

_PROVINCES = (
    "Newfoundland and Labrador",
    "Prince Edward Island",
    "Nova Scotia",
    "New Brunswick",
    "Quebec",
    "Ontario",
    "Manitoba",
    "Saskatchewan",
    "Alberta",
    "British Columbia",
)

SURVEY_PRESETS: dict[int, dict[str, Any]] = {
    14100287: {
        "survey": "Labour Force Survey",
        "value_dimension": "Statistics",
        "value_member": "Estimate",
        "se_members": {
            "level": "Standard error of estimate",
            "month_over_month": "Standard error of month-to-month change",
            "year_over_year": "Standard error of year-over-year change",
        },
        "filters": {"Data type": "Seasonally adjusted"},
        "aggregate_members": {
            "Geography": ["Canada"],
            "Gender": ["Total - Gender"],
            "Age group": ["15 years and over", "15 to 64 years"],
            "Labour force characteristics": ["Population", "Labour force"],
        },
        "hold_at": {
            "Geography": "Canada",
            "Gender": "Total - Gender",
            "Age group": "25 to 54 years",
        },
        "measure_dimension": "Labour force characteristics",
        "measures": [
            "Employment",
            "Employment rate",
            "Unemployment rate",
            "Participation rate",
        ],
        "rate_measures": ["Unemployment rate", "Employment rate", "Participation rate"],
        "count_measures": ["Employment"],
        "primary_count_measure": "Employment",
        "comparable_members": {
            "Geography": list(_PROVINCES),
            "Age group": ["15 to 24 years", "25 to 54 years", "55 years and over"],
            "Gender": ["Men+", "Women+"],
        },
        "display_names": {
            "25 to 54 years": "core-aged workers",
            "15 to 24 years": "young workers",
            "55 years and over": "workers aged 55 and over",
            "Women+": "core-aged women",
            "Men+": "core-aged men",
        },
        "synonyms": {
            "youth": "15 to 24 years",
            "young people": "15 to 24 years",
            "core-aged": "25 to 54 years",
            "seniors": "55 years and over",
            "women": "Women+",
            "men": "Men+",
        },
        "legibility": {
            "Geography": 1.0,
            "Age group": 0.9,
            "Gender": 0.9,
            "Labour force characteristics": 0.7,
        },
        "window_periods": 60,
        "min_magnitude": 0.1,
        "semantic_rule": {
            "comparative_dim": "Geography",
            "confounder_dim": "Age group",
            "required_benchmark": "25 to 54 years",
            "unstandardized_aggregate": "15 years and over",
            "explanation": (
                "Raw 15+ provincial rate gaps reflect population age composition "
                "rather than purely labour-market differences."
            ),
        },
    },
    34100292: {
        "survey": "Building Permits Survey",
        "filters": {"Seasonal adjustment, value type": "Seasonally adjusted, current"},
        "aggregate_members": {
            "Geography": ["Canada"],
            "Type of building": [
                "Total residential and non-residential",
                "Total major permits residential and non-residential",
                "Total minor residential and non-residential",
            ],
            "Type of work": ["Types of work, total"],
        },
        "hold_at": {
            "Geography": "Canada",
            "Type of building": "Total residential and non-residential",
            "Type of work": "Types of work, total",
            "Variables": "Value of permits",
        },
        "comparable_members": {
            "Geography": list(_PROVINCES),
            "Type of building": [
                "Total residential",
                "Single-detached",
                "Multi-unit",
                "Total non-residential",
                "Total commercial",
                "Total industrial",
                "Total institutional and governmental",
            ],
        },
        "legibility": {"Geography": 1.0, "Type of building": 0.9},
        "window_periods": 36,
        "min_magnitude": 100.0,
    },
    24100056: {
        "survey": "Leading indicator of international arrivals to Canada",
        "aggregate_members": {
            "Geography": ["Canada"],
            "Country of residence": [
                "International visitors entering or returning to Canada, air",
                "Non-resident visitors entering Canada, air",
                "Residents of countries other than the United States of America entering Canada, air",
            ],
            "Traveller type": ["Visitors"],
        },
        "hold_at": {
            "Geography": "Canada",
            "Traveller type": "Visitors",
            "Country of residence": "International visitors entering or returning to Canada, air",
        },
        "comparable_members": {
            "Traveller type": ["Excursionists (same-day)", "Tourists (overnight)"],
            "Country of residence": [
                "United States of America residents entering Canada, air",
                "United Kingdom",
                "France",
                "Germany",
                "Japan",
                "China",
                "India",
                "Mexico",
            ],
        },
        "legibility": {"Geography": 0.9, "Country of residence": 0.95, "Traveller type": 0.85},
        "window_periods": 30,
        "min_magnitude": 10.0,
    },
}


@dataclass(frozen=True)
class DailyArticle:
    """Parsed Statistics Canada The Daily release article and its linked table PIDs."""

    url: str
    title: str
    slug: str
    date: str
    pids: tuple[int, ...]
    prose: str
    highlights: str


@dataclass(frozen=True)
class CubeMember:
    """Single member inside a StatCan WDS cube dimension."""

    member_id: int
    name: str
    parent_id: int | None = None
    geo_level: int | None = None
    uom_code: int | None = None


@dataclass(frozen=True)
class CubeDimension:
    """Single dimension inside a StatCan WDS cube."""

    position_id: int
    name: str
    members: tuple[CubeMember, ...]


@dataclass(frozen=True)
class CubeMeta:
    """Metadata returned by Statistics Canada's WDS getCubeMetadata endpoint."""

    pid: int
    title: str
    release_time: str
    frequency_code: int
    dimensions: tuple[CubeDimension, ...]


def _http_get_text(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.encoding or "utf-8"
    return resp.text


def _http_get_json(url: str) -> Any:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _http_post_json(url: str, payload: Any) -> Any:
    resp = requests.post(url, json=payload, timeout=45)
    resp.raise_for_status()
    return resp.json()


def _clean_html_text(fragment: str) -> str:
    cleaned = _SCRIPT_STYLE_RE.sub(" ", fragment)
    cleaned = _SELECT_RE.sub(" ", cleaned)
    cleaned = _TABLEBTN_RE.sub(" ", cleaned)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = html_mod.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_body_html(html: str) -> str:
    for pattern in (_MAIN_RE, _WB_CONT_RE, _MWSBODYTEXT_RE):
        match = pattern.search(html)
        if match:
            return match.group(1)
    return html


def _extract_highlights(prose: str) -> str:
    idx = prose.find("Highlights")
    snippet = prose[idx + len("Highlights") :].strip() if idx != -1 else prose
    sentences = re.split(r"(?<=[.!?])\s+", snippet)
    summary = " ".join(sentences[:4]).strip()
    return summary[:900] if summary else prose[:500]


def parse_daily_index(html: str, target_date: str | None = None) -> tuple[str, list[str]]:
    """Extract release date (YYMMDD) and unique article URLs from The Daily index HTML."""
    matches = _ARTICLE_HREF.findall(html)
    if not matches:
        raise ValueError("no Daily article links found on index page")

    resolved_date = target_date or matches[0][2]
    urls: list[str] = []
    for _, path, found_date, _slug in matches:
        if found_date != resolved_date:
            continue
        url = f"{STATCAN_BASE}/n1{path}"
        if url not in urls:
            urls.append(url)
    return resolved_date, urls


def parse_daily_article(html: str, url: str) -> DailyArticle:
    """Parse one Daily HTML page into structured prose, highlights, and 8-digit table PIDs."""
    pids = tuple(sorted({int(pid[:8]) for pid in _PID_LINK.findall(html)}))
    if not pids:
        raise ValueError(f"no product IDs found in {url}")

    title_match = _TITLE_RE.search(html)
    raw_title = _clean_html_text(title_match.group(1)) if title_match else url
    title = raw_title.split("—")[-1].strip() or raw_title

    slug_match = re.search(r"(dq(\d{6})[a-z])", url)
    slug = slug_match.group(1) if slug_match else url.rsplit("/", 1)[-1]
    date = slug_match.group(2) if slug_match else ""
    prose = _clean_html_text(_extract_body_html(html))
    return DailyArticle(
        url=url,
        title=title,
        slug=slug,
        date=date,
        pids=pids,
        prose=prose,
        highlights=_extract_highlights(prose),
    )


def discover_daily_articles(
    date: str | None = None,
    fetch_text: Callable[[str], str] = _http_get_text,
) -> tuple[str, list[DailyArticle]]:
    """Discover Daily articles that cite StatCan cubes for the latest or requested YYMMDD date."""
    resolved_date, urls = parse_daily_index(fetch_text(DAILY_INDEX_URL), target_date=date)
    if not urls and date:
        # Older releases drop off the front index; probe standard dqYYMMDD[a-f] URLs directly.
        urls = [
            f"{STATCAN_BASE}/n1/daily-quotidien/{date}/dq{date}{suffix}-eng.htm"
            for suffix in ("a", "b", "c", "d", "e", "f")
        ]

    articles: list[DailyArticle] = []
    for url in urls:
        try:
            page_html = fetch_text(url)
            articles.append(parse_daily_article(page_html, url))
        except (requests.RequestException, ValueError):
            continue
    if not articles:
        raise ValueError(f"no Daily articles citing a StatCan cube found for {resolved_date}")
    return resolved_date, articles


def parse_cube_metadata(payload: list[dict[str, Any]]) -> CubeMeta:
    """Parse WDS getCubeMetadata JSON into CubeMeta."""
    entry = payload[0]
    if entry.get("status") != "SUCCESS":
        raise RuntimeError(f"WDS getCubeMetadata failed: {entry}")
    obj = entry["object"]
    dims = tuple(
        CubeDimension(
            position_id=int(dim["dimensionPositionId"]),
            name=dim["dimensionNameEn"],
            members=tuple(
                CubeMember(
                    member_id=int(m["memberId"]),
                    name=m["memberNameEn"],
                    parent_id=m.get("parentMemberId"),
                    geo_level=m.get("geoLevel"),
                    uom_code=m.get("memberUomCode"),
                )
                for m in dim["member"]
            ),
        )
        for dim in sorted(obj["dimension"], key=lambda d: int(d["dimensionPositionId"]))
    )
    return CubeMeta(
        pid=int(obj["productId"]),
        title=obj["cubeTitleEn"],
        release_time=obj.get("releaseTime", ""),
        frequency_code=int(obj.get("frequencyCode", 0)),
        dimensions=dims,
    )


def fetch_cube_metadata(
    pid: int,
    post_json: Callable[[str, Any], Any] = _http_post_json,
) -> CubeMeta:
    """Fetch metadata for one Statistics Canada cube PID via WDS."""
    payload = post_json(f"{WDS_BASE}/getCubeMetadata", [{"productId": int(pid)}])
    return parse_cube_metadata(payload)


def fetch_code_sets(
    cache_dir: Path | None = None,
    get_json: Callable[[str], Any] = _http_get_json,
) -> dict[str, dict[int, str]]:
    """Fetch or load cached WDS code lookup tables for status, UOM, and scalar factors."""
    cache_file = cache_dir / "codesets.json" if cache_dir else None
    if cache_file and cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        raw = get_json(f"{WDS_BASE}/getCodeSets")["object"]
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    return {
        "status": {
            int(item["statusCode"]): (item.get("statusRepresentationEn") or "")
            for item in raw.get("status", [])
        },
        "uom": {
            int(item["memberUomCode"]): (item.get("memberUomEn") or "")
            for item in raw.get("uom", [])
        },
        "scalar": {
            int(item["scalarFactorCode"]): (item.get("scalarFactorDescEn") or "")
            for item in raw.get("scalar", [])
        },
    }


def infer_survey_config(meta: CubeMeta) -> dict[str, Any]:
    """Build probe configuration from curated presets or infer clean defaults from CubeMeta."""
    if meta.pid in SURVEY_PRESETS:
        return dict(SURVEY_PRESETS[meta.pid])

    aggregate_members: dict[str, list[str]] = {}
    hold_at: dict[str, str] = {}
    comparable_members: dict[str, list[str]] = {}
    filters: dict[str, str] = {}
    value_dimension: str | None = None
    value_member: str | None = None

    for dim in meta.dimensions:
        names = [m.name for m in dim.members]
        if dim.name == "Statistics" and "Estimate" in names:
            value_dimension = "Statistics"
            value_member = "Estimate"
            continue
        if "Seasonal" in dim.name and any("Seasonally adjusted" in n for n in names):
            sa = next(n for n in names if "Seasonally adjusted" in n)
            filters[dim.name] = sa
            continue

        aggs = [
            m.name
            for m in dim.members
            if m.parent_id is None
            or m.name.startswith(("Total", "All ", "Canada"))
        ]
        if aggs:
            aggregate_members[dim.name] = aggs[:4]
        hold_at[dim.name] = aggs[0] if aggs else names[0]

        if dim.name == "Geography":
            provinces = [n for n in names if n in _PROVINCES]
            if provinces:
                comparable_members[dim.name] = provinces
            else:
                non_agg = [n for n in names if n not in set(aggs)]
                comparable_members[dim.name] = non_agg[:10]
        else:
            non_agg = [n for n in names if n not in set(aggs)]
            if non_agg:
                comparable_members[dim.name] = non_agg[:10]

    return {
        "survey": meta.title,
        "value_dimension": value_dimension,
        "value_member": value_member,
        "filters": filters,
        "aggregate_members": aggregate_members,
        "hold_at": hold_at,
        "comparable_members": comparable_members,
        "legibility": {"Geography": 1.0},
        "window_periods": 36,
        "min_magnitude": 0.1,
    }


def _coord_string(positions: dict[int, int]) -> str:
    parts = [str(positions.get(idx, 0)) for idx in range(1, 11)]
    return ".".join(parts)


def build_probe_coordinates(
    meta: CubeMeta, cfg: dict[str, Any]
) -> list[tuple[str, dict[str, str], int]]:
    """Build targeted 10-part StatCan WDS coordinates needed to probe a cube rapidly."""
    dim_by_name = {dim.name: dim for dim in meta.dimensions}
    member_id_lookup = {
        dim.name: {m.name: m.member_id for m in dim.members} for dim in meta.dimensions
    }
    base_members: dict[str, str] = {}
    for dim in meta.dimensions:
        if dim.name in (cfg.get("filters") or {}):
            base_members[dim.name] = cfg["filters"][dim.name]
        elif dim.name in (cfg.get("hold_at") or {}):
            base_members[dim.name] = cfg["hold_at"][dim.name]
        elif dim.name == cfg.get("value_dimension") and cfg.get("value_member"):
            base_members[dim.name] = cfg["value_member"]
        else:
            base_members[dim.name] = dim.members[0].name

    mdim = cfg.get("measure_dimension")
    measures = list(cfg.get("measures") or [None])
    value_variants = [base_members[cfg["value_dimension"]]] if cfg.get("value_dimension") else [None]
    for se_name in (cfg.get("se_members") or {}).values():
        if cfg.get("value_dimension") and se_name in member_id_lookup.get(cfg["value_dimension"], {}):
            value_variants.append(se_name)

    hold = cfg.get("hold_at") or {}
    probe_dims = [d.name for d in meta.dimensions if d.name in hold] or [
        d.name for d in meta.dimensions if d.name not in {mdim, cfg.get("value_dimension")}
    ]
    comparable = cfg.get("comparable_members") or {}

    requests_map: dict[str, tuple[dict[str, str], int]] = {}

    def _add_coord(combo: dict[str, str], latest_n: int) -> None:
        pos_map = {
            dim_by_name[d_name].position_id: member_id_lookup[d_name][m_name]
            for d_name, m_name in combo.items()
            if d_name in member_id_lookup and m_name in member_id_lookup[d_name]
        }
        coord = _coord_string(pos_map)
        prev = requests_map.get(coord)
        if prev is None or latest_n > prev[1]:
            requests_map[coord] = (dict(combo), latest_n)

    for measure in measures:
        for p_dim in probe_dims:
            members = comparable.get(p_dim) or [m.name for m in dim_by_name[p_dim].members[:10]]
            for member in members:
                for val_var in value_variants:
                    combo = dict(base_members)
                    combo[p_dim] = member
                    if mdim and measure:
                        combo[mdim] = measure
                    if cfg.get("value_dimension") and val_var:
                        combo[cfg["value_dimension"]] = val_var
                    n_periods = 650 if (p_dim == "Gender" and cfg.get("survey") == "Labour Force Survey") else 60
                    _add_coord(combo, n_periods)

        # Also fetch unstandardized Geography slice if a semantic confounder rule is configured.
        s_rule = cfg.get("semantic_rule")
        if s_rule and s_rule["comparative_dim"] in dim_by_name and s_rule["confounder_dim"] in dim_by_name:
            geo_members = comparable.get(s_rule["comparative_dim"], _PROVINCES[:4])
            for geo in geo_members[:4]:
                combo = dict(base_members)
                combo[s_rule["comparative_dim"]] = geo
                combo[s_rule["confounder_dim"]] = s_rule["unstandardized_aggregate"]
                if mdim and measure:
                    combo[mdim] = measure
                _add_coord(combo, 12)

    return [(coord, combo, n) for coord, (combo, n) in requests_map.items()]


def fetch_cube_slice_via_wds(
    meta: CubeMeta,
    cfg: dict[str, Any],
    codesets: dict[str, dict[int, str]],
    cache_dir: Path | None = None,
    post_json: Callable[[str, Any], Any] = _http_post_json,
) -> pd.DataFrame:
    """Fetch targeted cube time series directly from WDS coordinate endpoints with caching."""
    stamp = re.sub(r"[-:T]", "", meta.release_time or "latest")
    cache_file = cache_dir / f"wds_{meta.pid}_{stamp}.json" if cache_dir else None
    coord_specs = build_probe_coordinates(meta, cfg)
    coord_to_combo = {coord: combo for coord, combo, _ in coord_specs}

    if cache_file and cache_file.exists():
        responses = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        responses = []
        batch_size = 80
        for idx in range(0, len(coord_specs), batch_size):
            batch = [
                {"productId": meta.pid, "coordinate": coord, "latestN": latest_n}
                for coord, _, latest_n in coord_specs[idx : idx + batch_size]
            ]
            responses.extend(
                post_json(f"{WDS_BASE}/getDataFromCubePidCoordAndLatestNPeriods", batch)
            )
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(responses), encoding="utf-8")

    # Resolve dimension-level UOM fallback from metadata when memberUomCode is present.
    member_uom_map: dict[tuple[str, str], str] = {}
    for dim in meta.dimensions:
        for m in dim.members:
            if m.uom_code:
                member_uom_map[(dim.name, m.name)] = codesets["uom"].get(m.uom_code, "")

    rows: list[dict[str, Any]] = []
    for entry in responses:
        if entry.get("status") != "SUCCESS":
            continue
        obj = entry.get("object") or {}
        coord = obj.get("coordinate", "")
        combo = coord_to_combo.get(coord)
        if not combo:
            continue
        vector_id = f"v{obj.get('vectorId', '')}"
        uom_text = ""
        for d_name, m_name in combo.items():
            if (d_name, m_name) in member_uom_map:
                uom_text = member_uom_map[(d_name, m_name)]
                break

        for pt in obj.get("vectorDataPoint", []):
            status_rep = codesets["status"].get(int(pt.get("statusCode", 0) or 0), "")
            scalar_desc = codesets["scalar"].get(int(pt.get("scalarFactorCode", 0) or 0), "units")
            row = {
                **combo,
                "REF_DATE": pt["refPer"],
                "VALUE": pt.get("value"),
                "VECTOR": vector_id,
                "COORDINATE": coord,
                "STATUS": status_rep,
                "DECIMALS": int(pt.get("decimals", 1) or 1),
                "SCALAR_FACTOR": scalar_desc,
                "UOM": uom_text or ("Dollars" if "permit" in meta.title.lower() else ""),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def load_cube_from_zip(zip_path: Path, pid: int) -> pd.DataFrame:
    """Load a full-table Statistics Canada CSV zip archive (used for offline testing or bulk cache)."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"{pid}.csv") as handle:
            df = pd.read_csv(handle, encoding="utf-8-sig", dtype=str, low_memory=False)
    df.columns = [col.lstrip("\ufeff") for col in df.columns]
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    return df


def _build_methodology_rules(meta: CubeMeta, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    s_rule = cfg.get("semantic_rule")
    if s_rule:
        rules.append(
            {
                "id": "M1",
                "title": f"{s_rule['confounder_dim']}-structure confounding in {s_rule['comparative_dim']} comparisons",
                "rule": (
                    f"Do not publish raw comparisons across {s_rule['comparative_dim']} while "
                    f"{s_rule['confounder_dim']} is held at {s_rule['unstandardized_aggregate']!r}; "
                    f"standardize to {s_rule['required_benchmark']!r}. {s_rule['explanation']}"
                ),
                "source": f"surveys.yaml#{meta.pid}",
                "confirmed": True,
                "semantic_rule": s_rule,
            }
        )
    else:
        rules.append(
            {
                "id": "M1",
                "title": "Dimensional comparability and aggregate isolation",
                "rule": "Compare non-aggregate partition members on the same unit and seasonal-adjustment basis.",
                "source": f"wds:getCubeMetadata#{meta.pid}",
                "confirmed": True,
            }
        )

    rules.extend(
        [
            {
                "id": "M2",
                "title": "Population denominator and subgroup precision",
                "rule": "Name the exact held population denominator and dimension cut rather than generalizing to an unconditioned total.",
                "source": f"wds:getCubeMetadata#{meta.pid}",
                "confirmed": True,
            },
            {
                "id": "M3",
                "title": "Endpoint and multi-period comparison caveats",
                "rule": "Long-run and window changes compare specific start and end periods rather than asserting linear paths or causal proof.",
                "source": f"wds:getCubeMetadata#{meta.pid}",
                "confirmed": True,
            },
            {
                "id": "M4",
                "title": "Coverage subtraction against The Daily prose",
                "rule": "Skip candidate bundles whose main cut and movement duplicate figures already highlighted in The Daily article.",
                "source": "remine/tools.py#check_coverage_overlaps",
                "confirmed": True,
            },
            {
                "id": "M5",
                "title": "Unconfirmed informal seasonal speculation",
                "rule": "Unverified scratch note: treat single-period weather swings as permanent structural breaks.",
                "source": "scratch/unverified-notes.txt",
                "confirmed": False,
            },
        ]
    )
    return rules


def _fact_to_brief_entry(
    idx: int,
    release_id: str,
    fact: ProbeFact,
    mentions: dict[str, bool],
    methodology_ids: list[str],
    *,
    brief_id: str,
    title: str,
    summary: str,
    standardized: bool = True,
    confound_risk: str | None = None,
) -> dict[str, Any]:
    held_at = dict(fact.meta.get("held_at") or {})
    cut_mentions: dict[str, bool] = {}
    for key, val in fact.cut.items():
        base_dim = key.split("_", 1)[0]
        m_key = f"{base_dim}|{val}"
        if m_key in mentions:
            cut_mentions[m_key] = mentions[m_key]

    return {
        "id": brief_id,
        "release_id": release_id,
        "title": title,
        "summary": summary,
        "quality_flag": "A",
        "reliable": True,
        "standardized": standardized,
        "confound_risk": confound_risk,
        "methodology_ids": methodology_ids,
        "mentions": cut_mentions,
        "facts": [
            {
                "id": f"F{idx}_1",
                "probe_type": fact.probe,
                "human": fact.human,
                "cut": dict(fact.cut),
                "held_at": held_at,
                "vectors": list(fact.vectors),
                "periods": list(fact.periods),
                **({"breakdown": fact.meta["breakdown"]} if "breakdown" in fact.meta else {}),
            }
        ],
    }


def _find_unstandardized_confound_brief(
    df: pd.DataFrame,
    meta: CubeMeta,
    cfg: dict[str, Any],
    release_id: str,
    mentions: dict[str, bool],
    idx: int,
) -> dict[str, Any] | None:
    s_rule = cfg.get("semantic_rule")
    if not s_rule:
        return None
    comp_dim = s_rule["comparative_dim"]
    conf_dim = s_rule["confounder_dim"]
    agg_val = s_rule["unstandardized_aggregate"]
    if comp_dim not in df.columns or conf_dim not in df.columns:
        return None

    sub = df[df[conf_dim] == agg_val]
    mdim = cfg.get("measure_dimension")
    rate_measures = cfg.get("rate_measures") or []
    measure = rate_measures[0] if rate_measures else None
    if mdim and measure and mdim in sub.columns:
        sub = sub[sub[mdim] == measure]
    if sub[comp_dim].nunique() < 2:
        return None

    latest = sorted(sub["REF_DATE"].unique())[-1]
    latest_rows = sub[(sub["REF_DATE"] == latest) & (~sub[comp_dim].isin((cfg.get("aggregate_members") or {}).get(comp_dim, [])))]
    if len(latest_rows) < 2:
        return None
    latest_rows = latest_rows.sort_values("VALUE", ascending=False)
    hi, lo = latest_rows.iloc[0], latest_rows.iloc[-1]
    gap = float(hi["VALUE"] - lo["VALUE"])
    uom = str(hi.get("UOM", ""))
    scalar = str(hi.get("SCALAR_FACTOR", ""))
    decimals = int(hi.get("DECIMALS", 1) or 1)
    human = (
        f"in {humanize_period(latest)}, the unstandardized {agg_val} {str(measure or '').lower()} "
        f"gap between {hi[comp_dim]} and {lo[comp_dim]} was "
        f"{humanize_delta(gap, uom, scalar, decimals, include_direction=False)} "
        f"({humanize_value(float(hi['VALUE']), uom, scalar, decimals)} vs "
        f"{humanize_value(float(lo['VALUE']), uom, scalar, decimals)})"
    )
    cut = {f"{comp_dim}_high": str(hi[comp_dim]), f"{comp_dim}_low": str(lo[comp_dim])}
    if measure and mdim:
        cut[mdim] = measure
    return {
        "id": f"LIVE-{idx}",
        "release_id": release_id,
        "title": f"Unstandardized {agg_val} {comp_dim.lower()} gap: {hi[comp_dim]} vs {lo[comp_dim]}",
        "summary": (
            f"Compares {comp_dim} on unstandardized {conf_dim}='{agg_val}' instead of "
            f"benchmark '{s_rule['required_benchmark']}'."
        ),
        "quality_flag": "A",
        "reliable": True,
        "standardized": False,
        "confound_risk": f"Raw {comp_dim} comparison leaves {conf_dim} at {agg_val!r} (violates M1).",
        "methodology_ids": ["M1"],
        "mentions": {f"{comp_dim}|{hi[comp_dim]}": mentions.get(f"{comp_dim}|{hi[comp_dim]}", False)},
        "facts": [
            {
                "id": f"F{idx}_1",
                "probe_type": "gap_between_members",
                "human": human,
                "cut": cut,
                "held_at": {conf_dim: agg_val},
                "vectors": [str(hi["VECTOR"]), str(lo["VECTOR"])],
                "periods": [str(latest)],
            }
        ],
    }


def _build_suppressed_brief(
    suppressed_samples: list[dict[str, Any]],
    meta: CubeMeta,
    release_id: str,
    idx: int,
) -> dict[str, Any] | None:
    if not suppressed_samples:
        return None
    row = suppressed_samples[0]
    flag = str(row.get("STATUS") or "F").strip() or "F"
    dim_cols = [d.name for d in meta.dimensions if d.name in row]
    cut = {dim: str(row[dim]) for dim in dim_cols[:2]}
    held_at = {dim: str(row[dim]) for dim in dim_cols[2:4]}
    return {
        "id": f"LIVE-{idx}",
        "release_id": release_id,
        "title": f"Suppressed cell in {meta.title[:48]} (Quality Flag {flag})",
        "summary": f"Statistics Canada suppressed or flagged this estimate (STATUS={flag!r}).",
        "quality_flag": flag,
        "reliable": False,
        "methodology_ids": ["M2"],
        "mentions": {},
        "facts": [
            {
                "id": f"F{idx}_1",
                "probe_type": "suppressed_cell",
                "human": f"suppressed estimate (quality flag {flag})",
                "cut": cut or {"Cube": str(meta.pid)},
                "held_at": held_at,
                "vectors": [str(row.get("VECTOR") or f"v{meta.pid}_suppressed")],
                "periods": [str(row.get("REF_DATE") or "latest")],
            }
        ],
    }


def _extract_coverage_manifest(
    article: DailyArticle,
    mentions: dict[str, bool],
    ranked_facts: list[ProbeFact],
) -> dict[str, Any]:
    mentioned_pairs = [
        tuple(k.split("|", 1))
        for k, seen in mentions.items()
        if seen and "|" in k
    ]
    coords: list[dict[str, Any]] = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", article.highlights) if s.strip()]
    for fact in ranked_facts:
        if not fact.mentioned or len(coords) >= 3:
            continue
        dims = {k.split("_", 1)[0]: str(v) for k, v in fact.cut.items()}
        quote = next(
            (s for s in sentences if any(val.lower() in s.lower() for val in dims.values())),
            article.highlights[:220],
        )
        coords.append(
            {
                "tier": "daily",
                "prose_quote": quote,
                "dimensions": dims,
                "value": fact.values[0] if fact.values else None,
                "period": fact.periods[-1] if fact.periods else None,
            }
        )
    if not coords and mentioned_pairs:
        dims = {dim: member for dim, member in mentioned_pairs[:3]}
        coords.append(
            {
                "tier": "daily",
                "prose_quote": article.highlights[:220],
                "dimensions": dims,
                "value": None,
                "period": None,
            }
        )
    return {
        "release_id": article.slug,
        "paper_catalogue": None,
        "claimed_coordinates": coords,
    }


def build_snapshot_from_release(
    article: DailyArticle,
    meta: CubeMeta,
    raw_df: pd.DataFrame,
    cfg: dict[str, Any],
    max_briefs: int = 5,
) -> dict[str, Any]:
    """Convert a live Daily article and cube DataFrame into a RemineTools snapshot dict."""
    reliable_df, suppressed_samples = prepare_cube_frame(raw_df, cfg)
    dimensions_info = [(d.name, tuple(m.name for m in d.members)) for d in meta.dimensions]
    mentions = extract_mentions(article.prose, dimensions_info, cfg.get("synonyms"))

    probe_dims = [d.name for d in meta.dimensions if d.name in reliable_df.columns]
    facts = run_probes(reliable_df, probe_dims, cfg)
    ranked = gate_and_rank_facts(facts, mentions, cfg)

    briefs: list[dict[str, Any]] = []
    idx = 1

    # Select up to 3 unmentioned/novel high-scoring findings first, then 1 already-mentioned slice.
    novel = [f for f in ranked if not f.mentioned]
    mentioned = [f for f in ranked if f.mentioned]

    for fact in novel[:3]:
        cut_summary = ", ".join(f"{k}={v}" for k, v in fact.cut.items())
        briefs.append(
            _fact_to_brief_entry(
                idx,
                article.slug,
                fact,
                mentions,
                ["M2", "M3"],
                brief_id=f"LIVE-{idx}",
                title=f"{fact.probe.replace('_', ' ').title()}: {cut_summary}"[:96],
                summary=f"Deterministic {fact.probe} finding ({fact.human[:140]}).",
            )
        )
        idx += 1

    if mentioned:
        fact = mentioned[0]
        cut_summary = ", ".join(f"{k}={v}" for k, v in fact.cut.items())
        briefs.append(
            _fact_to_brief_entry(
                idx,
                article.slug,
                fact,
                mentions,
                ["M4", "M3"],
                brief_id=f"LIVE-{idx}",
                title=f"Daily-mentioned cut ({fact.probe}): {cut_summary}"[:96],
                summary=f"Cut shares dimension members explicitly named in The Daily ({fact.human[:120]}).",
            )
        )
        idx += 1

    if len(briefs) < max_briefs:
        confound_brief = _find_unstandardized_confound_brief(
            reliable_df, meta, cfg, article.slug, mentions, idx
        )
        if confound_brief:
            briefs.append(confound_brief)
            idx += 1

    if len(briefs) < max_briefs:
        suppressed_brief = _build_suppressed_brief(suppressed_samples, meta, article.slug, idx)
        if suppressed_brief:
            briefs.append(suppressed_brief)
            idx += 1

    # Fallback if novel+mentioned were sparse: top remaining ranked findings.
    if not briefs and ranked:
        for fact in ranked[:max_briefs]:
            cut_summary = ", ".join(f"{k}={v}" for k, v in fact.cut.items())
            briefs.append(
                _fact_to_brief_entry(
                    idx,
                    article.slug,
                    fact,
                    mentions,
                    ["M2", "M3"],
                    brief_id=f"LIVE-{idx}",
                    title=f"{fact.probe.replace('_', ' ').title()}: {cut_summary}"[:96],
                    summary=f"Deterministic {fact.probe} finding ({fact.human[:140]}).",
                )
            )
            idx += 1

    if not briefs:
        raise ValueError(f"no reliable candidate briefs could be generated for {article.slug} (pid={meta.pid})")

    snapshot = {
        "release": {
            "id": article.slug,
            "date": f"20{article.date[:2]}-{article.date[2:4]}-{article.date[4:6]}"
            if len(article.date) == 6
            else article.date,
            "title": article.title,
            "url": article.url,
            "pid": meta.pid,
            "table_url": f"{STATCAN_BASE}/t1/tbl1/en/tv.action?pid={meta.pid}01",
            "highlights": article.highlights,
        },
        "coverage_manifest": _extract_coverage_manifest(article, mentions, ranked),
        "methodology_rules": _build_methodology_rules(meta, cfg),
        "candidate_briefs": briefs,
        "editorial_memory": [
            {
                "id": "H1",
                "summary": "Rejected unstandardized cross-geography comparisons when compositional confounders were held at an aggregate total.",
            },
            {
                "id": "H2",
                "summary": "Rejected draft prose containing unverified quantity words or bare digits; all claims must bind via {{fact_id.human}}.",
            },
        ],
    }
    RemineTools(snapshot)  # Validate contract before returning.
    return snapshot


def harvest_daily_release(
    date: str | None = None,
    pid: int | None = None,
    out_dir: Path | None = None,
    cache_dir: Path | None = None,
    fetch_text: Callable[[str], str] = _http_get_text,
    get_json: Callable[[str], Any] = _http_get_json,
    post_json: Callable[[str, Any], Any] = _http_post_json,
) -> dict[str, Any]:
    """Discover a live Daily release, fetch its cube data via WDS, and build a verified snapshot."""
    resolved_date, articles = discover_daily_articles(date=date, fetch_text=fetch_text)
    selected_article: DailyArticle | None = None
    selected_pid: int | None = None

    if pid is not None:
        for art in articles:
            if pid in art.pids:
                selected_article = art
                selected_pid = pid
                break
        if selected_article is None:
            selected_article = articles[0]
            selected_pid = pid
    else:
        # Prefer curated surveys (like LFS 14100287) first when multiple Daily articles exist.
        for art in articles:
            for art_pid in art.pids:
                if art_pid in SURVEY_PRESETS:
                    selected_article = art
                    selected_pid = art_pid
                    break
            if selected_article:
                break
        if selected_article is None:
            selected_article = articles[0]
            selected_pid = selected_article.pids[0]

    assert selected_article is not None and selected_pid is not None
    meta = fetch_cube_metadata(selected_pid, post_json=post_json)
    codesets = fetch_code_sets(cache_dir=cache_dir, get_json=get_json)
    cfg = infer_survey_config(meta)
    df = fetch_cube_slice_via_wds(
        meta, cfg, codesets, cache_dir=cache_dir, post_json=post_json
    )
    snapshot = build_snapshot_from_release(selected_article, meta, df, cfg)

    saved_path: Path | None = None
    if out_dir is not None:
        filename = f"{resolved_date}-{selected_article.slug}-{selected_pid}.json"
        saved_path = save_snapshot(snapshot, out_dir / filename)

    return {
        "date": resolved_date,
        "article": selected_article.title,
        "slug": selected_article.slug,
        "pid": selected_pid,
        "snapshot_path": str(saved_path) if saved_path else None,
        "brief_count": len(snapshot["candidate_briefs"]),
        "snapshot": snapshot,
    }
