#!/usr/bin/env python3
"""Build static GitHub Pages bundle under docs/ from src/remine/web/ and data/."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from remine.agents import ReplayModel  # noqa: E402
from remine.evaluation import serialize_run  # noqa: E402
from remine.tools import RemineTools, list_snapshots  # noqa: E402
from remine.workflow import run_snapshot  # noqa: E402


def _can_replay(tools: RemineTools, responses: dict[str, Any]) -> bool:
    """Check whether replay_responses.json covers all non-trivial briefs in a snapshot."""
    for brief in tools.list_briefs():
        if not brief.reliable:
            continue
        has_angle = f"angle:{brief.id}:attempt1" in responses or f"angle:{brief.id}" in responses
        if not has_angle:
            return False
    return True


def build_pages(
    data_dir: Path | None = None,
    web_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> Path:
    """Sync frontend assets and emit docs/static/site_data.json for static hosting."""
    data_root = (data_dir or ROOT / "data").resolve()
    web_root = (web_dir or ROOT / "src" / "remine" / "web").resolve()
    out_root = (docs_dir or ROOT / "docs").resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    static_dir = out_root / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    for asset in ("index.html", "styles.css", "app.js"):
        src = web_root / asset
        if not src.is_file():
            raise FileNotFoundError(f"Missing frontend asset: {src}")
        shutil.copy2(src, out_root / asset)

    (out_root / ".nojekyll").write_text("", encoding="utf-8")

    responses = json.loads((data_root / "replay_responses.json").read_text(encoding="utf-8"))
    labels_path = data_root / "teaching_labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.is_file() else {}

    precomputed_path = data_root / "precomputed_harvested_runs.json"
    precomputed: dict[str, dict[str, Any]] = (
        json.loads(precomputed_path.read_text(encoding="utf-8"))
        if precomputed_path.is_file()
        else {}
    )

    benchmarks_path = data_root / "benchmark_results.json"
    benchmarks = (
        json.loads(benchmarks_path.read_text(encoding="utf-8"))
        if benchmarks_path.is_file()
        else {"runs": []}
    )

    bundled_snapshots: list[dict[str, Any]] = []
    runs: dict[str, dict[str, Any]] = {}

    for snap in list_snapshots(data_root):
        sid = snap["id"]
        tools = RemineTools(Path(snap["path"]))
        if _can_replay(tools, responses):
            pair: dict[str, Any] = {}
            for enforce_gate in (False, True):
                model = ReplayModel(responses)
                states = run_snapshot(
                    tools,
                    model,
                    max_attempts=2,
                    enforce_semantic_standardization=enforce_gate,
                    use_mcp=True,
                )
                key = "strict" if enforce_gate else "off"
                pair[key] = serialize_run(
                    tools,
                    states,
                    model_name="replay",
                    enforce_semantic_standardization=enforce_gate,
                    labels=labels or None,
                    model=model,
                )
            runs[sid] = pair
            bundled_snapshots.append({k: v for k, v in snap.items() if k != "path"})
        elif sid in precomputed and {"off", "strict"} <= precomputed[sid].keys():
            runs[sid] = precomputed[sid]
            bundled_snapshots.append({k: v for k, v in snap.items() if k != "path"})

    bundle = {
        "snapshots": bundled_snapshots,
        "runs": runs,
        "benchmarks": benchmarks,
    }
    bundle_path = static_dir / "site_data.json"
    bundle_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Built static site in {out_root}: "
        f"{len(bundled_snapshots)} snapshots -> {bundle_path.name} ({bundle_path.stat().st_size:,} bytes)"
    )
    return out_root


if __name__ == "__main__":
    build_pages()
