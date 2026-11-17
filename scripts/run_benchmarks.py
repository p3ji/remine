"""Reproducible benchmark & Gate 0 ablation runner across Replay and local /v1 models."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remine.agents import OpenAICompatibleModel, ReplayModel  # noqa: E402
from remine.evaluation import count_llm_calls, evaluate  # noqa: E402
from remine.tools import RemineTools, list_snapshots  # noqa: E402
from remine.workflow import run_snapshot  # noqa: E402


def run_one(
    dataset_name: str,
    snapshot_path: Path,
    model_name: str,
    *,
    base_url: str,
    gate0: bool,
    labels: dict[str, dict[str, str | None]],
    replay_responses: dict[str, Any],
) -> dict[str, Any]:
    """Execute one Remine snapshot run and capture accuracy, call count, and latency."""
    tools = RemineTools(snapshot_path)
    model: Any = (
        ReplayModel(replay_responses)
        if model_name == "replay"
        else OpenAICompatibleModel(model=model_name, base_url=base_url)
    )
    t0 = time.perf_counter()
    states = run_snapshot(
        tools,
        model,
        max_attempts=2,
        enforce_semantic_standardization=gate0,
        use_mcp=True,
    )
    elapsed_s = round(time.perf_counter() - t0, 2)
    call_log = getattr(model, "call_log", None) or []
    if call_log and all(not rec.ok for rec in call_log):
        raise RuntimeError(
            f"All LLM calls failed for {model_name} on {dataset_name}: {call_log[0].error}"
        )
    brief_ids = {s["brief"].id for s in states}
    ev = evaluate(states, labels) if brief_ids <= labels.keys() else None

    verdicts = {s["brief"].id: s["decision"].verdict.value for s in states}
    attempts = {s["brief"].id: s["decision"].attempts for s in states}
    return {
        "dataset": dataset_name,
        "snapshot_file": snapshot_path.name,
        "model": model_name,
        "gate0": gate0,
        "verdict_accuracy": round(ev.verdict_accuracy, 4) if ev else None,
        "rejection_accuracy": round(ev.rejection_accuracy, 4) if ev else None,
        "correct_verdicts": ev.correct_verdicts if ev else None,
        "total_briefs": len(states),
        "recovered_via_retry": ev.recovered_via_retry if ev else sum(
            1 for s in states if s["decision"].verdict.value == "publish" and s["decision"].attempts > 1
        ),
        "llm_calls": count_llm_calls(states, model=model),
        "elapsed_s": elapsed_s,
        "verdicts": verdicts,
        "attempts": attempts,
    }


def main() -> None:
    data_dir = ROOT / "data"
    base_url = "http://127.0.0.1:1234/v1"
    labels = json.loads((data_dir / "teaching_labels.json").read_text(encoding="utf-8"))
    replay_responses = json.loads(
        (data_dir / "replay_responses.json").read_text(encoding="utf-8")
    )

    datasets = [
        ("LFS Aug 2026 (Benchmark)", data_dir / "lfs_260807_snapshot.json"),
        ("AI Adoption Q2 2026 (Benchmark)", data_dir / "ai_adoption_snapshot.json"),
    ]
    models = [
        "replay",
        "qwen3.8-27b",
        "deepseek-v4-flash@q8_0",
        "microsoft/phi-4",
    ]

    runs: list[dict[str, Any]] = []
    for ds_name, ds_path in datasets:
        for model_name in models:
            for gate0 in (False, True):
                print(
                    f"Running {ds_name} | model={model_name} | gate0={gate0}...",
                    flush=True,
                )
                row = run_one(
                    ds_name,
                    ds_path,
                    model_name,
                    base_url=base_url,
                    gate0=gate0,
                    labels=labels,
                    replay_responses=replay_responses,
                )
                runs.append(row)
                acc = (
                    f"{row['correct_verdicts']}/{row['total_briefs']}"
                    if row["correct_verdicts"] is not None
                    else "unlabelled"
                )
                print(
                    f"  -> acc={acc} calls={row['llm_calls']} "
                    f"retries={row['recovered_via_retry']} time={row['elapsed_s']}s",
                    flush=True,
                )

    # Also validate one live harvested snapshot if present using qwen3.8-27b.
    harvested = [s for s in list_snapshots(data_dir) if s["source"] == "harvested"]
    if harvested:
        first = harvested[0]
        print(f"Running live harvested {first['name']} with qwen3.8-27b...", flush=True)
        row = run_one(
            f"Live Harvest: {first['title'][:42]}",
            Path(first["path"]),
            "qwen3.8-27b",
            base_url=base_url,
            gate0=True,
            labels=labels,
            replay_responses=replay_responses,
        )
        runs.append(row)
        print(f"  -> verdicts={row['verdicts']} time={row['elapsed_s']}s", flush=True)

    out_path = data_dir / "benchmark_results.json"
    out_path.write_text(
        json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": runs}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(runs)} benchmark runs to {out_path}")


if __name__ == "__main__":
    main()
