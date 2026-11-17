"""Command-line interface for Remine runs, live StatCan harvesting, MCP stdio, and web UI."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .agents import OpenAICompatibleModel, ReplayModel
from .evaluation import evaluate, serialize_run
from .harvester import harvest_daily_release
from .mcp_server import RemineMCPServer
from .tools import RemineTools
from .web import serve
from .workflow import run_snapshot

_SUBCOMMANDS = frozenset({"run", "harvest", "serve", "mcp"})


def _add_run_args(parser: argparse.ArgumentParser, default_snapshot: Path) -> None:
    parser.add_argument("--mode", choices=("replay", "live"), default="replay")
    parser.add_argument("--snapshot", type=Path, default=default_snapshot)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--model", default=os.getenv("REMINE_MODEL", "qwen3.8-27b"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("REMINE_BASE_URL", "http://127.0.0.1:1234/v1"),
    )
    parser.add_argument(
        "--semantic-gate",
        action="store_true",
        help="Enforce deterministic metadata semantic comparability at Gate 0",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Call RemineTools directly instead of routing retrieve through MCP",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON serialization instead of plain text summary",
    )


def _cmd_run(args: argparse.Namespace, root: Path) -> int:
    tools = RemineTools(args.snapshot)
    if args.mode == "replay":
        responses = json.loads(
            (root / "data/replay_responses.json").read_text(encoding="utf-8")
        )
        model = ReplayModel(responses)
        model_label = "replay"
    else:
        if not args.model:
            raise SystemExit("live mode needs --model or REMINE_MODEL")
        model = OpenAICompatibleModel(
            model=args.model,
            base_url=args.base_url,
            api_key=os.getenv("REMINE_API_KEY", "local"),
        )
        model_label = args.model

    states = run_snapshot(
        tools,
        model,
        max_attempts=args.max_attempts,
        enforce_semantic_standardization=args.semantic_gate,
        use_mcp=not args.no_mcp,
    )
    labels_path = root / "data/teaching_labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else None

    if args.json:
        payload = serialize_run(
            tools,
            states,
            model_name=model_label,
            enforce_semantic_standardization=args.semantic_gate,
            labels=labels,
            model=model,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(
        f"Mode: {model_label} | Semantic Gate 0: "
        f"{'ENABLED' if args.semantic_gate else 'DISABLED'}"
    )
    for state in states:
        d = state["decision"]
        detail = (
            d.bound_story.headline
            if d.bound_story
            else f"{d.rejection_category or ''} — {d.reason}".strip(" —")
        )
        print(f"{d.brief_id:8} {d.verdict.value:22} (attempts={d.attempts}) {detail}")
        if state.get("feedback_log"):
            for entry in state["feedback_log"]:
                print(f"           retry feedback: {entry}")
        if d.critic_note:
            print(f"           critic: {d.critic_note}")

    if labels and {s["brief"].id for s in states} <= labels.keys():
        result = evaluate(states, labels)
        print(
            f"Evaluation verdicts: {result.correct_verdicts}/{result.total}; "
            f"rejection categories: {result.correct_rejections}/{result.expected_skips}; "
            f"recovered via retry: {result.recovered_via_retry}"
        )
    return 0


def _cmd_harvest(args: argparse.Namespace, root: Path) -> int:
    out_dir = args.out_dir or (root / "data/harvested")
    cache_dir = root / "data/cache"
    result = harvest_daily_release(
        date=args.date,
        pid=args.pid,
        out_dir=out_dir,
        cache_dir=cache_dir,
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "snapshot"},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _build_parser(root: Path) -> argparse.ArgumentParser:
    default_snapshot = root / "data/lfs_260807_snapshot.json"
    parser = argparse.ArgumentParser(
        prog="remine",
        description="Remine: token-bound StatCan secondary journalism workbench",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Execute LangGraph editorial swarm on a snapshot")
    _add_run_args(run_p, default_snapshot)

    harv_p = sub.add_parser("harvest", help="Harvest live The Daily article + WDS cube")
    harv_p.add_argument("--date", default=None, help="YYMMDD release date (default: latest)")
    harv_p.add_argument("--pid", type=int, default=None, help="Optional 8-digit StatCan PID")
    harv_p.add_argument("--out-dir", type=Path, default=None)

    serve_p = sub.add_parser("serve", help="Launch interactive web workbench")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8765)
    serve_p.add_argument("--data-dir", type=Path, default=root / "data")
    serve_p.add_argument(
        "--base-url",
        default=os.getenv("REMINE_BASE_URL", "http://127.0.0.1:1234/v1"),
    )
    serve_p.add_argument("--model", default=os.getenv("REMINE_MODEL", "qwen3.8-27b"))

    mcp_p = sub.add_parser("mcp", help="Run JSON-RPC 2.0 MCP server over stdio")
    mcp_p.add_argument("--snapshot", type=Path, default=default_snapshot)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[2]
    args_list = list(sys.argv[1:] if argv is None else argv)
    # Preserve direct flag invocation (`remine --mode replay ...`) alongside subcommands.
    if not args_list or (
        args_list[0] not in _SUBCOMMANDS and args_list[0] not in ("-h", "--help")
    ):
        args_list = ["run", *args_list]

    parser = _build_parser(root)
    args = parser.parse_args(args_list)

    if args.command == "run":
        raise SystemExit(_cmd_run(args, root))
    if args.command == "harvest":
        raise SystemExit(_cmd_harvest(args, root))
    if args.command == "serve":
        serve(
            host=args.host,
            port=args.port,
            data_dir=args.data_dir,
            base_url=args.base_url,
            default_model=args.model,
        )
        return
    if args.command == "mcp":
        tools = RemineTools(args.snapshot)
        RemineMCPServer(tools, data_dir=root / "data").serve_stdio()


if __name__ == "__main__":
    main()
