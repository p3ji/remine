# remine — Agent Guide

> Single source of truth for *how to work on this repo*. Claude and Antigravity both read this (`CLAUDE.md` → `@AGENTS.md`; `GEMINI.md` → pointer). Keep it short. *(Auto-generated 2026-08-08; edit freely — re-runs won't overwrite an existing AGENTS.md.)*

**Brain note (goals, backlog, full context):** `/Users/pushp/Brain2/Projects/remine.md`
**GitHub:** https://github.com/p3ji/remine.git
**Stack:** Python 3.11+ (`src/remine/`, `/Users/pushp/Library/Python/watspeed-venv` compatible) · LangGraph · JSON-RPC 2.0 MCP · Pandas · Zero-dependency HTTP web workbench (`src/remine/web/`).

## Run / build / test
- **Run tests (21 cases):** `python -m unittest discover -s tests -v`
- **Replay audit CLI:** `PYTHONPATH=src python -m remine.cli run --mode replay --semantic-gate`
- **Launch Web UI:** `PYTHONPATH=src python -m remine.cli serve --port 8765`
- **Harvest StatCan live:** `PYTHONPATH=src python -m remine.cli harvest`
- **Live `/v1` inference:** `PYTHONPATH=src python -m remine.cli run --mode live --model qwen3.8-27b --base-url http://127.0.0.1:1234/v1 --semantic-gate`

## Conventions & gotchas
- Keep `workflow.py` on LangGraph best practices (`Command[Literal[...]]`, `RetryPolicy`, `operator.add` reducers, fail-closed `review` escalation on model errors, surface unexpected bugs).
- Any `/v1` OpenAI-compatible client calls must keep the `/v1` suffix and handle `<think>` / `reasoning_content` fallback (`agents.py`).
- Keep `AGENTS.md` short (<100 lines); put backlog/context in the linked Brain note.

## Do NOT
- Commit secrets (`.env`), `data/cache/`, large temporary outputs, or build artifacts.
