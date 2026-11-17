"""Zero-dependency HTTP web server for inspecting Remine LangGraph runs and live harvests."""
from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.request
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from . import harvester
from .agents import OpenAICompatibleModel, ReplayModel
from .evaluation import serialize_run
from .mcp_server import RemineMCPServer
from .tools import RemineTools, list_snapshots
from .workflow import run_snapshot

_WEB_ASSETS_DIR = Path(__file__).resolve().parent / "web"
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _probe_local_models(base_url: str, timeout: float = 1.5) -> dict[str, Any]:
    """Query OpenAI-compatible /v1/models cleanly with a short non-blocking timeout."""
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        models = [item["id"] for item in payload.get("data", []) if "id" in item]
        return {"online": True, "base_url": base_url, "models": models}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"online": False, "base_url": base_url, "models": [], "error": str(exc)}


class RemineWebApp:
    """Application state and JSON API handlers backing the Remine web dashboard."""

    def __init__(
        self,
        data_dir: Path | None = None,
        base_url: str = "http://127.0.0.1:1234/v1",
        default_model: str = "qwen3.8-27b",
        harvest_fn: Callable[..., dict[str, Any]] = harvester.harvest_daily_release,
        model_factory: Callable[[str, str], Any] | None = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.base_url = base_url
        self.default_model = default_model
        self._harvest_fn = harvest_fn
        self._model_factory = model_factory

    def _load_json_optional(self, filename: str) -> dict[str, Any]:
        path = self.data_dir / filename
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def resolve_snapshot_path(self, snapshot_id: str | None) -> Path:
        """Resolve a snapshot ID or filename strictly from discovered data_dir snapshots."""
        entries = list_snapshots(self.data_dir)
        if not entries:
            raise FileNotFoundError(f"no snapshots found under {self.data_dir}")
        if not snapshot_id:
            return Path(entries[0]["path"])
        for entry in entries:
            if snapshot_id in (entry["id"], entry["name"], entry["release_id"]):
                return Path(entry["path"])
        raise FileNotFoundError(f"unknown snapshot: {snapshot_id}")

    def health(self, base_url: str | None = None) -> dict[str, Any]:
        snapshots = list_snapshots(self.data_dir)
        first_tools = RemineTools(Path(snapshots[0]["path"])) if snapshots else None
        mcp_tools: list[dict[str, Any]] = []
        if first_tools is not None:
            server = RemineMCPServer(first_tools, data_dir=self.data_dir)
            resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            if resp and "result" in resp:
                mcp_tools = resp["result"].get("tools", [])
        return {
            "status": "ok",
            "version": "0.2.0",
            "data_dir": str(self.data_dir),
            "snapshot_count": len(snapshots),
            "default_model": self.default_model,
            "local_llm": _probe_local_models(base_url or self.base_url),
            "mcp_tools": [{"name": t["name"], "description": t["description"]} for t in mcp_tools],
        }

    def snapshots(self) -> dict[str, Any]:
        return {"snapshots": list_snapshots(self.data_dir)}

    def benchmarks(self) -> dict[str, Any]:
        payload = self._load_json_optional("benchmark_results.json")
        return payload if payload else {"runs": []}

    def run(self, body: dict[str, Any]) -> dict[str, Any]:
        snapshot_path = self.resolve_snapshot_path(body.get("snapshot_id"))
        tools = RemineTools(snapshot_path)
        mode = str(body.get("mode", "replay")).lower()
        enforce_gate = bool(body.get("enforce_semantic_standardization", False))
        use_mcp = bool(body.get("use_mcp", True))
        max_attempts = max(1, min(int(body.get("max_attempts", 2)), 4))

        if mode == "replay":
            responses = self._load_json_optional("replay_responses.json")
            model_name = "replay"
            model: Any = ReplayModel(responses)
        elif mode == "live":
            model_name = str(body.get("model") or self.default_model)
            base_url = str(body.get("base_url") or self.base_url)
            if self._model_factory is not None:
                model = self._model_factory(model_name, base_url)
            else:
                model = OpenAICompatibleModel(model=model_name, base_url=base_url)
        else:
            raise ValueError(f"unsupported execution mode: {mode}")

        states = run_snapshot(
            tools,
            model,
            max_attempts=max_attempts,
            enforce_semantic_standardization=enforce_gate,
            use_mcp=use_mcp,
        )
        labels = self._load_json_optional("teaching_labels.json")
        return serialize_run(
            tools,
            states,
            model_name=model_name,
            enforce_semantic_standardization=enforce_gate,
            labels=labels or None,
            model=model,
        )

    def harvest(self, body: dict[str, Any]) -> dict[str, Any]:
        date_val = str(body["date"]).strip() if body.get("date") else None
        pid_raw = body.get("pid")
        pid_val = int(pid_raw) if pid_raw not in (None, "") else None
        result = self._harvest_fn(
            date=date_val,
            pid=pid_val,
            out_dir=self.data_dir / "harvested",
            cache_dir=self.data_dir / "cache",
        )
        saved_name = Path(result["snapshot_path"]).stem if result.get("snapshot_path") else None
        return {
            "harvested": {k: v for k, v in result.items() if k != "snapshot"},
            "snapshot_id": saved_name,
            "snapshots": list_snapshots(self.data_dir),
        }


class _RequestHandler(BaseHTTPRequestHandler):
    """HTTP router dispatching static dashboard assets and Remine JSON APIs."""

    server_version = "RemineHTTP/0.2"

    def __init__(self, app: RemineWebApp, *args: Any, **kwargs: Any):
        self.app = app
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_asset(self, filename: str) -> None:
        asset_path = (_WEB_ASSETS_DIR / filename).resolve()
        if not asset_path.is_file() or asset_path.parent != _WEB_ASSETS_DIR.resolve():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"asset not found: {filename}"})
            return
        mime_type, _ = mimetypes.guess_type(str(asset_path))
        content = asset_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime_type or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("JSON request body must be an object")
        return parsed

    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route == "/":
            self._send_asset("index.html")
        elif route in ("/styles.css", "/app.js"):
            self._send_asset(route.lstrip("/"))
        elif route == "/api/health":
            self._send_json(HTTPStatus.OK, self.app.health())
        elif route == "/api/snapshots":
            self._send_json(HTTPStatus.OK, self.app.snapshots())
        elif route == "/api/benchmarks":
            self._send_json(HTTPStatus.OK, self.app.benchmarks())
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"not found: {route}"})

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        try:
            body = self._read_json_body()
            if route == "/api/run":
                self._send_json(HTTPStatus.OK, self.app.run(body))
            elif route == "/api/harvest":
                self._send_json(HTTPStatus.OK, self.app.harvest(body))
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"not found: {route}"})
        except FileNotFoundError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - return clean HTTP 500 JSON payload
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})


def create_server(
    data_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    base_url: str = "http://127.0.0.1:1234/v1",
    default_model: str = "qwen3.8-27b",
    app: RemineWebApp | None = None,
) -> ThreadingHTTPServer:
    """Instantiate the Remine ThreadingHTTPServer without starting its blocking loop."""
    web_app = app or RemineWebApp(
        data_dir=data_dir, base_url=base_url, default_model=default_model
    )
    handler = partial(_RequestHandler, web_app)
    return ThreadingHTTPServer((host, port), handler)


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    data_dir: Path | None = None,
    base_url: str = "http://127.0.0.1:1234/v1",
    default_model: str = "qwen3.8-27b",
) -> None:
    """Start the Remine interactive web workbench and block until Ctrl-C."""
    server = create_server(
        data_dir=data_dir,
        host=host,
        port=port,
        base_url=base_url,
        default_model=default_model,
    )
    actual_host, actual_port = server.server_address
    print(f"Remine Workbench listening on http://{actual_host}:{actual_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
