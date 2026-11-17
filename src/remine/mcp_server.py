"""Zero-dependency Model Context Protocol (MCP) server and in-process JSON-RPC 2.0 client.

Exposes Remine's methodology lookup, editorial memory recall, Two-Tier coverage
manifest inspection, and Statistics Canada WDS/Daily harvesting tools over standard
JSON-RPC 2.0 (`initialize`, `tools/list`, `tools/call`) via either `stdio` transport
or in-process `MCPToolClient` dispatch for the LangGraph workflow.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

from . import harvester
from .tools import RemineTools

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "remine-mcp", "version": "0.2.0"}


@dataclass(frozen=True)
class MCPToolSpec:
    """Metadata and handler binding for one registered MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]

    def to_mcp_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def _jsonrpc_result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _mcp_text_result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, ensure_ascii=False)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": payload if isinstance(payload, dict) else {"items": payload},
        "isError": is_error,
    }


class RemineMCPServer:
    """JSON-RPC 2.0 MCP server wrapping RemineTools and live StatCan harvesters."""

    def __init__(
        self,
        tools: RemineTools,
        data_dir: Path | None = None,
        harvest_fn: Callable[..., dict[str, Any]] = harvester.harvest_daily_release,
        cube_meta_fn: Callable[[int], harvester.CubeMeta] = harvester.fetch_cube_metadata,
    ):
        self.tools = tools
        self.data_dir = data_dir or (
            tools.snapshot_path.parent if tools.snapshot_path else Path("data")
        )
        self._harvest_fn = harvest_fn
        self._cube_meta_fn = cube_meta_fn
        self._registry: dict[str, MCPToolSpec] = self._build_registry()

    def _build_registry(self) -> dict[str, MCPToolSpec]:
        specs = (
            MCPToolSpec(
                name="list_candidate_briefs",
                description=(
                    "List candidate statistical briefs in the loaded snapshot with "
                    "quality flags, reliability status, and linked methodology IDs."
                ),
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self._tool_list_candidate_briefs,
            ),
            MCPToolSpec(
                name="search_methodology_rules",
                description=(
                    "Search human-confirmed survey methodology rules scoped strictly "
                    "to a candidate brief's linked methodology_ids."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "brief_id": {"type": "string", "description": "Candidate brief ID."},
                        "query": {"type": "string", "description": "Keywords from the brief."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                    },
                    "required": ["brief_id", "query"],
                    "additionalProperties": False,
                },
                handler=self._tool_search_methodology_rules,
            ),
            MCPToolSpec(
                name="recall_editorial_memory",
                description=(
                    "Retrieve similar prior editorial rulings from past briefs as non-binding "
                    "context only (never proof for quantitative claims)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Topic or brief summary text."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 2},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=self._tool_recall_editorial_memory,
            ),
            MCPToolSpec(
                name="inspect_coverage_manifest",
                description=(
                    "Check whether a candidate brief overlaps Tier 1 (The Daily) or "
                    "Tier 2 (Analytical Paper) claims in the CoverageManifest."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "brief_id": {"type": "string", "description": "Candidate brief ID."},
                    },
                    "required": ["brief_id"],
                    "additionalProperties": False,
                },
                handler=self._tool_inspect_coverage_manifest,
            ),
            MCPToolSpec(
                name="fetch_statcan_cube_metadata",
                description=(
                    "Query Statistics Canada Web Data Service (WDS) getCubeMetadata for an "
                    "8-digit cube PID to inspect dimensions, release time, and member counts."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "pid": {"type": "integer", "description": "8-digit StatCan cube PID."},
                    },
                    "required": ["pid"],
                    "additionalProperties": False,
                },
                handler=self._tool_fetch_statcan_cube_metadata,
            ),
            MCPToolSpec(
                name="harvest_daily_release",
                description=(
                    "Discover a live Statistics Canada The Daily release, pull its WDS cube "
                    "slice, run deterministic probes, and return a validated snapshot summary."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "Optional YYMMDD date."},
                        "pid": {"type": "integer", "description": "Optional 8-digit cube PID."},
                        "save": {"type": "boolean", "default": True},
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_harvest_daily_release,
            ),
        )
        return {spec.name: spec for spec in specs}

    def _tool_list_candidate_briefs(self, _args: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "id": brief.id,
                "title": brief.title,
                "quality_flag": brief.quality_flag,
                "reliable": brief.reliable,
                "standardized": brief.standardized,
                "is_decomposition": brief.is_decomposition,
                "fact_ids": [fact.id for fact in brief.facts],
                "methodology_ids": list(brief.methodology_ids),
            }
            for brief in self.tools.list_briefs()
        ]

    def _tool_search_methodology_rules(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        brief_id = str(args.get("brief_id", "")).strip()
        query = str(args.get("query", "")).strip()
        if not brief_id or not query:
            raise ValueError("search_methodology_rules requires non-empty brief_id and query")
        limit = int(args.get("limit", 3))
        brief = self.tools.get_brief(brief_id)
        rules = self.tools.search_methodology(brief, query=query, limit=limit)
        return [asdict(rule) for rule in rules]

    def _tool_recall_editorial_memory(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("recall_editorial_memory requires non-empty query")
        limit = int(args.get("limit", 2))
        return list(self.tools.recall_editorial_memory(query=query, limit=limit))

    def _tool_inspect_coverage_manifest(self, args: dict[str, Any]) -> dict[str, Any]:
        brief_id = str(args.get("brief_id", "")).strip()
        if not brief_id:
            raise ValueError("inspect_coverage_manifest requires non-empty brief_id")
        brief = self.tools.get_brief(brief_id)
        overlaps = self.tools.check_coverage_overlaps(brief)
        manifest = self.tools.coverage_manifest
        return {
            "brief_id": brief.id,
            "has_manifest": manifest is not None,
            "paper_catalogue": manifest.paper_catalogue if manifest else None,
            "overlap_count": len(overlaps),
            "overlaps": overlaps,
        }

    def _tool_fetch_statcan_cube_metadata(self, args: dict[str, Any]) -> dict[str, Any]:
        if "pid" not in args:
            raise ValueError("fetch_statcan_cube_metadata requires integer pid")
        pid = int(args["pid"])
        meta = self._cube_meta_fn(pid)
        return {
            "pid": meta.pid,
            "title": meta.title,
            "release_time": meta.release_time,
            "frequency_code": meta.frequency_code,
            "dimensions": [
                {
                    "position_id": dim.position_id,
                    "name": dim.name,
                    "member_count": len(dim.members),
                    "sample_members": [m.name for m in dim.members[:5]],
                }
                for dim in meta.dimensions
            ],
        }

    def _tool_harvest_daily_release(self, args: dict[str, Any]) -> dict[str, Any]:
        save = bool(args.get("save", True))
        out_dir = (self.data_dir / "harvested") if save else None
        cache_dir = self.data_dir / "cache"
        pid_arg = args.get("pid")
        pid = int(pid_arg) if pid_arg is not None else None
        date_arg = str(args["date"]).strip() if args.get("date") else None
        result = self._harvest_fn(
            date=date_arg,
            pid=pid,
            out_dir=out_dir,
            cache_dir=cache_dir,
        )
        return {k: v for k, v in result.items() if k != "snapshot"}

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Process one JSON-RPC 2.0 message and return a response dict (or None for notifications)."""
        if request.get("jsonrpc") != "2.0" or "method" not in request:
            return _jsonrpc_error(request.get("id"), -32600, "Invalid JSON-RPC 2.0 Request")

        method = request["method"]
        req_id = request.get("id")
        params = request.get("params") or {}

        # Notifications carry no id and expect no reply per JSON-RPC 2.0 / MCP.
        if req_id is None:
            return None

        if method == "initialize":
            return _jsonrpc_result(
                req_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                },
            )
        if method == "ping":
            return _jsonrpc_result(req_id, {})
        if method == "tools/list":
            return _jsonrpc_result(
                req_id,
                {"tools": [spec.to_mcp_dict() for spec in self._registry.values()]},
            )
        if method == "tools/call":
            return self._handle_tool_call(req_id, params)

        return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")

    def _handle_tool_call(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(params, dict) or "name" not in params:
            return _jsonrpc_error(req_id, -32602, "tools/call requires 'name'")
        name = str(params["name"])
        spec = self._registry.get(name)
        if spec is None:
            return _jsonrpc_error(req_id, -32602, f"Unknown MCP tool: {name}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _jsonrpc_error(req_id, -32602, "tools/call 'arguments' must be an object")

        try:
            payload = spec.handler(arguments)
        except Exception as exc:  # noqa: BLE001 - surface cleanly as MCP tool error
            return _jsonrpc_result(req_id, _mcp_text_result(str(exc), is_error=True))
        return _jsonrpc_result(req_id, _mcp_text_result(payload, is_error=False))

    def serve_stdio(
        self,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        """Run a line-delimited JSON-RPC 2.0 MCP loop over stdin/stdout."""
        in_stream = stdin or sys.stdin
        out_stream = stdout or sys.stdout
        for raw_line in in_stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                resp = _jsonrpc_error(None, -32700, f"Parse error: {exc}")
            else:
                resp = self.handle_request(msg)
            if resp is not None:
                out_stream.write(json.dumps(resp, ensure_ascii=False) + "\n")
                out_stream.flush()


class MCPToolClient:
    """In-process JSON-RPC 2.0 MCP client that logs every tool invocation for auditability."""

    def __init__(self, server: RemineMCPServer):
        self.server = server
        self._next_id = 1
        self.call_log: list[dict[str, Any]] = []
        self._initialized = False

    def initialize(self) -> dict[str, Any]:
        """Perform standard MCP initialize + notifications/initialized handshake."""
        if not self._initialized:
            resp = self._rpc("initialize", {"protocolVersion": PROTOCOL_VERSION, "clientInfo": {"name": "remine-workflow", "version": "0.2.0"}})
            self.server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"})
            self._initialized = True
            return resp
        return {"protocolVersion": PROTOCOL_VERSION}

    def list_tools(self) -> list[dict[str, Any]]:
        """List advertised MCP tools via `tools/list`."""
        self.initialize()
        return list(self._rpc("tools/list", {}).get("tools", []))

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Invoke an MCP tool over JSON-RPC 2.0 and append an audited call record."""
        self.initialize()
        args = dict(arguments or {})
        t0 = time.perf_counter()
        result = self._rpc("tools/call", {"name": name, "arguments": args})
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        is_error = bool(result.get("isError", False))
        structured = result.get("structuredContent", {})
        payload = structured["items"] if isinstance(structured, dict) and set(structured) == {"items"} else structured
        record = {
            "protocol": "mcp-jsonrpc-2.0",
            "tool": name,
            "arguments": args,
            "ok": not is_error,
            "duration_ms": duration_ms,
            "result_count": len(payload) if isinstance(payload, (list, dict)) else 1,
        }
        self.call_log.append(record)
        if is_error:
            text = (result.get("content") or [{}])[0].get("text", "MCP tool error")
            raise RuntimeError(f"MCP tool '{name}' failed: {text}")
        return payload

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        response = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }
        )
        if response is None:
            raise RuntimeError(f"Expected JSON-RPC response for {method}")
        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"MCP JSON-RPC error {err.get('code')}: {err.get('message')}")
        return dict(response["result"])
