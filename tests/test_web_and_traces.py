"""Integration tests for token bindings, workflow traces, evaluation serialization, and HTTP web workbench."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remine.agents import ReplayModel, _extract_json_object  # noqa: E402
from remine.evaluation import (  # noqa: E402
    ablation_comparison,
    count_llm_calls,
    serialize_run,
)
from remine.models import Verdict, render_bound_segments  # noqa: E402
from remine.tools import RemineTools, list_snapshots, save_snapshot  # noqa: E402
from remine.web import RemineWebApp, create_server  # noqa: E402
from remine.workflow import build_graph, run_snapshot  # noqa: E402


class TraceTokenAndAgentTests(unittest.TestCase):
    def test_extract_json_object_strips_think_blocks_and_fences(self) -> None:
        raw = '<think>pondering facts</think>\n```json\n{"approved": true, "note": "ok"}\n```'
        parsed = _extract_json_object(raw, "critic")
        self.assertEqual(parsed, {"approved": True, "note": "ok"})

    def test_workflow_emits_token_bindings_traces_mcp_and_ablation(self) -> None:
        tools = RemineTools(ROOT / "data/lfs_260807_snapshot.json")
        responses = json.loads(
            (ROOT / "data/replay_responses.json").read_text(encoding="utf-8")
        )
        labels = json.loads(
            (ROOT / "data/teaching_labels.json").read_text(encoding="utf-8")
        )
        events: list[dict] = []

        model_off = ReplayModel(responses)
        states_off = run_snapshot(
            tools,
            model_off,
            enforce_semantic_standardization=False,
            use_mcp=True,
            on_event=events.append,
        )
        self.assertGreater(len(events), 10)
        self.assertGreater(count_llm_calls(states_off, model=model_off), 0)

        lfs_201 = next(s for s in states_off if s["brief"].id == "LFS-201")
        story = lfs_201["decision"].bound_story
        self.assertIsNotNone(story)
        assert story is not None
        self.assertGreaterEqual(len(story.token_bindings), 2)
        segments = render_bound_segments(story.raw_body, {f.id: f for f in lfs_201["brief"].facts})
        token_segs = [seg for seg in segments if seg["kind"] == "token"]
        self.assertEqual(len(token_segs), len(story.token_bindings))
        self.assertGreaterEqual(len(lfs_201.get("mcp_calls", ())), 3)

        lfs_202 = next(s for s in states_off if s["brief"].id == "LFS-202")
        self.assertTrue(any(ev["node"] == "bind_check" and ev["status"] == "retry" for ev in lfs_202["trace"]))

        model_on = ReplayModel(responses)
        states_on = run_snapshot(
            tools,
            model_on,
            enforce_semantic_standardization=True,
            use_mcp=True,
        )
        ablation = ablation_comparison(states_off, states_on, labels=labels)
        self.assertGreater(ablation["llm_calls_saved"], 0)

        serialized = serialize_run(
            tools,
            states_on,
            model_name="replay",
            enforce_semantic_standardization=True,
            labels=labels,
            model=model_on,
        )
        self.assertEqual(serialized["summary"]["publish_count"], 2)
        self.assertEqual(serialized["summary"]["skip_count"], 2)
        self.assertEqual(serialized["summary"]["unreliable_count"], 1)
        self.assertEqual(len(serialized["stories"]), 2)
        self.assertEqual(len(serialized["ledger"]), 5)
        self.assertNotIn("next_step", lfs_201)

    def test_retry_policy_recovers_transient_errors_and_surfaces_bugs(self) -> None:
        from langgraph.types import RetryPolicy
        from remine.workflow import _is_transient_error

        tools = RemineTools(ROOT / "data/lfs_260807_snapshot.json")
        responses = json.loads(
            (ROOT / "data/replay_responses.json").read_text(encoding="utf-8")
        )
        replay = ReplayModel(responses)
        fast_policy = RetryPolicy(
            max_attempts=3,
            initial_interval=0.001,
            backoff_factor=1.0,
            jitter=False,
            retry_on=_is_transient_error,
        )

        # 1. Transient TimeoutError triggers LangGraph RetryPolicy without bumping domain attempts.
        angle_calls = {"n": 0}

        class TransientOnceModel:
            def complete(self, role, brief_id, system, user, allowed_sources=frozenset(), allowed_facts=frozenset(), attempt=1):
                if role == "angle":
                    angle_calls["n"] += 1
                    if angle_calls["n"] == 1:
                        raise TimeoutError("simulated transient timeout")
                return replay.complete(role, brief_id, system, user, allowed_sources, allowed_facts, attempt)

        graph = build_graph(tools, TransientOnceModel(), retry_policy=fast_policy)
        mermaid = graph.get_graph().draw_mermaid()
        self.assertIn("__start__ --> gate", mermaid)
        self.assertIn("bind_check -.-> draft", mermaid)

        out = graph.invoke({"brief": tools.get_brief("LFS-201")})
        self.assertEqual(angle_calls["n"], 2)
        self.assertEqual(out["decision"].verdict, Verdict.PUBLISH)
        self.assertEqual(out["decision"].attempts, 1)

        # 2. Malformed model output (ValueError) fails closed to Verdict.REVIEW.
        class BadModelOutput:
            def complete(self, role, brief_id, system, user, allowed_sources=frozenset(), allowed_facts=frozenset(), attempt=1):
                if role == "angle":
                    raise ValueError("angle returned invalid JSON")
                return replay.complete(role, brief_id, system, user, allowed_sources, allowed_facts, attempt)

        review_out = build_graph(tools, BadModelOutput(), retry_policy=fast_policy).invoke(
            {"brief": tools.get_brief("LFS-201")}
        )
        self.assertEqual(review_out["decision"].verdict, Verdict.REVIEW)

        # 3. Unexpected programming bugs (e.g. TypeError) are NOT swallowed as REVIEW.
        class BuggyCodeModel:
            def complete(self, role, brief_id, system, user, allowed_sources=frozenset(), allowed_facts=frozenset(), attempt=1):
                raise TypeError("unexpected internal bug")

        with self.assertRaises(TypeError):
            build_graph(tools, BuggyCodeModel(), retry_policy=fast_policy).invoke(
                {"brief": tools.get_brief("LFS-201")}
            )


class WebServerTests(unittest.TestCase):
    def _http_json(self, url: str, payload: dict | None = None) -> tuple[int, dict]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read().decode("utf-8"))

    def test_web_server_routes_run_and_harvest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            for name in ("lfs_260807_snapshot.json", "ai_adoption_snapshot.json", "replay_responses.json", "teaching_labels.json"):
                (data_dir / name).write_text(
                    (ROOT / "data" / name).read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

            def fake_harvest(date=None, pid=None, out_dir=None, cache_dir=None):
                snap = json.loads((data_dir / "lfs_260807_snapshot.json").read_text(encoding="utf-8"))
                snap["release"]["id"] = "R:HARVEST-TEST"
                snap["release"]["title"] = "Mocked Live Daily Harvest"
                path = save_snapshot(snap, out_dir / "261112-mock-14100287.json")
                return {
                    "date": date or "261112",
                    "article": "Mocked Live Daily Harvest",
                    "slug": "261112-mock",
                    "pid": pid or 14100287,
                    "snapshot_path": str(path),
                    "brief_count": len(snap["candidate_briefs"]),
                    "snapshot": snap,
                }

            app = RemineWebApp(data_dir=data_dir, harvest_fn=fake_harvest)
            server = create_server(data_dir=data_dir, host="127.0.0.1", port=0, app=app)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{port}"
                with urllib.request.urlopen(f"{base}/", timeout=5) as resp:
                    html = resp.read().decode("utf-8")
                    self.assertEqual(resp.status, 200)
                    self.assertIn("REMINE", html)

                status, health = self._http_json(f"{base}/api/health")
                self.assertEqual(status, 200)
                self.assertEqual(health["snapshot_count"], 2)
                self.assertEqual(len(health["mcp_tools"]), 6)

                status, run_payload = self._http_json(
                    f"{base}/api/run",
                    {
                        "snapshot_id": "lfs_260807_snapshot",
                        "mode": "replay",
                        "enforce_semantic_standardization": True,
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(run_payload["summary"]["publish_count"], 2)
                self.assertEqual(run_payload["summary"]["evaluation"]["correct_verdicts"], 5)

                status, harvest_payload = self._http_json(
                    f"{base}/api/harvest",
                    {"date": "261112", "pid": 14100287},
                )
                self.assertEqual(status, 200)
                self.assertEqual(harvest_payload["snapshot_id"], "261112-mock-14100287")
                self.assertGreaterEqual(len(list_snapshots(data_dir)), 3)

                bad_status, _ = self._http_json(
                    f"{base}/api/run",
                    {"snapshot_id": "does_not_exist"},
                )
                self.assertEqual(bad_status, 404)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
