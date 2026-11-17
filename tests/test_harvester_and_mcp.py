"""Unit and integration tests for deterministic probes, StatCan harvester, and MCP server."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remine import cli, harvester, probes  # noqa: E402
from remine.mcp_server import MCPToolClient, RemineMCPServer  # noqa: E402
from remine.tools import RemineTools  # noqa: E402


class ProbesTests(unittest.TestCase):
    def test_humanizers_format_periods_levels_and_deltas(self) -> None:
        self.assertEqual(probes.humanize_period("2026-10"), "October 2026")
        self.assertEqual(probes.humanize_period("2026-08-07"), "August 7, 2026")
        self.assertEqual(probes.humanize_value(65.43, "Percent", "units"), "65.4%")
        self.assertEqual(probes.humanize_value(123.4, "Persons", "thousands"), "about 123,000 people")
        self.assertEqual(probes.humanize_value(12.5, "Dollars", "thousands"), "$12,500")
        self.assertEqual(
            probes.humanize_delta(-2.5, "Percent", "units"),
            "2.5 percentage points lower",
        )
        self.assertEqual(
            probes.humanize_delta(14.2, "Persons", "thousands"),
            "about 14,000 more people",
        )

    def test_extract_mentions_respects_boundaries_and_synonyms(self) -> None:
        hits = probes.extract_mentions(
            "Youth employment fell in Ontario, while Canada edged down.",
            [("Geography", ("Canada", "Ontario", "Alberta")), ("Age", ("15 to 24 years", "25 to 54 years"))],
            synonyms={"youth": "15 to 24 years", "core-aged": "25 to 54 years"},
        )
        self.assertTrue(hits["Geography|Ontario"])
        self.assertFalse(hits["Geography|Alberta"])
        self.assertTrue(hits["Age|15 to 24 years"])
        self.assertFalse(hits["Age|25 to 54 years"])

    def test_prepare_run_probes_and_gate_rank_facts(self) -> None:
        rows = []
        for idx, period in enumerate(("2025-09", "2025-11", "2026-01", "2026-03", "2026-06", "2026-09")):
            rows.extend(
                [
                    {"REF_DATE": period, "Geography": "Alberta", "Labour": "Employment rate", "Statistics": "Estimate", "VALUE": 66.0 + idx * 0.5, "VECTOR": "v1", "UOM": "Percent", "SCALAR_FACTOR": "units", "DECIMALS": 1, "STATUS": ""},
                    {"REF_DATE": period, "Geography": "Alberta", "Labour": "Employment rate", "Statistics": "Standard error of estimate", "VALUE": 0.3, "VECTOR": "v1s", "UOM": "Percent", "SCALAR_FACTOR": "units", "DECIMALS": 1, "STATUS": ""},
                    {"REF_DATE": period, "Geography": "Ontario", "Labour": "Employment rate", "Statistics": "Estimate", "VALUE": 61.5 - idx * 0.2, "VECTOR": "v2", "UOM": "Percent", "SCALAR_FACTOR": "units", "DECIMALS": 1, "STATUS": ""},
                    {"REF_DATE": period, "Geography": "Ontario", "Labour": "Employment rate", "Statistics": "Standard error of estimate", "VALUE": 0.3, "VECTOR": "v2s", "UOM": "Percent", "SCALAR_FACTOR": "units", "DECIMALS": 1, "STATUS": ""},
                ]
            )
        rows.append({"REF_DATE": "2026-09", "Geography": "Prince Edward Island", "Labour": "Employment rate", "Statistics": "Estimate", "VALUE": None, "VECTOR": "v3", "UOM": "Percent", "SCALAR_FACTOR": "units", "DECIMALS": 1, "STATUS": "F"})
        df = pd.DataFrame(rows)
        cfg = {
            "value_dimension": "Statistics",
            "value_member": "Estimate",
            "se_members": {"level": "Standard error of estimate"},
            "measure_dimension": "Labour",
            "measures": ["Employment rate"],
            "rate_measures": ["Employment rate"],
            "hold_at": {"Geography": "Canada"},
            "comparable_members": {"Geography": ["Alberta", "Ontario", "Prince Edward Island"]},
            "legibility": {"Geography": 1.0},
            "window_periods": 12,
        }
        est, suppressed = probes.prepare_cube_frame(df, cfg)
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(suppressed[0]["STATUS"], "F")

        found = probes.run_probes(est, ["Geography"], cfg)
        probe_names = {f.probe for f in found}
        self.assertIn("gap_between_members", probe_names)
        self.assertIn("gap_trend", probe_names)
        self.assertIn("long_run_compare", probe_names)
        self.assertIn("period_change", probe_names)

        ranked = probes.gate_and_rank_facts(found, {"Geography|Ontario": True, "Geography|Alberta": False}, cfg)
        self.assertTrue(len(ranked) > 0)
        self.assertTrue(all(f.human for f in ranked))


class HarvesterTests(unittest.TestCase):
    def _sample_daily_index_html(self) -> str:
        return """
        <html><body>
          <a href="/n1/daily-quotidien/261106/dq261106a-eng.htm">Labour Force Survey, October 2026</a>
          <a href="/n1/daily-quotidien/261106/dq261106b-eng.htm">Building permits, September 2026</a>
        </body></html>
        """

    def _sample_article_html(self) -> str:
        return """
        <html>
          <head><title>The Daily — Labour Force Survey, October 2026</title></head>
          <body>
            <main>
              <h1>Labour Force Survey, October 2026</h1>
              <p>Highlights: Employment rose in Ontario in October 2026 while unemployment edged down.</p>
              <p>Table <a href="https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410028701">14-10-0287-01</a>
                 provides labour force characteristics by province.</p>
              <p>See analytical study Catalogue no. 11-621-M for national industry trends.</p>
            </main>
          </body>
        </html>
        """

    def _sample_cube_meta_payload(self) -> list[dict]:
        return [
            {
                "status": "SUCCESS",
                "object": {
                    "productId": "14100287",
                    "cubeTitleEn": "Labour force characteristics, monthly, seasonally adjusted",
                    "releaseTime": "2026-11-06T08:30",
                    "frequencyCode": 6,
                    "dimension": [
                        {
                            "dimensionPositionId": 1,
                            "dimensionNameEn": "Geography",
                            "member": [
                                {"memberId": 1, "memberNameEn": "Canada"},
                                {"memberId": 6, "memberNameEn": "Ontario", "parentMemberId": 1},
                                {"memberId": 9, "memberNameEn": "Alberta", "parentMemberId": 1},
                            ],
                        },
                        {
                            "dimensionPositionId": 2,
                            "dimensionNameEn": "Labour force characteristics",
                            "member": [
                                {"memberId": 1, "memberNameEn": "Employment"},
                                {"memberId": 2, "memberNameEn": "Employment rate"},
                            ],
                        },
                        {
                            "dimensionPositionId": 3,
                            "dimensionNameEn": "Gender",
                            "member": [
                                {"memberId": 1, "memberNameEn": "Total - Gender"},
                                {"memberId": 2, "memberNameEn": "Men+"},
                                {"memberId": 3, "memberNameEn": "Women+"},
                            ],
                        },
                        {
                            "dimensionPositionId": 4,
                            "dimensionNameEn": "Age group",
                            "member": [
                                {"memberId": 1, "memberNameEn": "15 years and over"},
                                {"memberId": 2, "memberNameEn": "15 to 24 years"},
                                {"memberId": 3, "memberNameEn": "25 to 54 years"},
                            ],
                        },
                        {
                            "dimensionPositionId": 5,
                            "dimensionNameEn": "Statistics",
                            "member": [
                                {"memberId": 1, "memberNameEn": "Estimate"},
                                {"memberId": 2, "memberNameEn": "Standard error of estimate"},
                            ],
                        },
                        {
                            "dimensionPositionId": 6,
                            "dimensionNameEn": "Data type",
                            "member": [
                                {"memberId": 1, "memberNameEn": "Seasonally adjusted"},
                            ],
                        },
                    ],
                },
            }
        ]

    def test_parse_daily_wds_slice_and_build_snapshot(self) -> None:
        date_code, urls = harvester.parse_daily_index(self._sample_daily_index_html())
        self.assertEqual(date_code, "261106")
        self.assertEqual(len(urls), 2)

        article = harvester.parse_daily_article(
            self._sample_article_html(),
            "https://www150.statcan.gc.ca/n1/daily-quotidien/261106/dq261106a-eng.htm",
        )
        self.assertEqual(article.pids, (14100287,))
        self.assertIn("11-621-M", article.prose)

        meta = harvester.parse_cube_metadata(self._sample_cube_meta_payload())
        cfg = harvester.infer_survey_config(meta)
        coords = harvester.build_probe_coordinates(meta, cfg)
        self.assertTrue(any(coord.startswith("6.") for coord, _, _ in coords))

        codesets = {
            "status": {},
            "uom": {1: "Percent", 2: "Persons"},
            "scalar": {0: "units", 3: "thousands"},
        }

        def fake_post_json(url: str, payload):
            if url.endswith("getCubeMetadata"):
                return self._sample_cube_meta_payload()
            if url.endswith("getDataFromCubePidCoordAndLatestNPeriods"):
                return [
                    {
                        "status": "SUCCESS",
                        "object": {
                            "productId": item["productId"],
                            "coordinate": item["coordinate"],
                            "vectorId": 1000 + idx,
                            "vectorDataPoint": [
                                {"refPer": "2025-10-01", "value": 65.0 + idx * 0.5, "decimals": 1, "scalarFactorCode": 0, "statusCode": 0},
                                {"refPer": "2026-10-01", "value": 66.2 + idx * 1.1, "decimals": 1, "scalarFactorCode": 0, "statusCode": 0},
                            ],
                        },
                    }
                    for idx, item in enumerate(payload)
                ]
            raise AssertionError(f"unexpected URL {url}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = harvester.harvest_daily_release(
                date="261106",
                pid=14100287,
                out_dir=tmp_path / "out",
                cache_dir=tmp_path / "cache",
                fetch_text=lambda url: (
                    self._sample_daily_index_html()
                    if url.endswith("index-eng.htm")
                    else self._sample_article_html()
                ),
                get_json=lambda _url: {
                    "object": {
                        "status": [{"statusCode": 0, "statusRepresentationEn": ""}],
                        "uom": [{"memberUomCode": 1, "memberUomEn": "Percent"}],
                        "scalar": [{"scalarFactorCode": 0, "scalarFactorDescEn": "units"}],
                    }
                },
                post_json=fake_post_json,
            )
            self.assertEqual(result["pid"], 14100287)
            self.assertGreaterEqual(result["brief_count"], 1)
            snap_file = Path(result["snapshot_path"])
            self.assertTrue(snap_file.exists())
            tools = RemineTools(snap_file)
            self.assertGreaterEqual(len(tools.list_briefs()), 1)
            self.assertIsNotNone(tools.coverage_manifest)


class MCPServerTests(unittest.TestCase):
    def test_mcp_jsonrpc_protocol_and_all_tools(self) -> None:
        tools = RemineTools(ROOT / "data/lfs_260807_snapshot.json")
        fake_meta = harvester.CubeMeta(
            pid=14100287,
            title="Labour force characteristics",
            release_time="2026-11-06T08:30",
            frequency_code=6,
            dimensions=(
                harvester.CubeDimension(
                    position_id=1,
                    name="Geography",
                    members=(harvester.CubeMember(member_id=1, name="Canada"),),
                ),
            ),
        )
        server = RemineMCPServer(
            tools,
            data_dir=ROOT / "data",
            cube_meta_fn=lambda _pid: fake_meta,
            harvest_fn=lambda **_kw: {
                "date": "261106",
                "pid": 14100287,
                "brief_count": 4,
                "snapshot_path": "mock.json",
                "snapshot": {},
            },
        )
        client = MCPToolClient(server)

        advertised = client.list_tools()
        names = {t["name"] for t in advertised}
        self.assertEqual(
            names,
            {
                "list_candidate_briefs",
                "search_methodology_rules",
                "recall_editorial_memory",
                "inspect_coverage_manifest",
                "fetch_statcan_cube_metadata",
                "harvest_daily_release",
            },
        )

        briefs = client.call_tool("list_candidate_briefs")
        self.assertEqual(len(briefs), 5)

        rules = client.call_tool(
            "search_methodology_rules",
            {"brief_id": "LFS-201", "query": "core-aged women employment rate"},
        )
        self.assertGreaterEqual(len(rules), 1)

        overlaps_lfs = client.call_tool("inspect_coverage_manifest", {"brief_id": "LFS-204"})
        self.assertFalse(overlaps_lfs["has_manifest"])

        ai_tools = RemineTools(ROOT / "data/ai_adoption_snapshot.json")
        ai_client = MCPToolClient(RemineMCPServer(ai_tools, data_dir=ROOT / "data"))
        overlaps_daily = ai_client.call_tool("inspect_coverage_manifest", {"brief_id": "AI-304"})
        self.assertTrue(overlaps_daily["has_manifest"])
        self.assertEqual(overlaps_daily["paper_catalogue"], "11-621-M")
        self.assertTrue(any(o["tier"] == "daily" for o in overlaps_daily["overlaps"]))

        overlaps_paper = ai_client.call_tool("inspect_coverage_manifest", {"brief_id": "AI-308"})
        self.assertTrue(any(o["tier"] == "analytical_paper" for o in overlaps_paper["overlaps"]))

        meta = client.call_tool("fetch_statcan_cube_metadata", {"pid": 14100287})
        self.assertEqual(meta["pid"], 14100287)

        harvested = client.call_tool("harvest_daily_release", {"pid": 14100287, "save": False})
        self.assertEqual(harvested["brief_count"], 4)
        self.assertEqual(len(client.call_log), 5)

    def test_mcp_stdio_and_error_handling(self) -> None:
        tools = RemineTools(ROOT / "data/lfs_260807_snapshot.json")
        server = RemineMCPServer(tools, data_dir=ROOT / "data")
        stdin = io.StringIO(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}) + "\n"
            + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "unknown_tool"}}) + "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "inspect_coverage_manifest", "arguments": {"brief_id": "NOPE"}}}) + "\n"
        )
        stdout = io.StringIO()
        server.serve_stdio(stdin=stdin, stdout=stdout)
        replies = [json.loads(line) for line in stdout.getvalue().strip().splitlines()]
        self.assertEqual(len(replies), 3)  # notification yields no response
        self.assertEqual(replies[0]["result"]["serverInfo"]["name"], "remine-mcp")
        self.assertIn("error", replies[1])
        self.assertTrue(replies[2]["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
