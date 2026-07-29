import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import backfill_run_meta, run_meta


class RunMetaTest(unittest.TestCase):
    def test_codex_cumulative_usage_is_not_double_counted(self):
        events = [
            {"type": "thread.started", "thread_id": "thread-1"},
            {"type": "turn.completed", "usage": {
                "input_tokens": 100, "cached_input_tokens": 80,
                "cache_write_input_tokens": 0, "output_tokens": 10,
                "reasoning_output_tokens": 3,
            }},
            {"type": "thread.started", "thread_id": "thread-1"},
            {"type": "item.completed", "item": {"type": "command_execution"}},
            {"type": "turn.completed", "usage": {
                "input_tokens": 250, "cached_input_tokens": 200,
                "cache_write_input_tokens": 5, "output_tokens": 25,
                "reasoning_output_tokens": 8,
            }},
            {"type": "error", "message": "You've hit your usage limit. try again at Aug 2, 2026 5:44 PM."},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.log"
            trace.write_text("\n".join(json.dumps(event) for event in events))
            usage = run_meta.parse_trace(trace)

        self.assertEqual(usage["input_tokens"], 250)
        self.assertEqual(usage["cache_read_tokens"], 200)
        self.assertEqual(usage["turns"], 2)
        self.assertEqual(usage["tool_calls"], 1)
        self.assertEqual(len(usage["quota_messages"]), 1)

    def test_claude_sessions_sum_provider_reported_usage_and_cost(self):
        events = [
            {"type": "result", "num_turns": 2, "total_cost_usd": 1.25,
             "usage": {"input_tokens": 10, "cache_read_input_tokens": 100,
                       "cache_creation_input_tokens": 20, "output_tokens": 30}},
            {"type": "result", "num_turns": 3, "total_cost_usd": 2.5,
             "usage": {"input_tokens": 5, "cache_read_input_tokens": 50,
                       "cache_creation_input_tokens": 10, "output_tokens": 15}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.log"
            trace.write_text("\n".join(json.dumps(event) for event in events))
            usage = run_meta.parse_trace(trace)

        self.assertEqual(usage["turns"], 5)
        self.assertEqual(usage["input_tokens"], 15)
        self.assertEqual(usage["cache_read_tokens"], 150)
        self.assertEqual(usage["provider_reported_cost_usd"], 3.75)

    def test_unknown_trace_uses_nulls_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.log"
            trace.write_text("plain output with no usage events\n")
            usage = run_meta.parse_trace(trace)

            self.assertEqual(usage["source"], "unknown")
            for key in ("turns", "input_tokens", "output_tokens", "provider_reported_cost_usd"):
                self.assertIsNone(usage[key])
            with patch.dict("os.environ", {
                "AGENT": "antigravity", "AGENT_AUTH_MODE": "subscription",
                "DURATION": "60", "HOURS": "1",
                "HARNESS_GIT_COMMIT": "abc123",
                "HARNESS_GIT_DIRTY": "true",
                "SCORER_SHA256": "def456",
                "BASE_MODEL": "answerdotai/ModernBERT-base",
                "BASE_REVISION": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
            }, clear=True):
                meta = run_meta.build_meta(trace)

        self.assertEqual(meta["usage"]["measurement_status"], "unavailable")
        self.assertIsNone(meta["usage"]["api_equivalent_basic_rate_usd"])
        self.assertIsNone(meta["usage"]["uncached_input_tokens"])
        self.assertIn("null values are not zero", meta["usage"]["cost_note"])
        self.assertEqual(meta["agent_model"], "gemini-3.6-flash")

    def test_partial_claude_stream_preserves_only_observed_output(self):
        event = {"type": "assistant", "message": {
            "id": "msg-1", "usage": {"output_tokens": 12}, "content": [],
        }}
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.log"
            trace.write_text(json.dumps(event))
            usage = run_meta.parse_trace(trace)

        self.assertEqual(usage["source"], "claude-stream-partial")
        self.assertIsNone(usage["input_tokens"])
        self.assertEqual(usage["output_tokens"], 12)
        self.assertIsNone(usage["provider_reported_cost_usd"])

    def test_codex_estimate_prices_cache_writes_separately(self):
        usage = {
            "input_tokens": 1_000_000, "cache_read_tokens": 700_000,
            "cache_creation_tokens": 100_000, "output_tokens": 10_000,
        }
        self.assertEqual(run_meta.estimate_codex_cost("gpt-5.6-sol", usage), 2.275)

    def test_meta_separates_subscription_bill_from_api_equivalent(self):
        events = [
            {"type": "thread.started", "thread_id": "thread-1"},
            {"type": "turn.completed", "usage": {
                "input_tokens": 1_000_000, "cached_input_tokens": 800_000,
                "output_tokens": 10_000,
            }},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.log"
            score = Path(tmp) / "scores.json"
            trace.write_text("\n".join(json.dumps(event) for event in events))
            score.write_text(json.dumps({
                "score_schema_version": 2,
                "contamination_policy": "open-data",
                "reportability": "reportable_with_query_exposure",
                "protocol_valid": True, "mean_task": 0.5, "mean_type": 0.4,
                "heldout_per_task": {"Task": 0.5}, "skipped": [],
            }))
            with patch.dict("os.environ", {
                "AGENT": "codex", "AGENT_CONFIG": "gpt-5.6-sol",
                "AGENT_AUTH_MODE": "subscription", "DURATION": "3590",
                "HOURS": "1",
                "HARNESS_GIT_COMMIT": "abc123",
                "HARNESS_GIT_DIRTY": "true",
                "SCORER_SHA256": "def456",
                "BASE_MODEL": "answerdotai/ModernBERT-base",
                "BASE_REVISION": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
            }, clear=True):
                meta = run_meta.build_meta(trace, score)

        self.assertEqual(meta["usage"]["api_equivalent_basic_rate_usd"], 1.7)
        self.assertIsNone(meta["usage"]["actual_billed_usd"])
        self.assertIsNone(meta["usage"]["api_key_estimated_cost_usd"])
        self.assertTrue(meta["near_budget"])
        self.assertTrue(meta["score"]["protocol_valid"])
        self.assertEqual(meta["score"]["score_schema_version"], 2)
        self.assertEqual(meta["score"]["contamination_policy"], "open-data")
        self.assertEqual(meta["score"]["reportability"], "reportable_with_query_exposure")
        self.assertEqual(meta["harness_git_commit"], "abc123")
        self.assertTrue(meta["harness_git_dirty"])
        self.assertEqual(meta["scorer_sha256"], "def456")
        self.assertEqual(meta["base_revision"], "8949b909ec900327062f0ebf497f51aef5e6f0c8")

    def test_backfill_marks_no_trace_directory_as_launch_stub(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "20260728-000000_1_codex"
            run_dir.mkdir()
            self.assertTrue(backfill_run_meta.rebuild(run_dir))
            meta = json.loads((run_dir / "meta.json").read_text())

        self.assertEqual(meta["run_kind"], "launch-stub")
        self.assertFalse(meta["harness_complete"])
        self.assertEqual(meta["usage"]["measurement_status"], "unavailable")
        self.assertIsNone(meta["usage"]["actual_billed_usd"])
        self.assertIsNone(meta["usage"]["input_tokens"])

if __name__ == "__main__":
    unittest.main()
