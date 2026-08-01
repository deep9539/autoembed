import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = {
    name: ROOT / "agents" / name / "solve.sh"
    for name in ("claude", "codex", "gemini")
}


class AgentWrapperTest(unittest.TestCase):
    def test_wrappers_are_valid_shell(self):
        for path in WRAPPERS.values():
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_all_providers_receive_the_same_prompts_and_cutoff(self):
        initial = set()
        continuation = set()
        for path in WRAPPERS.values():
            source = path.read_text()
            initial.add(re.search(r'^INITIAL_PROMPT="(.*)"$', source, re.M).group(1))
            continuation.add(re.search(r'^    continuation="(.*)"$', source, re.M).group(1))
            self.assertIn("MIN_REMAINING_MINUTES=30", source)
            self.assertNotIn("RANDOM", source)
            self.assertNotIn("sleep 60", source)
            self.assertNotIn("until the time budget expires", source)
        self.assertEqual(len(initial), 1)
        self.assertEqual(len(continuation), 1)

    def test_each_wrapper_uses_official_headless_and_resume_flags(self):
        claude = WRAPPERS["claude"].read_text()
        self.assertIn("claude --print --verbose", claude)
        self.assertIn("--output-format stream-json", claude)
        self.assertIn("agent --continue", claude)

        codex = WRAPPERS["codex"].read_text()
        self.assertIn("codex --search exec --json", codex)
        self.assertIn("codex --search exec resume --last --json", codex)

        gemini = WRAPPERS["gemini"].read_text()
        self.assertIn("--output-format stream-json", gemini)
        self.assertIn("--resume latest -p", gemini)
        self.assertIn('selectedType":"gemini-api-key', gemini)

    def test_agent_stdin_is_never_the_terminal(self):
        # A terminal read from the sandbox's background process group raises SIGTTIN
        # and stops the CLI for the remainder of the run.
        self.assertIn("</dev/null", WRAPPERS["gemini"].read_text())

    @unittest.skipUnless(
        "</dev/null" in (ROOT / "scripts" / "run_task.sh").read_text(),
        "harness-level stdin detach and raw trace are deferred until no run is "
        "executing run_task.sh; bash re-reads the file by offset mid-run",
    )
    def test_harness_detaches_stdin_and_retains_the_raw_stream(self):
        harness = (ROOT / "scripts" / "run_task.sh").read_text()
        self.assertRegex(harness, r"sandbox \"timeout[^\n]*\" </dev/null")
        # The raw tee must precede the timestamping filter, and append like the rest.
        raw = harness.index('tee -a "$RESULTS/trace.raw.log"')
        self.assertLess(raw, harness.index("timestamp_lines.py"))

    def test_authentication_paths_are_exclusive(self):
        claude = WRAPPERS["claude"].read_text()
        self.assertIn("unset ANTHROPIC_API_KEY", claude)
        self.assertIn("unset CLAUDE_CODE_OAUTH_TOKEN", claude)

        codex = WRAPPERS["codex"].read_text()
        self.assertIn("subscription ] && unset OPENAI_API_KEY", codex)

        gemini = WRAPPERS["gemini"].read_text()
        self.assertRegex(gemini, r"unset [^\n]*\bANTHROPIC_API_KEY\b")
        self.assertRegex(gemini, r"unset [^\n]*\bOPENAI_API_KEY\b")
        self.assertIn("unset GOOGLE_API_KEY", gemini)


if __name__ == "__main__":
    unittest.main()
