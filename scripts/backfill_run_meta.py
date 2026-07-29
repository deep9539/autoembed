"""Rebuild result meta.json files from their immutable traces and score outputs."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
from scripts import run_meta


NUMERIC_ENV_DEFAULTS = {
    "SCORE_RC": "0", "HOURS": "0", "DURATION": "0", "RC": "0",
    "FINAL_FILES": "0",
}

ENV_FIELDS = {
    "RUN_ID": "run_id", "RUN_KIND": "run_kind", "AGENT": "agent", "AGENT_CONFIG": "agent_config",
    "BASE_MODEL": "base_model", "BASE_REVISION": "base_revision", "AGENT_VERSION": "agent_version",
    "AGENT_REASONING": "agent_reasoning", "AGENT_AUTH_MODE": "agent_auth_mode",
    "PROTOCOL_VERSION": "protocol_version", "CONFIG_ID": "config",
    "CONFIG_SHA256": "config_sha256",
    "HARNESS_GIT_COMMIT": "harness_git_commit",
    "SCORER_SHA256": "scorer_sha256",
    "CONTAINER_IMAGE_SHA256": "container_image_sha256", "SCORE_RC": "score_exit",
    "HARNESS_GIT_DIRTY": "harness_git_dirty",
    "HOURS": "budget_hours", "MODE": "mode", "DURATION": "duration_s",
    "RC": "agent_exit", "BUDGET_HIT": "budget_hit", "NODE": "node",
    "GPU_NAME": "gpu", "GPU_BOUNDARY": "gpu_boundary",
    "GPU_SELECTOR": "gpu_selector", "FINAL_FILES": "final_model_files",
    "HARNESS_COMPLETE": "harness_complete",
}


def _trace_duration(path):
    if not path.is_file():
        return 0
    duration = 0
    with path.open(encoding="utf-8", errors="replace") as lines:
        for line in lines:
            match = re.match(r"\[\s*([0-9.]+)s\]", line)
            if match:
                duration = max(duration, int(float(match.group(1))))
    return duration


def _config_id(config_path):
    if not config_path.is_file():
        return ""
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    for candidate in (_REPO_ROOT / "configs").rglob("*.json"):
        if hashlib.sha256(candidate.read_bytes()).hexdigest() == digest:
            return str(candidate.relative_to(_REPO_ROOT))
    return "snapshot-only"


def _partial_provenance(run_dir, trace_path):
    config_path = run_dir / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}
    agent = next(
        (name for name in ("claude", "codex", "antigravity")
         if name in run_dir.name.lower()),
        run_dir.name.rsplit("_", 1)[-1],
    )
    defaults = {
        "claude": "claude-opus-5", "codex": "gpt-5.6-sol",
        "antigravity": "gemini-3.6-flash",
    }
    final_model = run_dir / "final_model"
    return {
        "run_id": run_dir.name, "agent": agent,
        "run_kind": "experiment" if trace_path.is_file() else "launch-stub",
        "agent_config": defaults.get(agent, ""),
        "base_model": config.get("base_model", ""),
        "base_revision": config.get("base_revision", ""),
        "protocol_version": config.get("protocol_version", ""),
        "config": _config_id(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()
        if config_path.is_file() else "",
        "score_exit": 3, "budget_hours": 0, "mode": "unknown",
        "duration_s": _trace_duration(trace_path), "agent_exit": -1,
        "budget_hit": False, "final_model_files": len(list(final_model.iterdir()))
        if final_model.is_dir() else 0,
        "harness_complete": False,
    }


def rebuild(run_dir):
    meta_path = run_dir / "meta.json"
    trace_path = run_dir / "trace.log"
    old = (json.loads(meta_path.read_text()) if meta_path.is_file()
           else _partial_provenance(run_dir, trace_path))
    if not old.get("config"):
        old["config"] = _config_id(run_dir / "config.json")
    environ = {}
    for env_name, meta_name in ENV_FIELDS.items():
        value = old.get(
            meta_name, True if env_name == "HARNESS_COMPLETE" else ""
        )
        if isinstance(value, bool):
            value = str(value).lower()
        rendered = str(value if value is not None else "")
        environ[env_name] = rendered or NUMERIC_ENV_DEFAULTS.get(env_name, "")
    score_path = run_dir / "scores.json"
    with patch.dict(os.environ, environ, clear=True):
        rebuilt = run_meta.build_meta(trace_path, score_path)
    temporary = meta_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rebuilt, indent=2))
    temporary.replace(meta_path)
    return True


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    count = sum(rebuild(path) for path in sorted(root.iterdir()) if path.is_dir())
    print(f"rebuilt {count} run metadata files")


if __name__ == "__main__":
    main()
