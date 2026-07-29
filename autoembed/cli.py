#!/usr/bin/env python3
"""autoembed CLI — thin wrapper over run_task.sh.

  autoembed list
  autoembed run --config finance --agent claude --model claude-opus-5 --hours 10
  autoembed run --config finance --local --gpu 0            # your own workstation GPU
  autoembed run --config legal --base <hf-id> --dry-run    # base sweep, preview only

Launcher: Slurm via gpu.sh by default; --local runs here on --gpu (own GPU).
Isolation: enroot (Slurm default) / docker / native; --local defaults to docker,
falling back to native (cooperative — not an enforcement boundary).
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

def _find_root():
    candidates = []
    if os.environ.get("AUTOEMBED_ROOT"):
        candidates.append(Path(os.environ["AUTOEMBED_ROOT"]).expanduser())
    candidates.append(Path(__file__).resolve().parents[1])
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "configs").is_dir() and (root / "scripts" / "run_task.sh").is_file():
            return root
    return Path(__file__).resolve().parents[1]


ROOT = _find_root()
CONFIGS = ROOT / "configs"
DEFAULT_MODEL = {"claude": "claude-opus-5", "codex": "gpt-5.6-sol", "antigravity": "gemini-3.6-flash"}


def _configs():
    return sorted(CONFIGS.rglob("*.json")) if CONFIGS.is_dir() else []


def _resolve_config(name):
    p = Path(name)
    if p.is_file():
        return p.resolve()
    hits = [c for c in _configs() if name in (c.stem, str(c.relative_to(CONFIGS))[:-5])]
    if len(hits) == 1:
        return hits[0].resolve()
    if not hits:
        sys.exit(f"!! no config '{name}' — run: autoembed list")
    sys.exit(f"!! ambiguous '{name}': {[str(h.relative_to(CONFIGS))[:-5] for h in hits]}")


def cmd_list(_):
    configs = _configs()
    if not configs:
        sys.exit(
            "!! project resources not found; run from an autoembed source checkout "
            "or set AUTOEMBED_ROOT"
        )
    for c in configs:
        try:
            cfg = json.loads(c.read_text())
            print(f"  {str(c.relative_to(CONFIGS))[:-5]:34s} {cfg.get('protocol_type','?'):20s} base={cfg.get('base_model','?')}")
        except Exception:
            print(f"  {c.relative_to(CONFIGS)}")


def _launch(cmd, env, a):
    shown = " ".join(f"{k}={shlex.quote(env[k])}" for k in
                     ("MODE", "AUTOEMBED_CONFIG", "AUTOEMBED_BASE_MODEL", "AUTOEMBED_BASE_REVISION", "GPU_ID", "HOURS", "TIME") if k in env)
    print(f">> {shown} \\\n   {' '.join(shlex.quote(c) for c in cmd)}")
    if a.dry_run:
        print(">> (dry run — not launched)")
        return
    sys.exit(subprocess.call(cmd, env=env, cwd=str(ROOT)))


def _gpu_wrap(inner, env, a):
    if a.local:
        if a.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
        return inner
    env.setdefault("TIME", getattr(a, "time", None) or "04:00:00")
    return [str(ROOT / "scripts" / "gpu.sh"), *inner]


def cmd_reference(a):
    env = dict(os.environ, AUTOEMBED_CONFIG=str(_resolve_config(a.config)), AUTOEMBED_ROOT=str(ROOT))
    inner = ["uv", "run", "--no-sync", "python", "-m", "autoembed.reference", *a.models]
    _launch(_gpu_wrap(inner, env, a), env, a)


def _resolve_model_dir(path):
    # run dir -> newest recovery snapshot holding a manifest (newest overall otherwise)
    recovery = Path(path) / "recovery"
    if not recovery.is_dir():
        return path
    snaps = sorted((d for d in recovery.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    complete = [d for d in snaps if (d / "training_manifest.json").exists()]
    chosen = (complete or snaps or [Path(path)])[0]
    print(f">> scoring snapshot: {chosen}")
    return str(chosen)


def cmd_score(a):
    env = dict(os.environ, AUTOEMBED_CONFIG=str(_resolve_config(a.config)), AUTOEMBED_ROOT=str(ROOT))
    inner = ["uv", "run", "--no-sync", "python", "-m", "autoembed.scoring", _resolve_model_dir(a.model)]
    if a.out:
        inner.append(a.out)
    _launch(_gpu_wrap(inner, env, a), env, a)


def cmd_run(a):
    if a.base and not a.base_revision:
        sys.exit("!! --base requires --base-revision so the starting checkpoint is reproducible")
    if a.base_revision and re.fullmatch(r"[0-9a-f]{40}", a.base_revision) is None:
        sys.exit("!! --base-revision must be a 40-character lowercase commit hash")
    config = _resolve_config(a.config)
    isolation = a.isolation or ("docker" if (a.local and shutil.which("docker")) else "native" if a.local else "enroot")
    env = dict(os.environ, AUTOEMBED_CONFIG=str(config), HOURS=str(a.hours), MODE=isolation)
    selected = json.loads(config.read_text())
    env["AUTOEMBED_BASE_MODEL"] = selected["base_model"]
    env["AUTOEMBED_BASE_REVISION"] = selected["base_revision"]
    if a.base:
        env["AUTOEMBED_BASE_MODEL"] = a.base
    if a.base_revision:
        env["AUTOEMBED_BASE_REVISION"] = a.base_revision
    if a.gpu is not None:
        env["GPU_ID"] = str(a.gpu)
    model = a.model or DEFAULT_MODEL.get(a.agent, "")
    run = [str(ROOT / "scripts" / "run_task.sh"), a.agent, *([model] if model else []), str(a.hours)]
    if a.local:
        cmd = run
    else:
        env.setdefault("TIME", a.time or f"{a.hours + 2:02d}:00:00")
        cmd = [str(ROOT / "scripts" / "gpu.sh"), *run]

    if a.agent == "claude" and isolation in ("enroot", "docker") \
            and not (env.get("CLAUDE_CODE_OAUTH_TOKEN") or env.get("ANTHROPIC_API_KEY")):
        print("!! warning: isolated Claude run needs CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY", file=sys.stderr)

    _launch(cmd, env, a)


def _common(sp):
    sp.add_argument("--config", required=True, help="config name (see `autoembed list`) or path")
    sp.add_argument("--local", action="store_true", help="run here (own GPU), not via Slurm")
    sp.add_argument("--gpu", type=int, help="physical GPU index (own GPU / non-Slurm)")
    sp.add_argument("--time", help="Slurm wall limit HH:MM:SS")
    sp.add_argument("--dry-run", action="store_true", help="print the command, do not launch")


def main():
    p = argparse.ArgumentParser(prog="autoembed", description="A framework for agentic embedding-model training.")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list available task configs").set_defaults(func=cmd_list)

    r = sub.add_parser("run", help="launch an agent on a task config")
    _common(r)
    r.add_argument("--agent", default="claude", choices=["claude", "codex", "antigravity"])
    r.add_argument("--model", help=f"exact agent model id (defaults: {DEFAULT_MODEL})")
    r.add_argument("--base", help="override base model (recorded in run provenance)")
    r.add_argument("--base-revision", help="immutable 40-character commit for --base")
    r.add_argument("--hours", type=int, default=10, help="agent budget in hours (default 10)")
    r.add_argument("--isolation", choices=["enroot", "docker", "native"], help="GPU/fs boundary")
    r.set_defaults(func=cmd_run)

    f = sub.add_parser("reference", help="score the config's reference ladder on its hidden split")
    _common(f)
    f.add_argument("models", nargs="*", help="pinned base/reference model ids to score (default: base plus all references)")
    f.set_defaults(func=cmd_reference)

    s = sub.add_parser("score", help="score a model dir on the config's hidden split")
    _common(s)
    s.add_argument("model", help="model path (e.g. results/<run>/recovery/<ts>)")
    s.add_argument("--out", help="write result JSON here")
    s.set_defaults(func=cmd_score)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
