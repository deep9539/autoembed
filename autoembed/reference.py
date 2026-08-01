# Canonical reference ladder on the frozen hidden split.
# With no CLI args, uses config["references"]; CLI model ids use MTEB canonical loaders.
import json
import hashlib
import sys

from autoembed.scoring import CONFIG, HELDOUT_TASKS, load_pinned_model
from task import RUNS_DIR, _score


def _reference_specs():
    if sys.argv[1:]:
        configured = {
            CONFIG["base_model"]: {
                "name": CONFIG["base_model"],
                "revision": CONFIG["base_revision"],
                "loader": CONFIG.get("base_loader", "sentence-transformer"),
                "role": "base-floor",
            },
            **{spec["name"]: spec for spec in CONFIG.get("references") or []},
        }
        unknown = [name for name in sys.argv[1:] if name not in configured]
        if unknown:
            raise ValueError(
                f"unpinned reference models {unknown}; add each model, revision, "
                "and loader to the config first"
            )
        return [
            configured[name]
            for name in sys.argv[1:]
        ]
    return list(CONFIG.get("references") or [])


def main():
    config_fingerprint = hashlib.sha256(
        json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    specs = _reference_specs()
    if not sys.argv[1:]:
        specs = [
            {
                "name": CONFIG["base_model"],
                "revision": CONFIG.get("base_revision"),
                "loader": CONFIG.get("base_loader", "sentence-transformer"),
                "role": "base-floor",
            },
            *specs,
        ]
    lines, records = [], []
    for spec in specs:
        name = spec["name"]
        revision = spec.get("revision")
        try:
            model = load_pinned_model(spec)
            result = _score(model, HELDOUT_TASKS, tag="ref", trust_remote_code=True)
            tasks = "  ".join(
                f"{task}={score:.3f}"
                for task, score in sorted(result["per_task"].items())
            )
            identity = f"{name}@{revision}" if revision else name
            if spec.get("role"):
                identity += f" [{spec['role']}]"
            line = f"- mean_type={result['mean_type']:.4f}  {identity}\n    {tasks}"
            if result["skipped"]:
                line += f"\n    INVALID skipped: {result['skipped']}"
            records.append({
                "name": name, "revision": revision, "loader": spec.get("loader"),
                "role": spec.get("role"), "protocol_version": CONFIG.get("protocol_version"),
                "config_fingerprint": config_fingerprint,
                **result,
            })
        except Exception as error:
            line = f"- FAIL  {name}@{revision}: {repr(error)[:160]}"
            records.append({
                "name": name, "revision": revision, "loader": spec.get("loader"),
                "role": spec.get("role"), "protocol_version": CONFIG.get("protocol_version"),
                "config_fingerprint": config_fingerprint,
                "error": repr(error),
            })
        print(line)
        lines.append(line)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    markdown = (
        "# Canonical reference ladder — frozen held-out split\n\n"
        f"Protocol: `{CONFIG.get("protocol_version")}`  \n"
        f"Config SHA-256: `{config_fingerprint}`\n\n"
        + "\n".join(lines) + "\n"
    )
    payload = json.dumps(records, indent=2)
    (RUNS_DIR / "reference.md").write_text(markdown)
    (RUNS_DIR / "reference.json").write_text(payload)

    selection = "full" if not sys.argv[1:] else "only-" + "-".join(
        name.rsplit("/", 1)[-1] for name in sys.argv[1:]
    )
    archive = RUNS_DIR / "references"
    archive.mkdir(parents=True, exist_ok=True)
    stem = f"{CONFIG.get('protocol_version', 'unspecified')}-{config_fingerprint[:12]}-{selection}"
    (archive / f"{stem}.md").write_text(markdown)
    (archive / f"{stem}.json").write_text(payload)


if __name__ == "__main__":
    main()
