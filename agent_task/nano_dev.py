# MTEB-nano: 40 MTEB(eng, v2) tasks. Nine heavy retrieval tasks use frozen
# subsamples, IMDb is sampled in memory, and non-discriminative MindSmall is omitted.
import copy
import hashlib
import json
import os
from pathlib import Path

import mteb
from mteb.abstasks.retrieval import AbsTaskRetrieval

HERE = Path(__file__).parent
MANIFEST_PATH = HERE / "nano_assets.json"


def _nano_dir():
    override = os.environ.get("AUTOEMBED_NANO_DIR")
    if override:
        return Path(override).expanduser().resolve()
    sandbox_dir = HERE / "nano"
    if sandbox_dir.is_dir():
        return sandbox_dir
    return HERE.parent / "runs" / "nano"


def asset_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text())
    files = manifest.get("files")
    if (
        manifest.get("version") != 1
        or manifest.get("benchmark") != "mteb-nano"
        or not isinstance(files, list)
    ):
        raise RuntimeError(f"invalid MTEB-nano manifest: {MANIFEST_PATH}")
    computed = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if computed != manifest.get("asset_set_sha256"):
        raise RuntimeError(f"invalid MTEB-nano asset-set checksum: {MANIFEST_PATH}")
    return manifest


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_assets(expected_revision=None):
    """Fail unless every frozen nano file is present and byte-for-byte correct."""
    manifest = asset_manifest()
    revision = manifest["asset_set_sha256"]
    if expected_revision is not None and revision != expected_revision:
        raise RuntimeError(
            f"MTEB-nano revision mismatch: config={expected_revision}, assets={revision}"
        )
    root = _nano_dir()
    problems = []
    for item in manifest["files"]:
        path = root / item["name"]
        if not path.is_file():
            problems.append(f"missing {path}")
            continue
        digest = _file_sha256(path)
        if digest != item["sha256"]:
            problems.append(
                f"checksum mismatch for {path}: expected {item['sha256']}, got {digest}"
            )
    if problems:
        detail = "\n  ".join(problems)
        raise RuntimeError(
            "MTEB-nano assets are unavailable or changed. "
            "Place the frozen bundle in runs/nano (or set AUTOEMBED_NANO_DIR).\n  "
            + detail
        )
    return revision


_RETR = {
    Path(item["name"]).stem
    for item in asset_manifest()["files"]
}


def _nano_retrieval(name):
    path = _nano_dir() / f"{name}.json"
    md = mteb.get_tasks(tasks=[name])[0].metadata.model_copy(update={
        "name": name, "eval_splits": ["test"], "adapted_from": [name],
        "dataset": {"path": str(path), "revision": "local"}})

    class _N(AbsTaskRetrieval):
        metadata = md
        _p = path

        def load_data(self, **kw):
            if self.data_loaded:
                return
            d = json.loads(self._p.read_text())
            self.corpus = {"test": {c: {"_id": c, "text": t} for c, t in d["corpus"].items()}}
            self.queries = {"test": dict(d["queries"])}
            self.relevant_docs = {"test": {q: {c: int(s) for c, s in v.items()}
                                           for q, v in d["qrels"].items()}}
            self.data_loaded = True
    _N.__name__ = f"Nano{name}"
    return _N()


def _nano_imdb(n=2000, seed=0):
    t = copy.deepcopy(mteb.get_tasks(tasks=["ImdbClassification"])[0])
    orig = t.load_data

    def load_data(**kw):
        if t.data_loaded:
            return
        orig(**kw)
        ds = t.dataset["test"]
        if not str(ds.features["label"]).startswith("ClassLabel"):
            ds = ds.class_encode_column("label")
        t.dataset["test"] = ds.train_test_split(test_size=n, seed=seed,
                                                stratify_by_column="label")["test"]
    t.load_data = load_data
    return t


def dev_tasks(expected_revision=None):
    """Return the frozen 40-task development/evaluation suite."""
    validate_assets(expected_revision)
    out = []
    for registered_task in mteb.get_benchmark("MTEB(eng, v2)").tasks:
        name = registered_task.metadata.name
        if name in _RETR:
            out.append(_nano_retrieval(name))
        elif name == "MindSmallReranking":
            # Dropped: weak discrimination in the validation study and too slow
            # at full size. AskUbuntu keeps reranking represented.
            continue
        elif name == "ImdbClassification":
            out.append(_nano_imdb())
        else:
            out.append(copy.deepcopy(registered_task))
    return out
