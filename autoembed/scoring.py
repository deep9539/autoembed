# Harness-only scoring on hidden held-out tasks plus protocol validation.
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Progress bars are carriage-return animations; in a redirected score log they
# become one multi-megabyte line. Must precede the datasets/mteb import.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import mteb

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("AUTOEMBED_ROOT", str(_REPO_ROOT))
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "specialization" / "legal.json"
os.environ.setdefault("AUTOEMBED_CONFIG", str(_DEFAULT_CONFIG_PATH))
sys.path.insert(0, str(_REPO_ROOT / "agent_task"))  # sandbox bundle imports

from task import (
    MODEL_DIR,
    PAIR_HASH_ALGORITHM,
    TEXT_HASH_ALGORITHM,
    TRAINING_MANIFEST,
    _EVAL_CACHE,
    _collect_hashes,
    _eval_groups,
    _eval_texts,
    _pair_h,
    _score,
    split_retrieval_queries,
    split_task_examples,
)

CONFIG_PATH = Path(os.environ["AUTOEMBED_CONFIG"])
CONFIG = json.loads(CONFIG_PATH.read_text())
_EVAL_CACHE_META = _EVAL_CACHE.with_name("_eval_texts.meta.json")
BASE_MODEL_ID = CONFIG.get("base_model")


def validate_config(config=None):
    config = CONFIG if config is None else config
    protocol_type = config.get("protocol_type", "transfer")
    if config.get("require_complete_score"):
        revision = config.get("base_revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("reportable configs require a 40-character base_revision commit")
        references = config.get("references") or []
        names = [reference.get("name") for reference in references]
        if config.get("base_model") in names:
            raise ValueError("base model must not be duplicated in references")
        if len(names) != len(set(names)):
            raise ValueError("reference model names must be unique")
        for reference in references:
            if reference.get("loader", "mteb") not in ("mteb", "sentence-transformer"):
                raise ValueError("reference loader must be mteb or sentence-transformer")
            if re.fullmatch(r"[0-9a-f]{40}", str(reference.get("revision", ""))) is None:
                raise ValueError("reportable references require 40-character revision commits")
    query_overlap_policy = config.get("query_overlap_policy", "fail")
    contamination_policy = config.get("contamination_policy", "strict")
    if contamination_policy not in ("strict", "open-data"):
        raise ValueError("unknown contamination_policy")
    if query_overlap_policy not in ("fail", "report"):
        raise ValueError("query_overlap_policy must be 'fail' or 'report'")
    if query_overlap_policy == "report":
        if contamination_policy != "open-data":
            raise ValueError("query reporting requires contamination_policy=open-data")
        if not config.get("require_source_provenance"):
            raise ValueError("open-data requires source provenance")
    if protocol_type not in ("transfer", "target-specialization", "benchmark-target"):
        raise ValueError(f"unknown protocol_type: {protocol_type}")
    query_split = config.get("query_split")
    example_split = config.get("example_split")
    if query_split and example_split:
        raise ValueError("configs cannot combine query_split and example_split")
    if example_split:
        if protocol_type != "benchmark-target":
            raise ValueError("example_split requires protocol_type=benchmark-target")
        if config.get("allow_target_corpus_training"):
            raise ValueError("benchmark-target does not permit evaluation-corpus training")
        if (
            config.get("max_incidental_overlap_hits", 0) > 10000
            or config.get("max_incidental_overlap_fraction", 0.0) > 0.005
        ):
            raise ValueError("benchmark-target caps incidental evaluation-text tolerance")
        if config.get("dev_tasks") or config.get("heldout_tasks"):
            raise ValueError(
                "benchmark-target configs use one benchmark or one shared task list"
            )
        dev_benchmark = config.get("dev_benchmark")
        heldout_benchmark = config.get("heldout_benchmark")
        if dev_benchmark == "mteb-nano":
            revision = str(config.get("benchmark_revision", ""))
            if re.fullmatch(r"[0-9a-f]{64}", revision) is None:
                raise ValueError("mteb-nano requires a 64-character benchmark_revision")
        tasks = list(config.get("tasks") or [])
        if tasks and (dev_benchmark or heldout_benchmark):
            raise ValueError(
                "benchmark-target cannot combine benchmark names and tasks"
            )
        if not tasks and (
            not dev_benchmark or dev_benchmark != heldout_benchmark
        ):
            raise ValueError(
                "benchmark-target requires a shared task list or matching benchmarks"
            )
        if tasks and len(tasks) != len(set(tasks)):
            raise ValueError("benchmark-target tasks must be unique")
        expected = config.get("expected_task_count")
        if tasks and expected is not None and len(tasks) != int(expected):
            raise ValueError(f"expected {expected} unique tasks, found {len(tasks)}")
        fraction = float(example_split.get("dev_fraction", 0))
        if not 0 < fraction < 1 or not str(example_split.get("seed", "")):
            raise ValueError(
                "example_split requires 0 < dev_fraction < 1 and a seed"
            )
        return True
    if query_split:
        tasks = list(config.get("tasks") or [])
        if protocol_type != "target-specialization":
            raise ValueError("query_split requires protocol_type=target-specialization")
        if not config.get("allow_target_corpus_training"):
            raise ValueError("query_split requires allow_target_corpus_training=true")
        if config.get("dev_tasks") or config.get("heldout_tasks"):
            raise ValueError("query_split configs use tasks, not dev_tasks/heldout_tasks")
        if not tasks or len(tasks) != len(set(tasks)):
            raise ValueError("query_split configs require unique tasks")
        fraction = float(query_split.get("dev_fraction", 0))
        if not 0 < fraction < 1 or not str(query_split.get("seed", "")):
            raise ValueError("query_split requires 0 < dev_fraction < 1 and a seed")
        expected = config.get("expected_task_count")
        if expected is not None and len(tasks) != int(expected):
            raise ValueError(f"expected {expected} unique tasks, found {len(tasks)}")
        return True
    if config.get("allow_target_corpus_training"):
        raise ValueError("allow_target_corpus_training requires query_split")
    dev = list(config.get("dev_tasks") or [])
    heldout = list(config.get("heldout_tasks") or [])
    if not dev or not heldout:
        raise ValueError("reportable configs require non-empty dev_tasks and heldout_tasks")
    if len(dev) != len(set(dev)) or len(heldout) != len(set(heldout)):
        raise ValueError("duplicate task names are not allowed within a split")
    overlap = set(dev) & set(heldout)
    if overlap:
        raise ValueError(f"dev/heldout task overlap: {sorted(overlap)}")

    assigned = set(dev) | set(heldout)
    expected = config.get("expected_task_count")
    if expected is not None and len(assigned) != int(expected):
        raise ValueError(
            f"expected {expected} unique tasks, found {len(assigned)}"
        )
    for group in config.get("linked_task_groups", []):
        missing = set(group) - assigned
        if missing:
            raise ValueError(f"linked task group contains unassigned tasks: {sorted(missing)}")
        placements = {
            "dev" if task in dev else "heldout"
            for task in group
        }
        if len(placements) != 1:
            raise ValueError(f"linked task group crosses dev/heldout: {group}")
    return True


validate_config()


def _benchmark_tasks(name):
    if name == "mteb-nano":
        import nano_dev

        return nano_dev.dev_tasks(CONFIG.get("benchmark_revision"))
    return mteb.get_benchmark(name).tasks


# Held-out grade set: explicit tasks > benchmark > MTEB-nano default.
HELDOUT_BENCHMARK = CONFIG.get("heldout_benchmark", "mteb-nano")
if CONFIG.get("example_split"):
    target_tasks = (
        mteb.get_tasks(tasks=CONFIG["tasks"])
        if CONFIG.get("tasks")
        else _benchmark_tasks(HELDOUT_BENCHMARK)
    )
    HELDOUT_TASKS = split_task_examples(
        target_tasks,
        "heldout",
        CONFIG["example_split"],
    )
    expected = CONFIG.get("expected_task_count")
    if expected is not None and len(HELDOUT_TASKS) != int(expected):
        raise ValueError(
            f"expected {expected} benchmark tasks, found {len(HELDOUT_TASKS)}"
        )
elif CONFIG.get("query_split"):
    HELDOUT_TASKS = split_retrieval_queries(
        mteb.get_tasks(tasks=CONFIG["tasks"]),
        "heldout",
        CONFIG["query_split"],
    )
elif CONFIG.get("heldout_tasks"):
    HELDOUT_TASKS = mteb.get_tasks(tasks=CONFIG["heldout_tasks"])
elif HELDOUT_BENCHMARK == "mteb-nano":
    HELDOUT_TASKS = _benchmark_tasks("mteb-nano")
else:
    HELDOUT_TASKS = _benchmark_tasks(HELDOUT_BENCHMARK)


def _config_fingerprint():
    payload = json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _collect_dataset_hashes(value, out):
    # DatasetDict is mapping-like but also exposes column_names; recurse into
    # its splits before handling a single row-iterable Dataset.
    if isinstance(value, dict):
        for nested in value.values():
            _collect_dataset_hashes(nested, out)
    elif hasattr(value, "column_names") and hasattr(value, "__iter__"):
        for row in value:
            _collect_hashes(row, out)


def _retrieval_splits(task):
    # mteb 2.x retrieval container: dataset[subset][split] = {corpus, queries, relevant_docs, ...}
    dataset = getattr(task, "dataset", None)
    if not isinstance(dataset, dict):
        return
    for splits in dataset.values():
        if not isinstance(splits, dict):
            continue
        for data in splits.values():
            if isinstance(data, dict) and ("corpus" in data or "queries" in data):
                yield data


def _collect_split_hashes(data, groups):
    queries = data.get("queries")
    query_hashes = {}
    if queries is not None:
        for row in queries:
            row = dict(row)
            hashes = set()
            _collect_hashes({k: v for k, v in row.items() if k != "id"}, hashes)
            query_hashes[row.get("id")] = hashes
            groups["queries"].update(hashes)
    doc_hashes = {}
    corpus = data.get("corpus")
    if corpus is not None:
        for row in corpus:
            row = dict(row)
            hashes = set()
            _collect_hashes({k: v for k, v in row.items() if k != "id"}, hashes)
            doc_hashes[row.get("id")] = hashes
    relevant = set()
    qrels = data.get("relevant_docs")
    if isinstance(qrels, dict):
        for query_id, docs in qrels.items():
            if not isinstance(docs, dict):
                continue
            for doc_id, relevance in docs.items():
                if relevance <= 0:
                    continue
                document_values = doc_hashes.get(doc_id, set())
                relevant.update(document_values)
                groups.setdefault("query_relevant_pairs", set()).update(
                    _pair_h(query_hash, document_hash)
                    for query_hash in query_hashes.get(query_id, set())
                    for document_hash in document_values
                    if query_hash != document_hash
                )
    groups["relevant"].update(relevant)
    if doc_hashes:
        groups["other_corpus"].update(set().union(*doc_hashes.values()) - relevant)


def build_eval_cache():
    """Build hidden text hashes grouped by their evaluation role."""
    groups = {
        name: set()
        for name in ("queries", "relevant", "other_corpus", "protected", "query_relevant_pairs")
    }
    for task in HELDOUT_TASKS:
        task.load_data()
        queries = getattr(task, "queries", None)
        corpus = getattr(task, "corpus", None)
        qrels = getattr(task, "relevant_docs", None)
        containers = [] if (queries or corpus) else list(_retrieval_splits(task))
        for data in containers:
            _collect_split_hashes(data, groups)
        _collect_hashes(queries, groups["queries"])
        corpus_hashes = set()
        _collect_hashes(corpus, corpus_hashes)

        if isinstance(corpus, dict) and isinstance(qrels, dict):
            for split, split_qrels in qrels.items():
                split_corpus = corpus.get(split)
                split_queries = queries.get(split) if isinstance(queries, dict) else None
                if not (
                    isinstance(split_corpus, dict)
                    and isinstance(split_queries, dict)
                    and isinstance(split_qrels, dict)
                ):
                    continue
                for query_id, docs in split_qrels.items():
                    if not isinstance(docs, dict):
                        continue
                    query_values = set()
                    _collect_hashes(split_queries.get(query_id), query_values)
                    for doc_id, relevance in docs.items():
                        if relevance <= 0 or doc_id not in split_corpus:
                            continue
                        document_values = set()
                        _collect_hashes(split_corpus[doc_id], document_values)
                        groups["relevant"].update(document_values)
                        groups.setdefault("query_relevant_pairs", set()).update(
                            _pair_h(query_hash, document_hash)
                            for query_hash in query_values
                            for document_hash in document_values
                            if query_hash != document_hash
                        )

        groups["other_corpus"].update(corpus_hashes - groups["relevant"])
        if not queries and not corpus and not containers:
            _collect_dataset_hashes(
                getattr(task, "dataset", None), groups["protected"]
            )

    text_hashes = set().union(*(
        groups[name] for name in ("queries", "relevant", "other_corpus", "protected")
    ))
    cache_tmp = _EVAL_CACHE.with_name("_eval_texts.tmp.json")
    meta_tmp = _EVAL_CACHE_META.with_name("_eval_texts.meta.tmp.json")
    cache_tmp.write_text(json.dumps({
        name: sorted(values) for name, values in groups.items()
    }))
    meta_tmp.write_text(json.dumps({
        "version": 3,
        "hash": TEXT_HASH_ALGORITHM,
        "pair_hash": PAIR_HASH_ALGORITHM,
        "schema": "evaluation-roles-and-pairs-v4",
        "config_fingerprint": _config_fingerprint(),
        "heldout_tasks": [task.metadata.name for task in HELDOUT_TASKS],
        "counts": {name: len(values) for name, values in groups.items()},
        "count": len(text_hashes),
        "pair_count": len(groups["query_relevant_pairs"]),
    }, indent=2))
    cache_tmp.replace(_EVAL_CACHE)
    meta_tmp.replace(_EVAL_CACHE_META)
    return len(text_hashes)


def write_agent_eval_cache(dest):
    """Ship hashes of public corpus text so agents can self-filter training data.

    Corpus documents are publicly downloadable, so their hashes reveal nothing;
    hidden query and protected example hashes stay with the harness.
    """
    ensure_eval_cache()
    payload = json.loads(_EVAL_CACHE.read_text())
    public = {
        name: payload.get(name, [])
        for name in ("relevant", "other_corpus")
    }
    Path(dest).write_text(json.dumps(public))
    return sum(len(hashes) for hashes in public.values())


def ensure_eval_cache():
    try:
        meta = json.loads(_EVAL_CACHE_META.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        meta = {}
    if (
        not _EVAL_CACHE.exists()
        or meta.get("config_fingerprint") != _config_fingerprint()
        or meta.get("hash") != TEXT_HASH_ALGORITHM
        or meta.get("pair_hash") != PAIR_HASH_ALGORITHM
        or meta.get("schema") != "evaluation-roles-and-pairs-v4"
    ):
        return build_eval_cache()
    return int(meta.get("count", 0))


def _eval_pair_hashes():
    try:
        payload = json.loads(_EVAL_CACHE.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, dict):
        return set()
    pairs = payload.get("query_relevant_pairs")
    return set(pairs) if isinstance(pairs, list) else set()


def audit_contamination(model_path):
    manifest_path = Path(model_path) / TRAINING_MANIFEST
    if not manifest_path.is_file():
        return {"present": False, "manifest": str(manifest_path), "hits": None}
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {
            "present": True, "manifest": str(manifest_path),
            "valid": False, "error": repr(error), "hits": None,
        }

    hashes = manifest.get("hashes")
    version = manifest.get("version")
    pair_hashes = manifest.get("query_document_pair_hashes")
    pair_schema_valid = version == 1 or (
        version == 2
        and manifest.get("pair_hash") == PAIR_HASH_ALGORITHM
        and isinstance(pair_hashes, list)
        and isinstance(manifest.get("sources", []), list)
    )
    structurally_valid = (
        version in (1, 2)
        and manifest.get("hash") == TEXT_HASH_ALGORITHM
        and pair_schema_valid
        and isinstance(hashes, list)
        and manifest.get("exhaustive") is True
        and manifest.get("checked_rows") == manifest.get("dataset_rows")
    )
    train_hashes = set(hashes or [])
    train_pair_hashes = set(pair_hashes or []) if version == 2 else set()
    eval_groups = _eval_groups()
    by_role = {
        name: train_hashes & eval_hashes
        for name, eval_hashes in eval_groups.items()
    }
    pair_hits = train_pair_hashes & _eval_pair_hashes()
    harmful = set().union(*(
        by_role.get(name, set())
        for name in ("relevant", "protected")
    ))
    attributed = harmful | by_role.get("queries", set())
    incidental = by_role.get("other_corpus", set()) - attributed
    hits = len(set().union(*by_role.values())) if by_role else 0
    return {
        "present": True,
        "manifest": str(manifest_path),
        "manifest_version": version,
        "valid": structurally_valid,
        "pair_audit_available": version == 2,
        "sources": manifest.get("sources", []),
        "exhaustive": manifest.get("exhaustive"),
        "dataset_rows": manifest.get("dataset_rows"),
        "checked_rows": manifest.get("checked_rows"),
        "checked_strings": manifest.get("checked_strings"),
        "unique_train_texts": len(hashes or []),
        "hits": hits,
        "frac": round(hits / max(len(hashes or []), 1), 8),
        "harmful_hits": len(harmful),
        "train_hashes": train_hashes,   # stripped before the result is written
        "train_pair_hashes": train_pair_hashes,  # stripped before writing
        "query_hits": len(by_role.get("queries", set())),
        "hidden_query_texts": len(eval_groups.get("queries", set())),
        "query_exposure_fraction": round(
            len(by_role.get("queries", set())) / max(len(eval_groups.get("queries", set())), 1), 8
        ),
        "query_relevant_pair_hits": len(pair_hits),
        "relevant_hits": len(by_role.get("relevant", set())),
        "protected_hits": len(by_role.get("protected", set())),
        "incidental_hits": len(incidental),
        "incidental_frac": round(len(incidental) / max(len(hashes or []), 1), 8),
    }


def load_pinned_model(spec):
    """Load a comparison encoder at its immutable revision."""
    name = spec["name"]
    if spec.get("loader", "mteb") == "mteb":
        return mteb.get_model(name, revision=spec.get("revision"), device="cuda")
    if spec.get("loader") == "sentence-transformer":
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            name, revision=spec.get("revision"), trust_remote_code=True, device="cuda"
        )
    raise ValueError(f"unknown reference loader: {spec.get('loader')}")


def contamination_failures(contamination, config=None):
    config = CONFIG if config is None else config
    failures = []
    if config.get("require_contamination_manifest"):
        if not contamination.get("present"):
            failures.append("missing artifact training_manifest.json")
        elif not contamination.get("valid"):
            failures.append("training manifest is not exhaustive or structurally valid")
    if (
        config.get("require_source_provenance")
        and contamination.get("valid")
        and contamination.get("dataset_rows", 0) > 0
        and not contamination.get("sources")
    ):
        failures.append("training manifest is missing source provenance")
    if config.get("require_zero_contamination"):
        if (contamination.get("hits") not in (None, 0)
                or contamination.get("query_relevant_pair_hits", 0)):
            failures.append(
                f"training/evaluation text overlap: {contamination.get('hits')} hashes"
            )
        return failures

    # A retrieval score is only invalidated by supervision that links a hidden
    # query to its relevant document. Query text and non-retrieval task text
    # collide with public corpora by construction — MTEB tasks are built from the
    # same datasets an embedder trains on — so those are measured and reported by
    # contamination_warnings rather than gated here.
    if contamination.get("query_relevant_pair_hits", 0):
        failures.append(
            "hidden query and relevant document co-occur in training rows: "
            f"{contamination.get('query_relevant_pair_hits')} pairs"
        )
    if config.get("allow_target_corpus_training"):
        # The corpus is already in hand here, so a hidden query completes the pair.
        if contamination.get("query_hits", 0):
            failures.append(
                "training overlap with hidden queries: "
                f"{contamination.get('query_hits')} hashes"
            )
    else:
        # Gold documents and non-retrieval task text are ordinary public text
        # (Wikipedia, PubMed, abstracts), so a lone hit is incidental; only
        # wholesale ingestion of the evaluation text is a flag. Both a count floor
        # and a fraction must be exceeded, so a small training set is not
        # condemned by a single collision.
        eval_hits = (
            contamination.get("relevant_hits", 0)
            + contamination.get("incidental_hits", 0)
            + contamination.get("protected_hits", 0)
        )
        unique = contamination.get("unique_train_texts") or 0
        eval_frac = eval_hits / unique if unique else 0.0
        if (eval_hits > config.get("max_incidental_overlap_hits", 10000)
                and eval_frac > config.get("max_incidental_overlap_fraction", 0.001)):
            failures.append(
                "evaluation-text ingestion: "
                f"{eval_hits} hashes ({eval_frac:.4f} of training text)"
            )
    return failures


def contamination_warnings(contamination, config=None):
    config = CONFIG if config is None else config
    warnings = []
    query_hits = contamination.get("query_hits", 0)
    if query_hits and config.get("query_overlap_policy", "fail") == "report":
        fraction = contamination.get("query_exposure_fraction", 0.0)
        warnings.append(
            f"hidden-query exposure: {query_hits} hashes ({fraction:.2%} of hidden query texts); "
            "the fixed full test set was scored"
        )
        if contamination.get("pair_audit_available"):
            if not contamination.get("query_relevant_pair_hits", 0):
                warnings.append("no exact hidden query/relevant-document row pair was detected")
        else:
            warnings.append(
                "legacy manifest has no row-pair evidence; query/relevant co-occurrence is unknown"
            )
    incidental_hits = contamination.get("incidental_hits", 0)
    if incidental_hits:
        warnings.append(
            "incidental evaluation-corpus overlap within configured limits: "
            f"{incidental_hits} hashes"
        )
    if contamination.get("manifest_version") == 2 and not contamination.get("sources"):
        warnings.append("training source provenance was not recorded in the manifest")
    return warnings


def contamination_reportability(contamination, failures):
    if failures:
        return "invalid"
    if contamination.get("query_hits", 0):
        return "reportable_with_query_exposure"
    if contamination.get("incidental_hits", 0):
        return "reportable_with_incidental_overlap"
    return "clean"


def _write_result(path, result):
    contam = result.get("contamination")
    if isinstance(contam, dict):
        private = {"train_hashes", "train_pair_hashes"}
        contam = {k: v for k, v in contam.items() if k not in private}
        result = {**result, "contamination": contam}
    if path:
        Path(path).write_text(json.dumps(result, indent=2))


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else str(MODEL_DIR)
    output = sys.argv[2] if len(sys.argv) > 2 else None
    ensure_eval_cache()

    contamination = audit_contamination(model)
    warnings = contamination_warnings(contamination)
    failures = contamination_failures(contamination)
    custom_entrypoint = Path(model) / "mteb_model.py"
    if custom_entrypoint.is_file() and not os.environ.get("AUTOEMBED_ENCODER_WORKER_COMMAND"):
        failures.append(
            "custom mteb_model.py requires an isolated hidden-scoring worker"
        )
    if failures:
        # A flagged submission reports the base model's score rather than being
        # discarded: the run keeps its place in the results table and the flag
        # costs the claimed gain, not the observation.
        for heldout_task in HELDOUT_TASKS:
            heldout_task.load_data()
        base = _score(
            load_pinned_model({
                "name": BASE_MODEL_ID,
                "revision": CONFIG.get("base_revision"),
                "loader": CONFIG.get("base_loader", "sentence-transformer"),
            }),
            HELDOUT_TASKS, tag="base-substituted", trust_remote_code=True,
        )
        result = {
            "score_schema_version": 2,
            "contamination_policy": CONFIG.get("contamination_policy", "strict"),
            "protocol_valid": False, "invalid_reasons": failures,
            "reportability": "flagged-base-substituted",
            "warnings": warnings,
            "mean_type": base["mean_type"], "mean_task": base["mean_task"],
            "per_type": base["per_type"], "heldout_per_task": base["per_task"],
            "skipped": base["skipped"],
            "substituted_base_model": BASE_MODEL_ID,
            "contamination": contamination,
        }
        _write_result(output, result)
        print(f"FLAGGED, base score substituted: {failures}; "
              f"base mean_type={base['mean_type']:.4f}")
        return 2

    for task in HELDOUT_TASKS:
        task.load_data()

    scored = _score(model, HELDOUT_TASKS, tag="heldout")
    if CONFIG.get("require_complete_score") and scored["skipped"]:
        failures.append(f"held-out tasks skipped: {scored['skipped']}")
    result = {
        "score_schema_version": 2,
        "contamination_policy": CONFIG.get("contamination_policy", "strict"),
        "protocol_valid": not failures,
        "reportability": contamination_reportability(contamination, failures),
        "warnings": warnings,
        "excluded_contaminated_queries": {},
        "invalid_reasons": failures,
        "mean_type": scored["mean_type"],
        "mean_task": scored["mean_task"],
        "per_type": scored["per_type"],
        "heldout_per_task": scored["per_task"],
        "skipped": scored["skipped"],
        "contamination": contamination,
    }
    _write_result(output, result)
    print(
        f"VALID={result['protocol_valid']}  STATUS={result['reportability']}  "
        f"MEAN_TYPE={scored['mean_type']:.4f}  "
        f"MEAN_TASK={scored['mean_task']:.4f}  skipped={len(scored['skipped'])}  "
        f"contam={contamination}"
    )
    return 0 if result["protocol_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
