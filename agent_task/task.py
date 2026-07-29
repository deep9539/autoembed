# Agent-facing API: fixed base model, dev evaluation, contamination check.
import atexit
import base64
import hashlib
import importlib.util
import io
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AUTOEMBED_ROOT") or Path(__file__).parent)  # sandbox: own dir; harness sets repo root
RUNS_DIR = ROOT / "runs"
MODEL_DIR = ROOT / "final_model"  # the agent's submitted model
_EVAL_CACHE = ROOT / "_eval_texts.json"  # hashed held-out text set

_config_path = os.environ.get("AUTOEMBED_CONFIG")
_config = json.loads(Path(_config_path).read_text()) if _config_path else {}
_base_override = os.environ.get("AUTOEMBED_BASE_MODEL")
BASE_MODEL = _base_override or _config.get("base_model")
BASE_REVISION = os.environ.get("AUTOEMBED_BASE_REVISION")
if BASE_REVISION is None and _base_override is None:
    BASE_REVISION = _config.get("base_revision")
if BASE_REVISION is None:
    raise RuntimeError(
        "BASE_REVISION is undefined; set AUTOEMBED_BASE_REVISION or use a pinned config"
    )
if BASE_MODEL is None:
    raise RuntimeError(
        "BASE_MODEL is undefined; set AUTOEMBED_BASE_MODEL or AUTOEMBED_CONFIG"
    )
# Dev task list provided by the harness; unset -> the built-in default dev suite.
_dev_env = os.environ.get("AUTOEMBED_DEV_TASKS", "").strip()
DEV_TASKS = [t for t in _dev_env.split(",") if t] or None
_query_split_env = os.environ.get("AUTOEMBED_QUERY_SPLIT", "").strip()
QUERY_SPLIT = json.loads(_query_split_env) if _query_split_env else None
_example_split_env = os.environ.get("AUTOEMBED_EXAMPLE_SPLIT", "").strip()
EXAMPLE_SPLIT = json.loads(_example_split_env) if _example_split_env else None
MAX_SEQ = int(os.environ.get("AUTOEMBED_MAX_SEQ", "0"))  # 0 = model's native length; >0 caps
PER_TASK_TIMEOUT = int(os.environ.get("AUTOEMBED_PER_TASK_TIMEOUT", "3600"))
MTEB_ENTRYPOINT = "mteb_model.py"
TRAINING_MANIFEST = "training_manifest.json"
TEXT_HASH_ALGORITHM = "sha256-normalized-whitespace-v1"
PAIR_HASH_ALGORITHM = "sha256-unordered-text-hash-pair-v1"
_QUERY_COLUMNS = ("anchor", "query", "question")
_DOCUMENT_COLUMNS = (
    "positive", "answer", "document", "documents", "passage", "text", "title",
)


class _Timeout(Exception):
    pass


def _on_timeout(signum, frame):
    raise _Timeout()


def _submission_meta(model_path):
    from mteb.models import ModelMeta

    return ModelMeta.create_empty(overwrites={
        "name": f"autoembed/{Path(model_path).name}",
        "revision": "local",
    })


def _as_mteb_encoder(model, model_path):
    """Validate/adapt a custom submission to the dense MTEB EncoderProtocol."""
    import mteb
    from mteb.models.abs_encoder import AbsEncoder
    from sentence_transformers import SentenceTransformer

    if isinstance(model, mteb.SearchProtocol) or isinstance(model, mteb.CrossEncoderProtocol):
        raise TypeError(
            "AutoEmbed accepts dense MTEB encoders, not search backends or cross-encoders"
        )
    if isinstance(model, SentenceTransformer):
        from mteb.models import SentenceTransformerEncoderWrapper

        model = SentenceTransformerEncoderWrapper(model)
    elif not callable(getattr(model, "encode", None)):
        raise TypeError(
            f"{MTEB_ENTRYPOINT}: load_model() must return an MTEB EncoderProtocol "
            "or an object with an MTEB-compatible encode() method"
        )

    meta = getattr(model, "mteb_model_meta", None) or _submission_meta(model_path)
    model_types = getattr(meta, "model_type", None) or []
    if any(model_type != "dense" for model_type in model_types):
        raise TypeError(
            f"AutoEmbed accepts dense encoders only; received model_type={model_types}"
        )

    class _ValidatedEncoder(AbsEncoder):
        """Complete and validate the dense encoder contract."""

        def __init__(self, wrapped):
            self.model = wrapped
            self.mteb_model_meta = meta
            embed_dim = getattr(meta, "embed_dim", None)
            self._embedding_dim = embed_dim if isinstance(embed_dim, int) else None

        def encode(
            self, inputs, *, task_metadata, hf_split, hf_subset,
            prompt_type=None, **kwargs,
        ):
            embeddings = self.model.encode(
                inputs,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=prompt_type,
                **kwargs,
            )
            shape = getattr(embeddings, "shape", None)
            if shape is None:
                import numpy as np

                embeddings = np.asarray(embeddings)
                shape = embeddings.shape
            if len(shape) != 2:
                raise ValueError(
                    f"EncoderProtocol submission must return one dense [N, D] array; got {shape}"
                )
            dimension = int(shape[1])
            if dimension < 1:
                raise ValueError(f"Embedding dimension must be positive; got {dimension}")
            if self._embedding_dim is None:
                self._embedding_dim = dimension
            elif dimension != self._embedding_dim:
                raise ValueError(
                    f"Embedding dimension changed from {self._embedding_dim} to {dimension}"
                )
            return embeddings

        def similarity(self, embeddings1, embeddings2):
            similarity = getattr(self.model, "similarity", None)
            if callable(similarity):
                return similarity(embeddings1, embeddings2)
            return super().similarity(embeddings1, embeddings2)

        def similarity_pairwise(self, embeddings1, embeddings2):
            similarity = getattr(self.model, "similarity_pairwise", None)
            if callable(similarity):
                return similarity(embeddings1, embeddings2)
            return super().similarity_pairwise(embeddings1, embeddings2)

    adapted = _ValidatedEncoder(model)
    if not isinstance(adapted, mteb.EncoderProtocol):
        raise TypeError("Internal error: custom model could not be adapted to EncoderProtocol")
    return adapted


def _load_custom_encoder(model_path):
    """Load final_model/mteb_model.py:load_model(model_path)."""
    entrypoint = Path(model_path).resolve() / MTEB_ENTRYPOINT
    module_name = f"_autoembed_submission_{hashlib.sha256(str(entrypoint).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import custom MTEB entrypoint: {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(entrypoint.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    factory = getattr(module, "load_model", None)
    if not callable(factory):
        raise AttributeError(
            f"{entrypoint} must define load_model(model_path)"
        )
    return _as_mteb_encoder(factory(str(entrypoint.parent)), entrypoint.parent)


def _json_value(value):
    """Convert MTEB inputs/metadata to the worker's small wire format."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump())
    if hasattr(value, "value"):
        return _json_value(value.value)
    if hasattr(value, "__dict__"):
        return {
            key: _json_value(item) for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _npy_payload(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _WorkerEncoder:
    """Dense encoder proxy; submission code runs in a restricted process."""

    def __init__(self, model_path, command):
        from mteb.models.abs_encoder import AbsEncoder

        class Proxy(AbsEncoder):
            def __init__(proxy_self, owner):
                proxy_self.owner = owner
                proxy_self.mteb_model_meta = _submission_meta(model_path)

            def encode(
                proxy_self, inputs, *, task_metadata, hf_split, hf_subset,
                prompt_type=None, **kwargs,
            ):
                return proxy_self.owner.encode(
                    inputs, task_metadata=task_metadata, hf_split=hf_split,
                    hf_subset=hf_subset, prompt_type=prompt_type, **kwargs,
                )

            def similarity(proxy_self, embeddings1, embeddings2):
                return proxy_self.owner.array_operation(
                    "similarity", embeddings1=embeddings1, embeddings2=embeddings2
                )

            def similarity_pairwise(proxy_self, embeddings1, embeddings2):
                return proxy_self.owner.array_operation(
                    "similarity_pairwise", embeddings1=embeddings1,
                    embeddings2=embeddings2,
                )

        self.process = subprocess.Popen(
            shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        atexit.register(self.close)
        self.encoder = Proxy(self)

    def encode(self, inputs, **kwargs):
        request = {
            "op": "encode",
            # MTEB supplies a DataLoader. Materialization preserves its batches
            # without exposing datasets, qrels, or scorer state to the worker.
            "inputs": _json_value(list(inputs)),
            **{key: _json_value(value) for key, value in kwargs.items()},
        }
        return self._rpc(request)

    def array_operation(self, operation, **values):
        return self._rpc({
            "op": operation,
            "arrays": {key: _npy_payload(value) for key, value in values.items()},
        })

    def _rpc(self, request):
        import numpy as np

        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(f"isolated encoder worker exited unexpectedly ({code})")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"isolated encoder error: {response.get('error')}")
        payload = base64.b64decode(response["npy"])
        return np.load(io.BytesIO(payload), allow_pickle=False)

    def close(self):
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            process.stdin.write('{"op":"close"}\n')
            process.stdin.flush()
            process.wait(timeout=5)
        except Exception:
            process.terminate()


def _load_isolated_encoder(model_path, command):
    return _WorkerEncoder(model_path, command).encoder


def load_encoder(model_path, trust_remote_code=False):
    """Load a custom/local encoder, canonical in-memory MTEB model, or SentenceTransformer."""
    if not isinstance(model_path, (str, os.PathLike)):
        if callable(getattr(model_path, "encode", None)):
            return model_path
        raise TypeError("model must be a path/model id or an MTEB-compatible encoder")
    path = Path(model_path)
    if path.is_dir() and (path / MTEB_ENTRYPOINT).is_file():
        worker_command = os.environ.get("AUTOEMBED_ENCODER_WORKER_COMMAND", "").strip()
        if worker_command:
            model = _load_isolated_encoder(path, worker_command)
        elif os.environ.get("AUTOEMBED_REQUIRE_ISOLATED_CUSTOM") == "1":
            raise RuntimeError("custom mteb_model.py requires an isolated encoder worker")
        else:
            model = _load_custom_encoder(path)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(model_path), trust_remote_code=trust_remote_code)
    if MAX_SEQ:
        # Custom encoders may expose this setting directly or on a wrapped model.
        target = model
        if not hasattr(target, "max_seq_length") and hasattr(target, "model"):
            target = target.model
        if hasattr(target, "max_seq_length"):
            native = getattr(target, "max_seq_length")
            target.max_seq_length = min(native or MAX_SEQ, MAX_SEQ)
    return model


def _score(model_path, tasks, tag, trust_remote_code=False):
    # Mean over tasks + per-type + per-task.
    import mteb
    type_of = {t.metadata.name: t.metadata.type for t in tasks}
    model = load_encoder(model_path, trust_remote_code=trust_remote_code)
    per_task, skipped = {}, []
    signal.signal(signal.SIGALRM, _on_timeout)
    configured_output = os.environ.get("AUTOEMBED_MTEB_OUTPUT")
    output_folder = (
        Path(configured_output) / tag
        if configured_output
        else RUNS_DIR / "mteb" / f"{tag}-{os.getpid()}"
    )
    for t in tasks:
        name = t.metadata.name
        for bs in (64, 16, 4):   # OOM-guard: retry at smaller batch
            try:
                signal.alarm(PER_TASK_TIMEOUT)
                res = mteb.MTEB(tasks=[t]).run(
                    model, output_folder=str(output_folder),
                    verbosity=0, overwrite_results=True, encode_kwargs={"batch_size": bs})
                per_task[name] = float(res[0].get_score())
                break
            except _Timeout:
                skipped.append(name); print(f"  !! skipped {name}: timeout"); break
            except Exception as e:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if "out of memory" in str(e).lower() and bs > 4:
                    print(f"  .. {name} OOM at batch {bs}; retrying smaller"); continue
                skipped.append(name); print(f"  !! skipped {name}: {repr(e)[:100]}"); break
            finally:
                signal.alarm(0)
    if skipped:
        print(f"  !! WARNING: {len(skipped)}/{len(tasks)} tasks skipped — mean is over {len(per_task)} tasks ONLY, not comparable")
    per_type = {}
    for name, sc in per_task.items():
        per_type.setdefault(type_of[name], []).append(sc)
    type_means = {ty: sum(v) / len(v) for ty, v in per_type.items()}
    mean_type = sum(type_means.values()) / len(type_means) if type_means else 0.0
    mean_task = sum(per_task.values()) / len(per_task) if per_task else 0.0
    return {"mean_type": mean_type, "mean_task": mean_task,
            "per_type": type_means, "per_task": per_task, "skipped": skipped}


def _split_loaded_retrieval_queries(task, partition, spec):
    """Split one loaded retrieval task in place."""
    marker = (partition, float(spec["dev_fraction"]), str(spec["seed"]))
    if getattr(task, "_autoembed_query_partition", None) == marker:
        return
    name = task.metadata.name
    fraction, seed = marker[1], marker[2]
    for hf_split, split_queries in task.queries.items():
        if not isinstance(split_queries, dict) or len(split_queries) < 2:
            raise ValueError(f"{name}/{hf_split} needs at least two queries")
        ranked = sorted(
            split_queries,
            key=lambda query_id: hashlib.sha256(
                f"{seed}\0{name}\0{hf_split}\0{query_id}".encode()
            ).digest(),
        )
        dev_count = min(max(round(len(ranked) * fraction), 1), len(ranked) - 1)
        dev_ids = set(ranked[:dev_count])
        keep = dev_ids if partition == "dev" else set(ranked) - dev_ids
        task.queries[hf_split] = {
            query_id: value for query_id, value in split_queries.items()
            if query_id in keep
        }
        task.relevant_docs[hf_split] = {
            query_id: value
            for query_id, value in task.relevant_docs[hf_split].items()
            if query_id in keep
        }
        top_ranked = getattr(task, "top_ranked", None)
        if isinstance(top_ranked, dict) and isinstance(top_ranked.get(hf_split), dict):
            top_ranked[hf_split] = {
                query_id: value
                for query_id, value in top_ranked[hf_split].items()
                if query_id in keep
            }
    task._autoembed_query_partition = marker


def split_retrieval_queries(tasks, partition, spec, task_names=None):
    """Attach a deterministic query partition while keeping the corpus shared."""
    if partition not in ("dev", "heldout"):
        raise ValueError("query partition must be dev or heldout")
    selected_names = set(task_names or [task.metadata.name for task in tasks])
    for task in tasks:
        name = task.metadata.name
        if name not in selected_names:
            continue
        if task.metadata.type != "Retrieval":
            raise TypeError(f"query splitting requires Retrieval tasks; {name} is {task.metadata.type}")
        queries = getattr(task, "queries", None)
        if isinstance(queries, dict) and queries:
            _split_loaded_retrieval_queries(task, partition, spec)
            continue
        original_load_data = task.load_data

        def load_partitioned(*args, _task=task, _load=original_load_data, **kwargs):
            _load(*args, **kwargs)
            _split_loaded_retrieval_queries(_task, partition, spec)

        task.load_data = load_partitioned
    return tasks


def _ranked_partition(keys, partition, spec, namespace):
    """Return deterministic complementary key sets for one evaluation unit."""
    if len(keys) < 2:
        raise ValueError(f"{namespace} needs at least two examples")
    fraction = float(spec["dev_fraction"])
    seed = str(spec["seed"])
    ranked = sorted(
        keys,
        key=lambda key: hashlib.sha256(
            f"{seed}\0{namespace}\0{key}".encode()
        ).digest(),
    )
    dev_count = min(max(round(len(ranked) * fraction), 1), len(ranked) - 1)
    dev = set(ranked[:dev_count])
    return dev if partition == "dev" else set(ranked) - dev


def _dataset_partition_indices(dataset, partition, spec, namespace, label_column=None):
    """Split rows deterministically, stratifying discrete labels when possible."""
    rows = list(dataset)
    if len(rows) < 2:
        raise ValueError(f"{namespace} needs at least two examples")

    def row_key(index):
        row = rows[index]
        if isinstance(row, dict) and row.get("id") is not None:
            return f"id:{row['id']}"
        payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
        return f"row:{hashlib.sha256(payload.encode()).hexdigest()}"

    groups = {}
    if label_column and all(
        isinstance(row, dict) and label_column in row for row in rows
    ):
        for index, row in enumerate(rows):
            groups.setdefault(str(row[label_column]), []).append(index)
    if not groups or any(len(indices) < 2 for indices in groups.values()):
        groups = {"all": list(range(len(rows)))}
    elif any(
        len({row_key(index) for index in indices}) < 2
        for indices in groups.values()
    ):
        # A rare label made entirely of duplicate rows cannot appear on both
        # sides without text leakage; keep duplicates together and split globally.
        groups = {"all": list(range(len(rows)))}

    fraction = float(spec["dev_fraction"])
    target = min(max(round(len(rows) * fraction), 1), len(rows) - 1)
    quotas = {
        label: min(max(int(len(indices) * fraction), 1), len(indices) - 1)
        for label, indices in groups.items()
    }
    # Allocate rounding remainder deterministically while preserving every stratum.
    label_order = sorted(
        groups,
        key=lambda label: (
            -(len(groups[label]) * fraction % 1),
            hashlib.sha256(f"{spec['seed']}\0{namespace}\0{label}".encode()).digest(),
        ),
    )
    while sum(quotas.values()) < target:
        changed = False
        for label in label_order:
            if quotas[label] < len(groups[label]) - 1:
                quotas[label] += 1
                changed = True
                if sum(quotas.values()) == target:
                    break
        if not changed:
            break
    while sum(quotas.values()) > target:
        changed = False
        for label in reversed(label_order):
            if quotas[label] > 1:
                quotas[label] -= 1
                changed = True
                if sum(quotas.values()) == target:
                    break
        if not changed:
            break

    keep = set()
    for label, indices in groups.items():
        duplicate_groups = {}
        for index in indices:
            duplicate_groups.setdefault(row_key(index), []).append(index)
        ranked_groups = sorted(
            duplicate_groups,
            key=lambda key: hashlib.sha256(
                f"{spec['seed']}\0{namespace}\0label:{label}\0{key}".encode()
            ).digest(),
        )
        if len(ranked_groups) < 2:
            raise ValueError(
                f"{namespace}/label:{label} needs two distinct examples"
            )
        # Keep exact duplicate evaluation rows together. Choose the ranked prefix
        # whose example count is closest to the stratum quota.
        cumulative = 0
        best_cut, best_distance = 0, float("inf")
        for cut, key in enumerate(ranked_groups, 1):
            cumulative += len(duplicate_groups[key])
            distance = abs(cumulative - quotas[label])
            if distance < best_distance:
                best_cut, best_distance = cut, distance
        best_cut = min(max(best_cut, 1), len(ranked_groups) - 1)
        dev_indices = {
            index
            for key in ranked_groups[:best_cut]
            for index in duplicate_groups[key]
        }
        keep.update(
            dev_indices if partition == "dev" else set(indices) - dev_indices
        )
    return sorted(keep)


def _split_loaded_v2_retrieval(task, partition, spec):
    """Split query groups in MTEB v2 retrieval/reranking data."""
    for subset, subset_data in task.dataset.items():
        for hf_split, split_data in subset_data.items():
            queries = split_data.get("queries")
            qrels = split_data.get("relevant_docs")
            if queries is None or not isinstance(qrels, dict):
                raise TypeError(
                    f"{task.metadata.name}/{subset}/{hf_split} has unsupported "
                    "retrieval data"
                )
            query_ids = [str(query_id) for query_id in queries["id"]]
            keep = _ranked_partition(
                query_ids,
                partition,
                spec,
                f"{task.metadata.name}\0{subset}\0{hf_split}",
            )
            split_data["queries"] = queries.select([
                index for index, query_id in enumerate(query_ids)
                if query_id in keep
            ])
            split_data["relevant_docs"] = {
                query_id: docs for query_id, docs in qrels.items()
                if str(query_id) in keep
            }
            top_ranked = split_data.get("top_ranked")
            if isinstance(top_ranked, dict):
                split_data["top_ranked"] = {
                    query_id: docs for query_id, docs in top_ranked.items()
                    if str(query_id) in keep
                }


def _split_loaded_nonretrieval(task, partition, spec):
    """Split complete evaluation examples while leaving training splits intact."""
    from datasets import Dataset, DatasetDict

    eval_splits = set(getattr(task, "eval_splits", task.metadata.eval_splits))
    label_column = getattr(task, "label_column_name", None)

    def split_mapping(mapping, subset):
        for hf_split in list(mapping):
            if hf_split not in eval_splits:
                continue
            dataset = mapping[hf_split]
            if not isinstance(dataset, Dataset):
                continue
            # Older pair-classification datasets store all pairs as lists in one row.
            if (
                task.metadata.type == "PairClassification"
                and len(dataset) == 1
                and all(isinstance(value, list) for value in dataset[0].values())
            ):
                dataset = Dataset.from_dict(dataset[0])
            indices = _dataset_partition_indices(
                dataset,
                partition,
                spec,
                f"{task.metadata.name}\0{subset}\0{hf_split}",
                label_column=label_column,
            )
            mapping[hf_split] = dataset.select(indices)

    dataset = task.dataset
    if isinstance(dataset, DatasetDict):
        split_mapping(dataset, "default")
    elif isinstance(dataset, dict):
        if any(isinstance(value, Dataset) for value in dataset.values()):
            split_mapping(dataset, "default")
        else:
            for subset, mapping in dataset.items():
                if isinstance(mapping, (dict, DatasetDict)):
                    split_mapping(mapping, subset)
    else:
        raise TypeError(
            f"{task.metadata.name} has unsupported dataset type {type(dataset)}"
        )


def _split_loaded_task_examples(task, partition, spec):
    marker = (partition, float(spec["dev_fraction"]), str(spec["seed"]))
    if getattr(task, "_autoembed_example_partition", None) == marker:
        return
    if task.metadata.type in ("Retrieval", "Reranking"):
        if hasattr(task, "queries") and getattr(task, "queries", None):
            _split_loaded_retrieval_queries(task, partition, spec)
        else:
            _split_loaded_v2_retrieval(task, partition, spec)
    else:
        _split_loaded_nonretrieval(task, partition, spec)
    task._autoembed_example_partition = marker


def split_task_examples(tasks, partition, spec):
    """Attach a deterministic example partition for all dense MTEB task types."""
    if partition not in ("dev", "heldout"):
        raise ValueError("example partition must be dev or heldout")
    for task in tasks:
        if getattr(task, "data_loaded", False):
            _split_loaded_task_examples(task, partition, spec)
            continue
        original_load_data = task.load_data

        def load_partitioned(*args, _task=task, _load=original_load_data, **kwargs):
            _load(*args, **kwargs)
            _split_loaded_task_examples(_task, partition, spec)

        task.load_data = load_partitioned
    return tasks


def evaluate(model_path=MODEL_DIR, task_names=None):
    # Score a model on the dev suite: the harness task list if provided, else the built-in default.
    import mteb
    names = task_names or DEV_TASKS
    if names:
        tasks = mteb.get_tasks(tasks=names)
    else:
        import nano_dev  # built-in default suite (only present when the harness ships it)
        tasks = nano_dev.dev_tasks()
    if QUERY_SPLIT:
        tasks = split_retrieval_queries(
            tasks, "dev", QUERY_SPLIT, task_names=DEV_TASKS or names
        )
    if EXAMPLE_SPLIT:
        tasks = split_task_examples(tasks, "dev", EXAMPLE_SPLIT)
    r = _score(model_path, tasks, tag="dev")
    print(f"mean_type={r['mean_type']:.4f}  mean_task={r['mean_task']:.4f}")
    for ty, s in sorted(r["per_type"].items()):
        print(f"   {ty:14s} {s:.4f}")
    if r["skipped"]:
        print(f"   (skipped: {r['skipped']})")
    return r


def _norm(s):
    return " ".join(s.lower().split())


def _h(s):
    return hashlib.sha256(_norm(s).encode("utf-8")).hexdigest()


def _iter_strings(value):
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_strings(nested)


def _collect_hashes(obj, out):
    if isinstance(obj, str):
        if obj.strip():
            out.add(_h(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_hashes(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_hashes(value, out)


def _pair_h(left, right):
    """Hash an unordered pair of already-normalized text hashes."""
    first, second = sorted((left, right))
    return hashlib.sha256(f"{first}:{second}".encode()).hexdigest()


def _eval_groups():
    """Return hidden hashes grouped by evaluation role.

    List payloads are the legacy cache format and remain fully protected.
    """
    if not _EVAL_CACHE.exists():
        return {}
    payload = json.loads(_EVAL_CACHE.read_text())
    if isinstance(payload, list):
        return {"protected": set(payload)}
    if isinstance(payload, dict):
        return {
            name: set(hashes)
            for name, hashes in payload.items()
            if name in ("queries", "relevant", "other_corpus", "protected")
            and isinstance(hashes, list)
        }
    return {}


def _eval_texts():
    # Hashed held-out text set, precomputed by the harness.
    groups = _eval_groups()
    return set().union(*groups.values()) if groups else set()


def check_contamination(train_dataset, sample=None, model_path=None, sources=None):
    """Audit training text against hidden evaluation text.

    The default is exhaustive. Pass model_path when auditing the exact dataset
    used by a submitted checkpoint; the scorer only accepts that artifact-bound
    manifest for reportable configurations.
    Agent workdirs ship hashes of public corpus text only (see audit_roles in
    the result); hidden query text is withheld and verified by the external
    scorer. Pass sources as dataset identifiers or provenance objects containing
    dataset, revision, and split. Version-2 manifests also hash query/document
    pairs within each row, without storing their text.
    """
    groups = _eval_groups()
    evalset = set().union(*groups.values()) if groups else set()
    hidden_audit_available = bool(evalset)
    dataset_rows = len(train_dataset)
    if sample is None:
        checked_rows = dataset_rows
    else:
        if sample < 0:
            raise ValueError("sample must be non-negative or None")
        checked_rows = min(sample, dataset_rows)

    columns = [
        column for column in (
            "anchor", "positive", "negative", "query", "text", "document",
            "documents", "passage", "question", "answer", "title",
        )
        if column in train_dataset.column_names
    ]
    if not columns:
        raise ValueError(
            "No recognized text columns found; rename text-bearing columns before auditing"
        )

    rows = (
        train_dataset
        if checked_rows == dataset_rows
        else train_dataset.select(range(checked_rows))
    )
    train_hashes, query_document_pair_hashes, hits, examples = set(), set(), 0, []
    checked_strings = 0
    for row in rows:
        row_hashes = {}
        for column in columns:
            column_hashes = set()
            for value in _iter_strings(row.get(column)):
                checked_strings += 1
                digest = _h(value)
                train_hashes.add(digest)
                column_hashes.add(digest)
                if digest in evalset:
                    hits += 1
                    if len(examples) < 5:
                        examples.append(value[:80])
            row_hashes[column] = column_hashes

        query_hashes = set().union(*(
            row_hashes.get(column, set()) for column in _QUERY_COLUMNS
        ))
        document_hashes = set().union(*(
            row_hashes.get(column, set()) for column in _DOCUMENT_COLUMNS
        ))
        query_document_pair_hashes.update(
            _pair_h(query_hash, document_hash)
            for query_hash in query_hashes
            for document_hash in document_hashes
            if query_hash != document_hash
        )

    if sources is None:
        sources = []
    if not isinstance(sources, list) or not all(
        isinstance(source, (str, dict)) for source in sources
    ):
        raise ValueError("sources must be a list of dataset identifiers or provenance objects")
    if dataset_rows and not sources:
        print(
            "!! no sources declared: this submission will be scored as the base model. "
            "Re-run check_contamination with sources=[{'dataset': 'org/name', 'revision': ...}].",
            file=sys.stderr,
        )

    manifest = {
        "version": 2,
        "hash": TEXT_HASH_ALGORITHM,
        "pair_hash": PAIR_HASH_ALGORITHM,
        "dataset_rows": dataset_rows,
        "checked_rows": checked_rows,
        "checked_strings": checked_strings,
        "columns": columns,
        "exhaustive": checked_rows == dataset_rows,
        "hashes": sorted(train_hashes),
        "query_document_pair_hashes": sorted(query_document_pair_hashes),
        "sources": sources,
        "hits_at_creation": hits if hidden_audit_available else None,
    }
    if model_path is None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path = RUNS_DIR / "_train_texts.json"
    else:
        artifact = Path(model_path)
        artifact.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact / TRAINING_MANIFEST
    manifest_path.write_text(json.dumps(manifest))

    denominator = max(checked_strings, 1)
    return {
        "checked_rows": checked_rows,
        "checked_strings": checked_strings,
        "hidden_audit_available": hidden_audit_available,
        "audit_roles": sorted(groups),
        "eval_texts": len(evalset),
        "unique_train_texts": len(train_hashes),
        "query_document_pairs": len(query_document_pair_hashes),
        "exhaustive": manifest["exhaustive"],
        "manifest": str(manifest_path),
        "hits": hits if hidden_audit_available else None,
        "frac": round(hits / denominator, 8) if hidden_audit_available else None,
        "examples": examples,
    }
