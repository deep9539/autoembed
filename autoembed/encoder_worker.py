"""JSON-lines worker for untrusted final_model/mteb_model.py code."""
import base64
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_task.task import _load_custom_encoder


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _decode_array(payload):
    return np.load(io.BytesIO(base64.b64decode(payload)), allow_pickle=False)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def main():
    model_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/model")
    # Submission prints belong in score.log, never on the RPC stream.
    with contextlib.redirect_stdout(sys.stderr):
        model = _load_custom_encoder(model_path)
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            if request.get("op") == "close":
                return 0
            operation = request.get("op")
            if operation in ("similarity", "similarity_pairwise"):
                method = getattr(model, operation)
                arrays = request["arrays"]
                with contextlib.redirect_stdout(sys.stderr):
                    result = method(
                        _decode_array(arrays["embeddings1"]),
                        _decode_array(arrays["embeddings2"]),
                    )
                buffer = io.BytesIO()
                np.save(buffer, _array(result), allow_pickle=False)
                response = {
                    "ok": True,
                    "npy": base64.b64encode(buffer.getvalue()).decode("ascii"),
                }
                sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
                sys.stdout.flush()
                continue
            if operation != "encode":
                raise ValueError("unknown worker operation")
            metadata = _namespace(request.get("task_metadata") or {})
            prompt_type = request.get("prompt_type")
            if prompt_type is not None:
                from mteb.models import PromptType
                prompt_type = PromptType(prompt_type)
            reserved = {
                "op", "inputs", "task_metadata", "hf_split", "hf_subset",
                "prompt_type",
            }
            kwargs = {key: value for key, value in request.items() if key not in reserved}
            with contextlib.redirect_stdout(sys.stderr):
                embeddings = model.encode(
                    request["inputs"], task_metadata=metadata,
                    hf_split=request.get("hf_split"),
                    hf_subset=request.get("hf_subset"), prompt_type=prompt_type,
                    **kwargs,
                )
            buffer = io.BytesIO()
            np.save(buffer, _array(embeddings), allow_pickle=False)
            response = {
                "ok": True,
                "npy": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
        except Exception:
            response = {"ok": False, "error": traceback.format_exc(limit=12)}
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
