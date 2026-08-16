# Task

Post-train the fixed checkpoint in `task.py` (`BASE_MODEL` at `BASE_REVISION`) to
maximize the development score returned by `evaluate()`. Conduct autonomous research and
development, run experiments, and iterate for the available time. Save your best model in
`final_model/`.

## Evaluation and submission

`evaluate(model_path)` evaluates an MTEB development suite. After the run, the harness evaluates
`final_model/` on a hidden held-out set under the same protocol. The scorer uses only files saved in
`final_model/`.

The submission must be a post-trained version of `BASE_MODEL` at `BASE_REVISION` and
must run offline in the starting environment. It may be a standard SentenceTransformer
directory or define
`mteb_model.py:load_model(model_path)` returning an MTEB-compatible dense encoder. The interfaces
and evaluation functions in `task.py` define the ground truth; do not modify that file.

## Resources

You have one NVIDIA GPU and a fixed time budget. Use `uv run` for Python and `bash timer.sh` to
check the remaining time.

Web research and public downloads are allowed. The submitted model must be derived from
`BASE_MODEL` at `BASE_REVISION`. Load it with the revision argument so the run cannot
drift. All computation must run on the resources allocated to this task; do not use remote
inference APIs or services. The final model must work in the starting environment.

## Rules

- Follow the run-specific data policy below. Do not use protected evaluation data for training.
- Run `check_contamination(final_training_ds, model_path="final_model")` on the complete data used
  to train the submitted model. Pass `sources=[{"dataset": "org/name", "revision": "...",
  "split": "train"}, ...]` for every source. A valid exhaustive manifest is required for hidden
  scoring; it records only normalized hashes and provenance, never raw training text.
- Work only in the current directory and its subdirectories. Do not search for hidden harness or
  configuration files.
- Do not ask the user for feedback. Operate autonomously, use the available time, and leave the
  best valid submission in `final_model/` when you finish.

## Helper Model API

A powerful LLM (Qwen-27B) is hosted locally on the host machine and is available to assist you.
- API Endpoint URL: The endpoint base URL is provided in the environment variable `QWEN_API_BASE` (e.g. `http://host.docker.internal:8000/v1` or `http://127.0.0.1:8000/v1`).
- Model Name: `Qwen/Qwen3.6-27B`
- Format: OpenAI-compatible API. You can write Python code inside your training scripts to query this model for text rewriting, synthetic data pair generation, labeling, or other tasks. Feel free to use standard Python libraries or install helper packages (e.g., `openai`) if needed.
