# Results (preliminary)

Base **`intfloat/e5-base-unsupervised`**. Held-out = **MTEB(eng, v2) − MindSmallReranking**
(40 tasks), scored as the **Mean over task types** (Mean(Task) shown where available). Every
number is scored by us on the same held-out with the same config (`score.py` / `reference.py`)
— the agent never runs the held-out, so these are not tune-on-test.

## Reference ladder
| Model | Mean(Type) | Mean(Task) | Role |
|---|---|---|---|
| `e5-base-unsupervised` (raw) | 0.533 | 0.533 | floor |
| `e5-base-v2` | **0.609** | 0.626 | anchor — same base, fully fine-tuned |
| `bge-base-en-v1.5` | 0.626 | 0.652 | strong small encoder |
| `Qwen3-Embedding-0.6B` | ~0.71 | — | SOTA-small (LLM-decoder; cited, not scored here) |

## Agent runs
| Agent | Budget | Mean(Type) | Mean(Task) | Notes |
|---|---|---|---|---|
| Claude Sonnet 5 | 1h | 0.593 | — | first e5 run |
| Claude Sonnet 5 | 5h | 0.595 | 0.614 | retrieval-focused (self-built retrieval-only validation) |
| Claude Sonnet 5 | 5h | 0.588 | 0.610 | broad objective + provided `evaluate_dev`; per-type-guided iteration |
| Claude Opus 4.8 | 5h | — | — | pending |

## Read
- All Sonnet runs cluster at **~0.59, below the 0.609 anchor**, whether retrieval-focused or
  broad — a likely Sonnet/base/budget ceiling, not a strategy problem.
- The provided multi-task dev changed the agent's *behavior* (broad, per-type-guided
  iteration) but not the held-out *outcome*. Under hard optimization the dev over-states
  (dev 0.725 vs held-out 0.588 — mild Goodhart).
- Open question: does **Opus** break the ~0.59 plateau and clear the anchor?

Excludes: a failed run (agent yielded before training → scored the untrained base, 0.533) and
an `mpnet-base` run on an earlier eval (not comparable). `MindSmallReranking` is dropped from
the held-out (>1h/model to score).
