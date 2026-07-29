# Opus wave 1 — preempted 2026-07-29

Three `claude-opus-5` runs launched 17:42, all preempted on slinky-1 between 19:28
and 19:58 when the node owner (`user-pgasawa`, tier 3) claimed it. Runs discarded
rather than resumed: a spliced run is a different experimental condition from the
uninterrupted Sonnet arm, so it is not comparable. Traces retained under
`archive/preempted-opus/`.

| job | config | survived | snapshots | notes |
|-----|--------|----------|-----------|-------|
| 9921 | mteb-nano-create | 1h40m | 0 | no `final_model/` written yet |
| 9922 | mteb-nano-improve | 2h10m | 1 | trained "model A" |
| 9923 | legal | 2h11m | 0 | still assembling training data |

Cost is unrecoverable: a SIGKILLed session emits no `result` event, so all three
record `source=claude-stream-partial` with `provider_reported_cost_usd=None`. Quota
was spent; the amount is not in the trace.

## Strategies observed

Each config drew a different approach, all more elaborate than the Sonnet arm's
single-stage contrastive fine-tuning.

**mteb-nano-create — teacher/student distillation.** Benchmarked candidate teachers
(`bench_teacher.py`), probed output dimensionality (`teacher_dim_test.py`), embedded
a 9.9M-example corpus (`embed_teacher.py`), then distilled into raw ModernBERT with a
combined MSE + cosine objective (`train_distill.py`), with a contrastive stage
prepared as an alternative. Reached step 400/15980 before preemption.

**mteb-nano-improve — prompt-space search.** Built `prompts.json` and swept
task-type prompt prefixes (`sweep_prompts.py`, `set_prompts.py`) against a base
measurement, exploiting the fact that MTEB submits task-type-specific prefixes,
alongside conventional training (`train.py`). Establishes that part of the available
gain on a strong embedder is prompt configuration rather than weights.

**legal — synthetic data generation with sub-agents.** Spawned parallel research
agents to survey candidate corpora, probed ~40 legal datasets, installed vLLM
(pinned to 0.11.0 after 0.26 pulled a torch requiring a newer driver than the node
has) to generate ~80k synthetic queries with Qwen3-8B, mined hard negatives, and
reverse-engineered `TEXT_HASH_ALGORITHM` from `task.py` to build its own
contamination blocklist before training. Dev baseline 0.6096.

## Protocol observations

**Sub-agents contend for the single GPU.** The legal agent found its own spawned
children occupying the H100 and had to reason about it:

> "A concurrent job I did not start is using my script and the GPU."
> "The sibling agent is running the real 80k-prompt synthesis ... I will not kill it."

A one-GPU budget is shared with any sub-agents the agent spawns, which is worth
stating explicitly in the protocol description.

**Agents self-filter contamination.** The legal agent independently located the
hashing scheme and filtered its training data against the hidden-text cache before
training — behaviour the harness permits by design (`write_agent_eval_cache`), and
evidence that the audit is not purely adversarial.

**Preemption exposure.** `guest` is the only partition available (`shared` holds no
nodes after the 2026-07-29 cutover), and only tier-3 owner partitions can preempt.
Node choice is therefore the sole lever, and owner history is a weak predictor:
slinky-1 had zero preemptions in the preceding 28 days.
