# Canonical experiment configs

The six canonical configs each define a complete inductive experiment:

- `general/mteb-nano-create.json`: create a general embedder from ModernBERT.
- `general/mteb-nano-improve.json`: improve a strong Nomic general embedder.
  (Both `general/` configs need the locally built MTEB-nano artifacts in `runs/nano/`,
  whose filenames and checksums are pinned by `agent_task/nano_assets.json` but whose large data files are not distributed in Git.)
- `specialization/legal.json`: specialize GTE-ModernBERT for legal retrieval
  (bar-exam QA, case summarisation, consumer contracts).
- `specialization/finance.json`: specialize GTE-ModernBERT for financial retrieval.
- `specialization/medical.json`: specialize GTE-ModernBERT for medical retrieval,
  spanning lexical and reasoning-driven search.
- `specialization/code.json`: specialize GTE-ModernBERT for code retrieval.

BRIGHT is currently omitted from the canonical run set. Its validated protocol spec is retained at
`experiments/reasoning-bright.json` for future work, but `autoembed list` does not expose it.

Specialization suites are retrieval-only, so the task-type mean reduces to a plain
mean over tasks. Every task must clear three measurement-validity screens: at least
50 hidden queries (below that a single query moves nDCG@10 by more than 0.02), a base
score at or below 0.90 (nothing left to win), and a raw-encoder-to-best-model range of
at least 0.10 (otherwise the task cannot separate a good encoder from a bad one).
Inherited suites (MTEB-Nano) keep official benchmark membership; individual tasks are
dropped only for documented measurement-validity failures (MindSmallReranking).

Every task is split deterministically into visible development examples and a
complementary hidden evaluation set. Evaluation text, including retrieval corpora,
is forbidden as training data. Historical runs retain their exact config in their
run directory.

Canonical configs use the `open-data` contamination policy. Evaluation tasks are
built from the same public datasets an embedder trains on, so text collisions are
expected rather than incriminating: an exact hidden-query match from a declared
source is scored on the unchanged full test set and labelled
`reportable_with_query_exposure`. A submission is flagged only when the evidence
points at the score itself:

1. the training manifest is absent, not exhaustive, or declares no sources;
2. a hidden query and one of its relevant documents co-occur in the same training
   row (version-2 manifests hash query/document pairs per row, which is the direct
   signature of fitting the test set);
3. evaluation text — relevant documents, corpus, and non-retrieval task text —
   is ingested wholesale, exceeding both an absolute count and a fraction of the
   training set.

A flagged run is not discarded. It is scored as its own base model and stays in the
results table, so a flag costs the claimed gain rather than the observation. Legacy
version-1 manifests remain readable but explicitly lack row-pair evidence.

All canonical configs set `allow_target_corpus_training` to `false`. Retrieval
development and held-out partitions have complementary query IDs and qrels; the
corpus is shared only for evaluation. Setting the flag to `true` is reserved for
the separate, non-canonical `target-specialization` protocol, whose claim is
transductive adaptation rather than inductive generalization.

Here, “held-out” means withheld by the harness during scoring, not secret against
an adversarial participant. The benchmark queries, qrels, and deterministic split
recipe are public. Use private or freshly annotated queries and qrels for claims
that require leakage resistance; the open-data protocol detects exact normalized
text and query/document-pair reuse but cannot prove intent or detect paraphrases.

`base_model` and its immutable `base_revision` form the agent’s starting checkpoint,
which is always scored as the base floor. `references` are additional fixed comparison
encoders scored externally on the same hidden split; they guide interpretation but are not
ceilings.

## Hidden-scoring isolation

Standard offline SentenceTransformer folders can be scored in Enroot or Docker. A submission that
contains `mteb_model.py` is executable, untrusted code and is therefore accepted by the hidden
scorer only in `MODE=docker`, where it receives the model directory and encoding batches but not
the hidden dataset objects, relevance judgments, repository, credentials, host network, or host
process namespace. Native and Enroot custom-code scoring fail closed.

