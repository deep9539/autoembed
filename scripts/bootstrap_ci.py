"""Per-query nDCG@10 and paired bootstrap confidence intervals on the hidden split.

Resamples hidden queries, not runs: the interval answers "given this many hidden
queries, is the gain distinguishable from zero?" Evaluation variance only; it says
nothing about how much an agent would vary if re-run.

  scripts/bootstrap_ci.py <config> <agent_model_dir> [--out FILE] [--draws N]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytrec_eval
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "agent_task"))


def _text(row):
    if isinstance(row, str):
        return row
    return ((row.get("title") or "") + " " + (row.get("text") or "")).strip()


def _splits(task):
    """Yield (queries, corpus, qrels) as id/text pairs.

    mteb exposes retrieval data two ways depending on the task: as direct
    queries/corpus/relevant_docs attributes, or nested under .dataset. Reading only
    one silently drops whole tasks, so handle both — the same rule the evaluation
    cache uses.
    """
    from autoembed.scoring import _retrieval_splits

    queries, corpus, qrels = (getattr(task, a, None)
                              for a in ("queries", "corpus", "relevant_docs"))
    if isinstance(queries, dict) and isinstance(corpus, dict):
        for split, split_queries in queries.items():
            split_corpus, split_qrels = corpus.get(split), (qrels or {}).get(split, {})
            if not isinstance(split_corpus, dict) or not isinstance(split_queries, dict):
                continue
            yield ([(q, _text(v)) for q, v in split_queries.items()],
                   [(c, _text(v)) for c, v in split_corpus.items()],
                   split_qrels)
        return
    for data in _retrieval_splits(task):
        yield ([(r["id"], _text(r)) for r in (data.get("queries") or [])],
               [(r["id"], _text(r)) for r in (data.get("corpus") or [])],
               data.get("relevant_docs") or {})


def per_query_ndcg(model, task, k=10, batch_size=128):
    """nDCG@10 per hidden query, scored the way pytrec_eval scores it for mteb."""

    task.load_data()
    qrels, run = {}, {}
    for queries, corpus, relevant in _splits(task):
        if not queries or not corpus:
            continue
        qids, qtext = zip(*queries)
        cids, ctext = zip(*corpus)
        qids, qtext, cids, ctext = list(qids), list(qtext), list(cids), list(ctext)
        qe = model.encode(qtext, batch_size=batch_size, convert_to_tensor=True,
                          normalize_embeddings=True, show_progress_bar=False)
        ce = model.encode(ctext, batch_size=batch_size, convert_to_tensor=True,
                          normalize_embeddings=True, show_progress_bar=False)
        sim = (qe @ ce.T).float().cpu()
        top = min(1000, sim.shape[1])
        scores, idx = torch.topk(sim, k=top, dim=1)
        for i, qid in enumerate(qids):
            gold = relevant.get(qid) or {}
            if not gold:
                continue
            qrels[qid] = {str(d): int(v) for d, v in gold.items()}
            run[qid] = {str(cids[j]): float(scores[i, r])
                        for r, j in enumerate(idx[i].tolist())}
    if not qrels:
        return {}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k}"})
    return {q: v[f"ndcg_cut_{k}"] for q, v in evaluator.evaluate(run).items()}


def bootstrap(base_scores, agent_scores, draws=10000, seed=0):
    """Paired bootstrap: identical resampled queries for both models each draw."""
    rng = np.random.default_rng(seed)
    tasks = sorted(base_scores)
    per_draw_base = np.zeros((draws, len(tasks)))
    per_draw_agent = np.zeros((draws, len(tasks)))
    for t, name in enumerate(tasks):
        qids = sorted(set(base_scores[name]) & set(agent_scores[name]))
        b = np.array([base_scores[name][q] for q in qids])
        a = np.array([agent_scores[name][q] for q in qids])
        picks = rng.integers(0, len(qids), size=(draws, len(qids)))
        per_draw_base[:, t] = b[picks].mean(axis=1)
        per_draw_agent[:, t] = a[picks].mean(axis=1)
    return per_draw_base.mean(axis=1), per_draw_agent.mean(axis=1), tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("model", help="agent final_model directory")
    ap.add_argument("--out")
    ap.add_argument("--draws", type=int, default=10000)
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        config_path = _ROOT / "configs" / f"{args.config}.json"
    os.environ["AUTOEMBED_CONFIG"] = str(config_path)

    from sentence_transformers import SentenceTransformer
    from autoembed.scoring import CONFIG, HELDOUT_TASKS, load_pinned_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = load_pinned_model({
        "name": CONFIG["base_model"], "revision": CONFIG.get("base_revision"),
        "loader": CONFIG.get("base_loader", "sentence-transformer"),
    })
    agent = SentenceTransformer(args.model, trust_remote_code=True, device=device)

    base_scores, agent_scores, aggregate = {}, {}, {}
    for task in HELDOUT_TASKS:
        name = task.metadata.name
        b = per_query_ndcg(base, task)
        a = per_query_ndcg(agent, task)
        if not b or not a:
            print(f"  {name:38s} skipped (no gradeable queries)")
            continue
        base_scores[name], agent_scores[name] = b, a
        aggregate[name] = {"n": len(b),
                           "base": float(np.mean(list(b.values()))),
                           "agent": float(np.mean(list(a.values())))}
        print(f"  {name:38s} n={len(b):5d}  base={aggregate[name]['base']:.4f} "
              f"agent={aggregate[name]['agent']:.4f}")

    db, da, tasks = bootstrap(base_scores, agent_scores, draws=args.draws)
    delta = da - db
    result = {
        "config": str(config_path), "model": args.model, "draws": args.draws,
        "tasks": tasks, "per_task": aggregate,
        "base_mean": float(db.mean()),
        "agent_mean": float(da.mean()),
        "delta_mean": float(delta.mean()),
        "delta_ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
        "p_delta_le_zero": float((delta <= 0).mean()),
    }
    lo, hi = result["delta_ci95"]
    print(f"\n  delta = {result['delta_mean']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"   P(delta<=0) = {result['p_delta_le_zero']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
