import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BRIGHT_INDUCTIVE_CONFIG = ROOT / "experiments" / "reasoning-bright.json"
MEDICAL_CONFIG = ROOT / "configs" / "specialization" / "medical.json"
FINANCE_CONFIG = ROOT / "configs" / "specialization" / "finance.json"
LEGAL_CONFIG = ROOT / "configs" / "specialization" / "legal.json"
GENERAL_CREATE_CONFIG = ROOT / "configs" / "general" / "mteb-nano-create.json"
GENERAL_IMPROVE_CONFIG = ROOT / "configs" / "general" / "mteb-nano-improve.json"
os.environ.setdefault("AUTOEMBED_CONFIG", str(BRIGHT_INDUCTIVE_CONFIG))

from autoembed import scoring as score  # noqa: E402
import task  # noqa: E402


class _Dataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(self.rows[0]) if self.rows else []

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def select(self, indices):
        return _Dataset([self.rows[index] for index in indices])


class _RetrievalTask:
    def __init__(self):
        self.metadata = SimpleNamespace(name="ExampleRetrieval", type="Retrieval")
        self.queries = {"test": {str(i): f"query {i}" for i in range(10)}}
        self.relevant_docs = {
            "test": {str(i): {f"doc-{i}": 1} for i in range(10)}
        }
        self.corpus = {"test": {f"doc-{i}": {"text": f"document {i}"} for i in range(10)}}
        self.top_ranked = {"test": {str(i): [f"doc-{i}"] for i in range(10)}}

    def load_data(self):
        return None


class _DatasetTask:
    def __init__(self, name, task_type, dataset):
        self.metadata = SimpleNamespace(
            name=name, type=task_type, eval_splits=["test"]
        )
        self.dataset = dataset
        self.data_loaded = True

    def load_data(self):
        return None


class PaperProtocolTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(BRIGHT_INDUCTIVE_CONFIG.read_text())
        self.transfer_config = dict(self.config)
        tasks = self.transfer_config.pop("tasks")
        self.transfer_config.pop("example_split")
        self.transfer_config.update({
            "protocol_version": "test-transfer-v1",
            "protocol_type": "transfer",
            "dev_tasks": tasks[:6],
            "heldout_tasks": tasks[6:],
            "linked_task_groups": [[
                "BrightTheoremQAQuestionsRetrieval",
                "BrightTheoremQATheoremsRetrieval",
            ]],
        })
        self.target_config = {
            "protocol_version": "test-target-v1",
            "protocol_type": "target-specialization",
            "expected_task_count": len(tasks),
            "tasks": tasks,
            "query_split": {
                "dev_fraction": 0.5,
                "seed": "test-target-seed",
            },
            "allow_target_corpus_training": True,
        }

    def test_paper_split_is_complete_and_keeps_linked_tasks_together(self):
        self.assertTrue(score.validate_config(self.transfer_config))
        dev = set(self.transfer_config["dev_tasks"])
        heldout = set(self.transfer_config["heldout_tasks"])

        self.assertEqual(len(dev), 6)
        self.assertEqual(len(heldout), 6)
        self.assertEqual(len(dev | heldout), 12)
        self.assertFalse(dev & heldout)
        theorem_tasks = {
            "BrightTheoremQAQuestionsRetrieval",
            "BrightTheoremQATheoremsRetrieval",
        }
        self.assertTrue(theorem_tasks <= dev or theorem_tasks <= heldout)

    def test_bright_fold_b_is_the_exact_complement(self):
        fold_b = dict(self.transfer_config)
        fold_b["dev_tasks"] = self.transfer_config["heldout_tasks"]
        fold_b["heldout_tasks"] = self.transfer_config["dev_tasks"]
        self.assertTrue(score.validate_config(fold_b))
        self.assertEqual(
            set(self.transfer_config["dev_tasks"]), set(fold_b["heldout_tasks"])
        )
        self.assertEqual(
            set(self.transfer_config["heldout_tasks"]), set(fold_b["dev_tasks"])
        )

    def test_target_protocol_is_same_corpus_query_holdout(self):
        target = self.target_config
        self.assertTrue(score.validate_config(target))
        self.assertEqual(target["protocol_type"], "target-specialization")
        self.assertTrue(target["allow_target_corpus_training"])
        self.assertEqual(len(target["tasks"]), 12)
        self.assertEqual(target["query_split"]["dev_fraction"], 0.5)

    def test_inductive_bright_uses_hidden_examples_without_corpus_training(self):
        inductive = json.loads(BRIGHT_INDUCTIVE_CONFIG.read_text())
        self.assertTrue(score.validate_config(inductive))
        self.assertEqual(inductive["protocol_type"], "benchmark-target")
        self.assertEqual(inductive["base_loader"], "sentence-transformer")
        self.assertFalse(inductive["allow_target_corpus_training"])
        self.assertEqual(len(inductive["tasks"]), 12)
        self.assertEqual(inductive["example_split"]["dev_fraction"], 0.5)

    def test_general_protocol_is_same_task_hidden_examples(self):
        create = json.loads(GENERAL_CREATE_CONFIG.read_text())
        improve = json.loads(GENERAL_IMPROVE_CONFIG.read_text())
        for general in (create, improve):
            self.assertTrue(score.validate_config(general))
            self.assertEqual(general["protocol_type"], "benchmark-target")
            self.assertEqual(general["dev_benchmark"], "mteb-nano")
            self.assertEqual(general["heldout_benchmark"], "mteb-nano")
            self.assertEqual(
                general["benchmark_revision"],
                "b1034a841eab373033794eca94da6dcd6391afa66f54da9438a35b56ab0776e0",
            )
            self.assertFalse(general["allow_target_corpus_training"])
            self.assertEqual(general["example_split"]["dev_fraction"], 0.5)
        self.assertEqual(create["base_model"], "answerdotai/ModernBERT-base")
        self.assertEqual(improve["base_model"], "nomic-ai/modernbert-embed-base")
        self.assertEqual(improve["base_loader"], "mteb")
        self.assertEqual(create["example_split"], improve["example_split"])

    def test_agent_task_has_no_model_specific_base_fallback(self):
        source = (ROOT / "agent_task" / "task.py").read_text()
        self.assertNotIn('os.environ.get("AUTOEMBED_BASE_MODEL",', source)
        self.assertNotIn("answerdotai/ModernBERT-base", source)

    def test_run_prompt_names_the_resolved_base_checkpoint(self):
        source = (ROOT / "scripts" / "run_task.sh").read_text()
        self.assertIn(
            "The fixed starting checkpoint for this run is \\`$BASE_MODEL\\`",
            source,
        )
        self.assertIn("Do not substitute a different checkpoint", source)
        self.assertIn("at immutable revision", source)
        self.assertIn("$BASE_REVISION", source)

    def test_domain_targets_use_shared_tasks_and_hidden_examples(self):
        for path, expected, base in (
            (MEDICAL_CONFIG, 6, "Alibaba-NLP/gte-modernbert-base"),
            (FINANCE_CONFIG, 4, "Alibaba-NLP/gte-modernbert-base"),
            (LEGAL_CONFIG, 3, "Alibaba-NLP/gte-modernbert-base"),
        ):
            config = json.loads(path.read_text())
            self.assertTrue(score.validate_config(config))
            self.assertEqual(config["protocol_type"], "benchmark-target")
            self.assertEqual(len(config["tasks"]), expected)
            self.assertEqual(config["base_model"], base)
            self.assertEqual(config["base_loader"], "sentence-transformer")
            self.assertFalse(config["allow_target_corpus_training"])
            self.assertEqual(config["example_split"]["dev_fraction"], 0.5)
            self.assertEqual(config["max_incidental_overlap_hits"], 10000)
            self.assertEqual(config["max_incidental_overlap_fraction"], 0.001)

    def test_eval_cache_extracts_retrieval_roles_from_container(self):
        groups = {
            name: set()
            for name in ("queries", "relevant", "other_corpus", "protected")
        }
        data = {
            "queries": [{"id": "q1", "text": "hidden query text"}],
            "corpus": [
                {"id": "d1", "title": "", "text": "gold document text"},
                {"id": "d2", "title": "", "text": "background document text"},
            ],
            "relevant_docs": {"q1": {"d1": 1}},
        }
        score._collect_split_hashes(data, groups)
        self.assertEqual(len(groups["queries"]), 1)
        self.assertEqual(len(groups["relevant"]), 1)
        self.assertEqual(len(groups["other_corpus"]), 1)
        self.assertEqual(len(groups["query_relevant_pairs"]), 1)
        self.assertFalse(groups["relevant"] & groups["other_corpus"])
        self.assertFalse(groups["protected"])

    def test_specialization_suites_have_no_singleton_task_types(self):
        import mteb

        for path in (MEDICAL_CONFIG, FINANCE_CONFIG, LEGAL_CONFIG):
            config = json.loads(path.read_text())
            counts = {}
            for name in config["tasks"]:
                task_type = mteb.get_task(name).metadata.type
                counts[task_type] = counts.get(task_type, 0) + 1
            self.assertTrue(
                all(count >= 2 for count in counts.values()),
                f"{path.name}: singleton task types {counts}",
            )

    def test_benchmark_target_gates_test_fitting_but_tolerates_incidental_text(self):
        general = json.loads(GENERAL_CREATE_CONFIG.read_text())

        # Gold documents are ordinary public text, so a sparse overlap is scored;
        # ingesting the evaluation corpus wholesale is the flag.
        gold_audit = {
            "present": True, "valid": True, "query_hits": 0,
            "protected_hits": 0, "relevant_hits": 459,
            "incidental_hits": 0, "unique_train_texts": 400_000,
        }
        self.assertFalse(score.contamination_failures(gold_audit, general))

        ingested = dict(gold_audit, incidental_hits=57_610, unique_train_texts=118_000)
        self.assertTrue(score.contamination_failures(ingested, general))

        paired = dict(gold_audit, query_relevant_pair_hits=1)
        self.assertTrue(score.contamination_failures(paired, general))

        # Query-only exposure is disclosed but remains scoreable in the canonical
        # open-data protocol; no examples are removed from the fixed test set.
        query_audit = dict(gold_audit, relevant_hits=0, query_hits=412)
        self.assertFalse(score.contamination_failures(query_audit, general))
        self.assertTrue(score.contamination_warnings(query_audit, general))
        self.assertEqual(
            score.contamination_reportability(query_audit, []),
            "reportable_with_query_exposure",
        )
        widespread_query_audit = dict(query_audit, query_hits=1001)
        self.assertFalse(
            score.contamination_failures(widespread_query_audit, general)
        )
        # a single incidental text collision in a large corpus passes and is reported
        incidental_audit = {
            "present": True, "valid": True, "query_hits": 0,
            "protected_hits": 0, "relevant_hits": 0,
            "incidental_hits": 1, "unique_train_texts": 427_764,
        }

        # protected task text carries no query-document mapping, so it is bounded
        # proportionally: a lone collision passes, a wholesale ingest does not
        protected_audit = dict(
            incidental_audit, protected_hits=1, incidental_hits=0,
        )
        self.assertFalse(score.contamination_failures(protected_audit, general))
        bulk_protected = dict(
            incidental_audit, protected_hits=60_000, incidental_hits=0,
            unique_train_texts=120_000,
        )
        self.assertTrue(score.contamination_failures(bulk_protected, general))
        self.assertFalse(score.contamination_failures(incidental_audit, general))

        # wholesale corpus ingestion still fails on the fraction limit
        corpus_audit = dict(
            incidental_audit, protected_hits=0, incidental_hits=57_151,
            unique_train_texts=116_324,
        )
        self.assertTrue(score.contamination_failures(corpus_audit, general))

        # tolerance is capped: configs cannot relax past the protocol ceiling
        relaxed = dict(general, max_incidental_overlap_fraction=0.05)
        with self.assertRaisesRegex(ValueError, "caps incidental evaluation-text"):
            score.validate_config(relaxed)

    def test_flagged_submission_is_scored_as_its_base_model(self):
        # A flag costs the claimed gain, not the observation: the run keeps a row
        # in the results table carrying the base model's score.
        base_result = {
            "mean_type": 0.4611, "mean_task": 0.4611,
            "per_type": {"Retrieval": 0.4611},
            "per_task": {"BarExamQA": 0.4611}, "skipped": [],
        }
        loaded = {}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scores.json"
            with patch.object(score, "ensure_eval_cache", lambda: None), \
                 patch.object(score, "audit_contamination", lambda _: {"present": False}), \
                 patch.object(score, "HELDOUT_TASKS", []), \
                 patch.object(score, "load_pinned_model", lambda spec: loaded.update(spec)), \
                 patch.object(score, "_score", lambda *a, **k: base_result), \
                 patch.object(sys, "argv", ["scoring", str(Path(tmp) / "final_model"), str(output)]):
                self.assertEqual(score.main(), 2)

            written = json.loads(output.read_text())

        self.assertFalse(written["protocol_valid"])
        self.assertEqual(written["reportability"], "flagged-base-substituted")
        self.assertEqual(written["substituted_base_model"], score.BASE_MODEL_ID)
        self.assertEqual(written["mean_type"], base_result["mean_type"])
        self.assertIn("missing artifact", written["invalid_reasons"][0])
        # the base floor is loaded at its pinned revision, not at HEAD
        self.assertEqual(loaded["revision"], score.CONFIG["base_revision"])

    def test_query_partitions_are_balanced_and_complementary(self):
        spec = {"dev_fraction": 0.5, "seed": "test-seed"}
        dev, heldout = _RetrievalTask(), _RetrievalTask()
        original_corpus = dev.corpus
        task.split_retrieval_queries([dev], "dev", spec)
        task.split_retrieval_queries([heldout], "heldout", spec)
        dev_ids = set(dev.queries["test"])
        heldout_ids = set(heldout.queries["test"])
        self.assertEqual(len(dev_ids), 5)
        self.assertFalse(dev_ids & heldout_ids)
        self.assertEqual(dev_ids | heldout_ids, {str(i) for i in range(10)})
        self.assertEqual(set(dev.relevant_docs["test"]), dev_ids)
        self.assertEqual(set(heldout.relevant_docs["test"]), heldout_ids)
        self.assertEqual(set(dev.top_ranked["test"]), dev_ids)
        self.assertEqual(set(heldout.top_ranked["test"]), heldout_ids)
        self.assertIs(dev.corpus, original_corpus)
        self.assertEqual(len(heldout.corpus["test"]), 10)

    def test_nonretrieval_partitions_are_stratified_and_complementary(self):
        from datasets import Dataset, DatasetDict

        rows = [
            {"id": str(index), "text": f"example {index}", "label": index % 2}
            for index in range(20)
        ]
        spec = {"dev_fraction": 0.5, "seed": "test-seed"}
        dev = _DatasetTask(
            "ExampleClassification", "Classification",
            DatasetDict({"test": Dataset.from_list(rows)}),
        )
        heldout = _DatasetTask(
            "ExampleClassification", "Classification",
            DatasetDict({"test": Dataset.from_list(rows)}),
        )
        dev.label_column_name = heldout.label_column_name = "label"
        task.split_task_examples([dev], "dev", spec)
        task.split_task_examples([heldout], "heldout", spec)
        dev_rows = list(dev.dataset["test"])
        heldout_rows = list(heldout.dataset["test"])
        dev_ids = {row["id"] for row in dev_rows}
        heldout_ids = {row["id"] for row in heldout_rows}
        self.assertFalse(dev_ids & heldout_ids)
        self.assertEqual(dev_ids | heldout_ids, {str(index) for index in range(20)})
        self.assertEqual({row["label"] for row in dev_rows}, {0, 1})
        self.assertEqual({row["label"] for row in heldout_rows}, {0, 1})

    def test_legacy_pair_rows_split_complete_pairs(self):
        from datasets import Dataset, DatasetDict

        row = {
            "sentence1": [f"left {index}" for index in range(10)],
            "sentence2": [f"right {index}" for index in range(10)],
            "labels": [index % 2 for index in range(10)],
        }
        spec = {"dev_fraction": 0.5, "seed": "test-seed"}
        dev = _DatasetTask(
            "ExamplePairs", "PairClassification",
            DatasetDict({"test": Dataset.from_list([row])}),
        )
        heldout = _DatasetTask(
            "ExamplePairs", "PairClassification",
            DatasetDict({"test": Dataset.from_list([row])}),
        )
        dev.label_column_name = heldout.label_column_name = "labels"
        task.split_task_examples([dev], "dev", spec)
        task.split_task_examples([heldout], "heldout", spec)
        dev_left = set(dev.dataset["test"]["sentence1"])
        heldout_left = set(heldout.dataset["test"]["sentence1"])
        self.assertEqual(len(dev_left), 5)
        self.assertFalse(dev_left & heldout_left)
        self.assertEqual(
            dev_left | heldout_left,
            {f"left {index}" for index in range(10)},
        )

    def test_nonretrieval_dataset_splits_are_fully_hashed(self):
        hashes = set()
        score._collect_dataset_hashes({
            "test": _Dataset([{"sentence": "protected hidden example"}])
        }, hashes)
        self.assertEqual(hashes, {task._h("protected hidden example")})

    def test_canonical_config_directories_contain_the_protocol_set(self):
        configs = ROOT / "configs"
        config_names = {
            str(path.relative_to(configs))
            for category in ("general", "specialization")
            for path in (configs / category).glob("*.json")
        }
        self.assertEqual(
            config_names,
            {
                "general/mteb-nano-create.json",
                "general/mteb-nano-improve.json",
                "specialization/code.json",
                "specialization/finance.json",
                "specialization/legal.json",
                "specialization/medical.json",
            },
        )

    def test_all_canonical_configs_forbid_target_corpus_training(self):
        for path in (ROOT / "configs").glob("*/*.json"):
            config = json.loads(path.read_text())
            self.assertIs(config.get("allow_target_corpus_training"), False, path)
            self.assertTrue(score.validate_config(config), path)
            self.assertRegex(config.get("base_revision", ""), r"^[0-9a-f]{40}$", path)

    def test_reference_subset_accepts_pinned_base_and_rejects_unknown_models(self):
        from autoembed import reference

        with patch.object(sys, "argv", ["autoembed.reference", self.config["base_model"]]):
            specs = reference._reference_specs()
        self.assertEqual(specs[0]["revision"], self.config["base_revision"])
        self.assertEqual(specs[0]["role"], "base-floor")

        with patch.object(sys, "argv", ["autoembed.reference", "example/unpinned"]):
            with self.assertRaisesRegex(ValueError, "unpinned reference"):
                reference._reference_specs()

    def test_reportable_configs_require_pinned_base_and_references(self):
        invalid_base = json.loads(json.dumps(self.config))
        invalid_base["base_revision"] = "main"
        with self.assertRaisesRegex(ValueError, "base_revision"):
            score.validate_config(invalid_base)

        invalid_reference = json.loads(json.dumps(self.config))
        invalid_reference["references"][0].pop("revision")
        with self.assertRaisesRegex(ValueError, "references require"):
            score.validate_config(invalid_reference)

    def test_linked_tasks_cannot_cross_dev_and_heldout(self):
        invalid = json.loads(json.dumps(self.transfer_config))
        questions = "BrightTheoremQAQuestionsRetrieval"
        theorems = "BrightTheoremQATheoremsRetrieval"
        invalid["heldout_tasks"].remove(theorems)
        invalid["dev_tasks"].append(theorems)
        self.assertIn(questions, invalid["heldout_tasks"])

        with self.assertRaisesRegex(ValueError, "linked task group crosses"):
            score.validate_config(invalid)

    def test_duplicate_task_is_rejected(self):
        invalid = json.loads(json.dumps(self.config))
        invalid["tasks"].append(invalid["tasks"][0])

        with self.assertRaisesRegex(ValueError, "tasks must be unique"):
            score.validate_config(invalid)

    def test_exhaustive_manifest_detects_hidden_overlap(self):
        dataset = _Dataset([
            {"anchor": "clean query", "positive": ["clean document"]},
            {"anchor": "Hidden Evaluation Text", "positive": ["another document"]},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = tmp / "_eval_texts.json"
            cache.write_text(json.dumps({
                "queries": [
                    task._h(" hidden   evaluation text "),
                    *(task._h(f"other hidden query {index}") for index in range(10)),
                ],
                "relevant": [], "other_corpus": [], "protected": [],
            }))
            model_dir = tmp / "final_model"

            with patch.object(task, "_EVAL_CACHE", cache):
                created = task.check_contamination(dataset, model_path=model_dir, sources=["test-fixture"])
                audited = score.audit_contamination(model_dir)

            self.assertTrue(created["exhaustive"])
            self.assertEqual(created["checked_rows"], 2)
            self.assertEqual(created["hits"], 1)
            self.assertTrue(audited["valid"])
            self.assertEqual(audited["hits"], 1)
            # Detected and attributed to the query role, but query-only exposure
            # is reported rather than treated as proof of paired leakage.
            self.assertEqual(audited["query_hits"], 1)
            self.assertFalse(score.contamination_failures(audited, self.config))
            missing_sources = dict(audited, sources=[])
            self.assertTrue(score.contamination_failures(missing_sources, self.config))

    def test_same_row_query_relevant_pair_is_fatal(self):
        dataset = _Dataset([
            {"anchor": "hidden query", "positive": ["gold document"]},
        ])
        query_hash = task._h("hidden query")
        document_hash = task._h("gold document")
        pair_hash = task._pair_h(query_hash, document_hash)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = tmp / "_eval_texts.json"
            cache.write_text(json.dumps({
                "queries": [query_hash],
                "relevant": [document_hash],
                "other_corpus": [], "protected": [],
                "query_relevant_pairs": [pair_hash],
            }))
            model_dir = tmp / "final_model"

            with (
                patch.object(task, "_EVAL_CACHE", cache),
                patch.object(score, "_EVAL_CACHE", cache),
            ):
                task.check_contamination(dataset, model_path=model_dir, sources=["test-fixture"])
                audited = score.audit_contamination(model_dir)

        self.assertEqual(audited["manifest_version"], 2)
        self.assertTrue(audited["pair_audit_available"])
        self.assertEqual(audited["query_relevant_pair_hits"], 1)
        failures = score.contamination_failures(audited, self.config)
        self.assertTrue(any("co-occur" in reason for reason in failures))

    def test_small_nonrelevant_corpus_overlap_is_bounded(self):
        dataset = _Dataset([
            {"text": "incidental public corpus text"},
            {"text": "clean text"},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = tmp / "_eval_texts.json"
            cache.write_text(json.dumps({
                "queries": [], "relevant": [], "protected": [],
                "other_corpus": [task._h("incidental public corpus text")],
            }))
            model_dir = tmp / "final_model"
            with patch.object(task, "_EVAL_CACHE", cache):
                task.check_contamination(dataset, model_path=model_dir, sources=["test-fixture"])
                audited = score.audit_contamination(model_dir)

        permissive = dict(self.transfer_config)
        permissive["max_incidental_overlap_hits"] = 1
        permissive["max_incidental_overlap_fraction"] = 0.5
        self.assertEqual(audited["harmful_hits"], 0)
        self.assertEqual(audited["incidental_hits"], 1)
        self.assertFalse(score.contamination_failures(audited, permissive))

        strict = dict(permissive)
        strict["max_incidental_overlap_hits"] = 0
        strict["max_incidental_overlap_fraction"] = 0.0
        self.assertTrue(score.contamination_failures(audited, strict))

    def test_target_protocol_allows_corpus_but_not_hidden_queries(self):
        target = self.target_config
        corpus_audit = {
            "present": True, "valid": True, "query_hits": 0,
            "protected_hits": 0, "relevant_hits": 3,
            "incidental_hits": 1000, "incidental_frac": 1.0,
        }
        self.assertFalse(score.contamination_failures(corpus_audit, target))
        query_audit = dict(corpus_audit, query_hits=1)
        self.assertTrue(score.contamination_failures(query_audit, target))

    def test_sampled_manifest_is_not_reportable(self):
        dataset = _Dataset([
            {"text": "first"},
            {"text": "second"},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = tmp / "_eval_texts.json"
            cache.write_text("[]")
            model_dir = tmp / "final_model"

            with patch.object(task, "_EVAL_CACHE", cache):
                created = task.check_contamination(
                    dataset, sample=1, model_path=model_dir
                )
                audited = score.audit_contamination(model_dir)

            self.assertFalse(created["exhaustive"])
            self.assertFalse(audited["valid"])
            failures = score.contamination_failures(audited, self.config)
            self.assertTrue(any("not exhaustive" in reason for reason in failures))

    def test_missing_manifest_is_not_reportable(self):
        with tempfile.TemporaryDirectory() as tmp:
            audited = score.audit_contamination(Path(tmp) / "final_model")

        failures = score.contamination_failures(audited, self.config)
        self.assertIn("missing artifact training_manifest.json", failures)


if __name__ == "__main__":
    unittest.main()
