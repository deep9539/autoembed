import os
import shlex
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import mteb

import task


class SubmissionProtocolTest(unittest.TestCase):
    def test_custom_encode_only_model_is_adapted_to_encoder_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "mteb_model.py").write_text(textwrap.dedent("""
                class Encoder:
                    def __init__(self, model_path):
                        self.model_path = model_path

                    def encode(self, inputs, *, task_metadata, hf_split, hf_subset,
                               prompt_type=None, **kwargs):
                        return [[0.0, 1.0] for _ in inputs]

                def load_model(model_path):
                    return Encoder(model_path)
            """))

            model = task.load_encoder(model_dir)

            self.assertIsInstance(model, mteb.EncoderProtocol)
            self.assertEqual(model.model.model_path, str(model_dir))
            self.assertEqual(model.mteb_model_meta.model_type, ["dense"])


    def test_custom_encoder_can_run_through_isolated_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "mteb_model.py").write_text(textwrap.dedent("""
                import numpy as np

                class Encoder:
                    def encode(self, inputs, **kwargs):
                        return np.asarray([[float(i), 1.0] for i, _ in enumerate(inputs)])

                    def similarity(self, a, b):
                        return np.asarray(a) @ np.asarray(b).T

                    def similarity_pairwise(self, a, b):
                        return (np.asarray(a) * np.asarray(b)).sum(axis=1)

                def load_model(model_path):
                    return Encoder()
            """))
            command = (
                f"{shlex.quote(sys.executable)} -m autoembed.encoder_worker "
                f"{shlex.quote(str(model_dir))}"
            )
            with patch.dict(os.environ, {
                "AUTOEMBED_ENCODER_WORKER_COMMAND": command,
                "AUTOEMBED_REQUIRE_ISOLATED_CUSTOM": "1",
            }):
                model = task.load_encoder(model_dir)
                embeddings = model.encode(
                    ["one", "two"], task_metadata=None, hf_split="test",
                    hf_subset="default",
                )
                similarity = model.similarity(embeddings, embeddings)
                pairwise = model.similarity_pairwise(embeddings, embeddings)

            self.assertEqual(embeddings.shape, (2, 2))
            self.assertEqual(similarity.tolist(), [[1.0, 1.0], [1.0, 2.0]])
            self.assertEqual(pairwise.tolist(), [1.0, 2.0])

    def test_required_custom_isolation_fails_closed_without_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "mteb_model.py").write_text("def load_model(path): return None\n")
            with patch.dict(os.environ, {
                "AUTOEMBED_REQUIRE_ISOLATED_CUSTOM": "1",
                "AUTOEMBED_ENCODER_WORKER_COMMAND": "",
            }):
                with self.assertRaisesRegex(RuntimeError, "isolated encoder worker"):
                    task.load_encoder(model_dir)

    def test_standard_folder_falls_back_to_sentence_transformer(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = object()
            with patch("sentence_transformers.SentenceTransformer", return_value=sentinel) as load:
                model = task.load_encoder(tmp)

            self.assertIs(model, sentinel)
            load.assert_called_once_with(tmp, trust_remote_code=False)

    def test_custom_entrypoint_requires_factory(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "mteb_model.py").write_text("VALUE = 1\n")

            with self.assertRaisesRegex(AttributeError, "load_model"):
                task.load_encoder(model_dir)

    def test_custom_encoder_rejects_multi_vector_output(self):
        class MultiVectorEncoder:
            def encode(self, inputs, **kwargs):
                return [[[0.0, 1.0]] for _ in inputs]

        model = task._as_mteb_encoder(MultiVectorEncoder(), "/tmp/final_model")
        with self.assertRaisesRegex(ValueError, "one dense"):
            model.encode(
                ["one", "two"],
                task_metadata=None,
                hf_split="test",
                hf_subset="default",
            )

    def test_custom_encoder_requires_stable_dimension(self):
        class ChangingEncoder:
            def __init__(self):
                self.dimension = 2

            def encode(self, inputs, **kwargs):
                return [[0.0] * self.dimension for _ in inputs]

        raw = ChangingEncoder()
        model = task._as_mteb_encoder(raw, "/tmp/final_model")
        model.encode(
            ["one"], task_metadata=None, hf_split="test", hf_subset="default"
        )
        raw.dimension = 3
        with self.assertRaisesRegex(ValueError, "dimension changed"):
            model.encode(
                ["two"], task_metadata=None, hf_split="test", hf_subset="default"
            )

    def test_search_protocol_is_rejected(self):
        class Search:
            mteb_model_meta = None

            def index(self, corpus, **kwargs):
                return None

            def search(self, corpus, queries, top_k, **kwargs):
                return {}

        with self.assertRaisesRegex(TypeError, "dense MTEB encoders"):
            task._as_mteb_encoder(Search(), "/tmp/final_model")


if __name__ == "__main__":
    unittest.main()
