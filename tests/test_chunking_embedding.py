import json
import math
import tempfile
import unittest
from pathlib import Path

from chunking.splitter import chunk_entity_file, chunk_raw_documents, split_document
from embedding.encoder import normalize_vectors, vector_norm


class ChunkingTests(unittest.TestCase):
    def test_split_document_uses_sentence_overlap(self) -> None:
        text = "Alpha beta. Gamma delta. Epsilon zeta. Eta theta."

        chunks = split_document(text, target_tokens=5, overlap=2)

        self.assertEqual([chunk.text for chunk in chunks], ["Alpha beta. Gamma delta.", "Gamma delta. Epsilon zeta.", "Epsilon zeta. Eta theta."])
        self.assertTrue(all(chunk.n_tokens <= 5 for chunk in chunks))

    def test_chunk_entity_file_writes_prd_record_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ada-lovelace.json"
            path.write_text(
                json.dumps(
                    {
                        "entity_name": "Ada Lovelace",
                        "type": "person",
                        "wikipedia_url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
                        "text": "Ada wrote notes. She studied engines. Her work is remembered.",
                    }
                ),
                encoding="utf-8",
            )

            records = chunk_entity_file(path, "person", target_tokens=12, overlap=2)

            self.assertEqual(records[0].chunk_id, "ada-lovelace__0000")
            self.assertEqual(records[0].entity_name, "Ada Lovelace")
            self.assertEqual(records[0].type, "person")
            self.assertEqual(records[0].position, 0)
            self.assertTrue(records[0].text.startswith("Ada Lovelace — "))
            self.assertEqual(records[0].n_tokens, len(records[0].text.split()))
            self.assertLessEqual(records[0].n_tokens, 12)

    def test_chunk_raw_documents_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw" / "people" / "ada-lovelace.json"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_text(
                json.dumps(
                    {
                        "entity_name": "Ada Lovelace",
                        "type": "person",
                        "wikipedia_url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
                        "text": "Ada wrote notes. She studied engines. Her work is remembered.",
                    }
                ),
                encoding="utf-8",
            )

            first = chunk_raw_documents(str(root / "raw"), str(root / "chunks"), target_tokens=12, overlap=2)
            out_path = root / "chunks" / "people" / "ada-lovelace.jsonl"
            first_payload = out_path.read_text(encoding="utf-8")
            second = chunk_raw_documents(str(root / "raw"), str(root / "chunks"), target_tokens=12, overlap=2)

            self.assertEqual(first.entities, 1)
            self.assertEqual(second.entities, 1)
            self.assertEqual(first_payload, out_path.read_text(encoding="utf-8"))


class EncoderTests(unittest.TestCase):
    def test_normalize_vectors_returns_unit_vectors(self) -> None:
        vectors = normalize_vectors([[3.0, 4.0], [1.0, 1.0]])

        self.assertAlmostEqual(vector_norm(vectors[0]), 1.0, places=6)
        self.assertAlmostEqual(vector_norm(vectors[1]), 1.0, places=6)
        self.assertTrue(math.isclose(vectors[0][0], 0.6))

    def test_normalize_vectors_rejects_zero_vector(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-length"):
            normalize_vectors([[0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
