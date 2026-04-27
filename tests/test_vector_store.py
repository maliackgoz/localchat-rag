import json
import tempfile
import unittest
from pathlib import Path

from store.vector_store import VectorStore, build_store


class FakeEncoder:
    model_id = "fake-encoder"
    dim = 3

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "eiffel" in lowered or "paris" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "curie" in lowered or "radium" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def chunk(
    chunk_id: str,
    entity_name: str,
    type_: str,
    position: int,
    text: str,
) -> dict[str, object]:
    slug = entity_name.replace(" ", "_")
    return {
        "chunk_id": chunk_id,
        "entity_name": entity_name,
        "type": type_,
        "wikipedia_url": f"https://en.wikipedia.org/wiki/{slug}",
        "position": position,
        "n_tokens": len(text.split()),
        "text": text,
    }


class VectorStoreTests(unittest.TestCase):
    def test_query_respects_type_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = VectorStore(
                str(root / "chroma"),
                FakeEncoder(),
                manifest_path=str(root / "store_manifest.json"),
            )
            store.upsert_entity(
                "Eiffel Tower",
                "place",
                [
                    chunk("eiffel-tower__0000", "Eiffel Tower", "place", 0, "Eiffel Tower in Paris."),
                    chunk("eiffel-tower__0001", "Eiffel Tower", "place", 1, "Paris landmark with iron lattice."),
                    chunk("eiffel-tower__0002", "Eiffel Tower", "place", 2, "Visitors climb the Eiffel Tower."),
                ],
            )
            store.upsert_entity(
                "Marie Curie",
                "person",
                [chunk("marie-curie__0000", "Marie Curie", "person", 0, "Marie Curie studied radium.")],
            )

            results = store.query("Eiffel Tower", type_filter="place", k=3)

            self.assertEqual(len(results), 3)
            self.assertTrue(all(result.type == "place" for result in results))
            self.assertTrue(all(result.entity_name == "Eiffel Tower" for result in results))

    def test_build_manifest_is_idempotent_after_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunks_dir = root / "chunks"
            place_path = chunks_dir / "places" / "eiffel-tower.jsonl"
            person_path = chunks_dir / "people" / "marie-curie.jsonl"
            place_path.parent.mkdir(parents=True)
            person_path.parent.mkdir(parents=True)
            place_rows = [
                chunk("eiffel-tower__0000", "Eiffel Tower", "place", 0, "Eiffel Tower in Paris."),
                chunk("eiffel-tower__0001", "Eiffel Tower", "place", 1, "Paris landmark."),
            ]
            person_rows = [chunk("marie-curie__0000", "Marie Curie", "person", 0, "Marie Curie studied radium.")]
            place_path.write_text("".join(json.dumps(row) + "\n" for row in place_rows), encoding="utf-8")
            person_path.write_text("".join(json.dumps(row) + "\n" for row in person_rows), encoding="utf-8")

            manifest_path = root / "store_manifest.json"
            store = VectorStore(str(root / "chroma"), FakeEncoder(), manifest_path=str(manifest_path))
            first = build_store(str(chunks_dir), store)
            first_manifest = manifest_path.read_text(encoding="utf-8")
            store.reset()
            second = build_store(str(chunks_dir), store)

            self.assertEqual(first["entities_built"], 2)
            self.assertEqual(second["entities"], 2)
            self.assertEqual(first_manifest, manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(second["total_chunks"], 3)


if __name__ == "__main__":
    unittest.main()
