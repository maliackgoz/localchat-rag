import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from ingest.wikipedia import DisambiguationError, RawDocument, fetch_entity, ingest_roster


class WikipediaIngestionTests(unittest.TestCase):
    def test_disambiguation_summary_fails_loudly(self) -> None:
        with patch(
            "ingest.wikipedia._fetch_summary",
            return_value={"type": "disambiguation", "title": "Mercury"},
        ):
            with self.assertRaisesRegex(DisambiguationError, "disambiguation"):
                fetch_entity("Mercury", "place")

    def test_roster_ingest_uses_cached_document_without_fetching(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            roster_path = root / "roster.json"
            out_dir = root / "raw"
            roster_path.write_text(json.dumps({"people": ["Ada Lovelace"], "places": []}), encoding="utf-8")

            cached = RawDocument(
                entity_name="Ada Lovelace",
                type="person",
                wikipedia_url="https://en.wikipedia.org/wiki/Ada_Lovelace",
                fetched_at="2026-04-27T20:30:00Z",
                title="Ada Lovelace",
                summary="English mathematician.",
                text="Ada Lovelace was an English mathematician and writer.",
            )
            cached_path = out_dir / "people" / "ada-lovelace.json"
            cached_path.parent.mkdir(parents=True)
            cached_path.write_text(json.dumps(asdict(cached), indent=2), encoding="utf-8")

            with patch("ingest.wikipedia.fetch_entity") as fetch:
                report = ingest_roster(str(roster_path), str(out_dir))

            fetch.assert_not_called()
            self.assertEqual(report.fetched, 0)
            self.assertEqual(report.cached, 1)

            manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["entities"][0]["entity_name"], "Ada Lovelace")
            self.assertEqual(len(manifest["entities"][0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
