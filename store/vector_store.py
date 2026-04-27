"""Persistent Chroma vector store for local Wikipedia chunks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

import chromadb

from embedding.encoder import Encoder, get_encoder


COLLECTION_NAME = "wikipedia_rag"
DEFAULT_CHUNKS_DIR = "data/chunks"
DEFAULT_PERSIST_DIR = "data/chroma"
DEFAULT_MANIFEST_PATH = "data/store_manifest.json"
BATCH_SIZE = 64

EntityType = Literal["person", "place"]
TypeFilter = Literal["person", "place", "any"]


class ChunkInput(Protocol):
    chunk_id: str
    entity_name: str
    type: EntityType
    wikipedia_url: str
    position: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    entity_name: str
    type: EntityType
    wikipedia_url: str
    position: int
    text: str
    distance: float
    similarity: float


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    entity_name: str
    type: EntityType
    wikipedia_url: str
    position: int
    text: str


class VectorStore:
    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        encoder: Encoder | None = None,
        *,
        manifest_path: str = DEFAULT_MANIFEST_PATH,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(manifest_path)
        self.encoder = encoder or get_encoder()
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_entity(
        self,
        entity_name: str,
        type_: EntityType,
        chunks: Sequence[ChunkInput | Mapping[str, Any]],
    ) -> None:
        if type_ not in ("person", "place"):
            raise ValueError(f"type_ must be 'person' or 'place', got {type_!r}")

        records = [_coerce_chunk(chunk) for chunk in chunks]
        if not records:
            raise ValueError(f"No chunks provided for {entity_name!r}")
        for record in records:
            if record.entity_name != entity_name:
                raise ValueError(f"Chunk {record.chunk_id} belongs to {record.entity_name!r}, expected {entity_name!r}")
            if record.type != type_:
                raise ValueError(f"Chunk {record.chunk_id} has type {record.type!r}, expected {type_!r}")

        self.remove_entity(entity_name)
        for batch in _batches(records, BATCH_SIZE):
            embeddings = self.encoder.encode([record.text for record in batch])
            self._collection.add(
                ids=[record.chunk_id for record in batch],
                documents=[record.text for record in batch],
                embeddings=embeddings,
                metadatas=[_metadata(record) for record in batch],
            )

        manifest = self._read_manifest()
        manifest[entity_name] = [record.chunk_id for record in records]
        self._write_manifest(manifest)

    def remove_entity(self, entity_name: str) -> None:
        manifest = self._read_manifest()
        ids = manifest.pop(entity_name, [])
        if ids:
            self._collection.delete(ids=ids)
        else:
            existing = self._collection.get(where={"entity_name": entity_name}, include=[])
            existing_ids = _flat_list(existing.get("ids"))
            if existing_ids:
                self._collection.delete(ids=existing_ids)
        self._write_manifest(manifest)

    def query(
        self,
        text: str,
        *,
        k: int = 6,
        type_filter: TypeFilter = "any",
        entity_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k must be positive")
        if type_filter not in ("person", "place", "any"):
            raise ValueError(f"Unsupported type_filter: {type_filter!r}")
        if entity_filter == []:
            return []

        query_vector = self.encoder.encode([text])[0]
        where = _where_clause(type_filter, entity_filter)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        return _retrieved_chunks(results)

    def stats(self) -> dict[str, Any]:
        person_count = self._collection.get(where={"type": "person"}, include=[])
        place_count = self._collection.get(where={"type": "place"}, include=[])
        manifest = self._read_manifest()
        return {
            "collection": COLLECTION_NAME,
            "persist_dir": str(self.persist_dir),
            "total_chunks": self._collection.count(),
            "entities": len(manifest),
            "by_type": {
                "person": len(_flat_list(person_count.get("ids"))),
                "place": len(_flat_list(place_count.get("ids"))),
            },
            "dim": self.encoder.dim,
            "model_id": self.encoder.model_id,
        }

    def reset(self) -> None:
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._write_manifest({})

    def _read_manifest(self) -> dict[str, list[str]]:
        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"{self.manifest_path} must contain a JSON object")
        manifest: dict[str, list[str]] = {}
        for entity_name, ids in data.items():
            if not isinstance(entity_name, str) or not isinstance(ids, list):
                raise ValueError(f"{self.manifest_path} must map entity names to chunk id lists")
            manifest[entity_name] = [str(chunk_id) for chunk_id in ids]
        return manifest

    def _write_manifest(self, manifest: dict[str, list[str]]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dict(sorted(manifest.items())), indent=2, sort_keys=True) + "\n"
        self.manifest_path.write_text(payload, encoding="utf-8")


def build_store(chunks_dir: str, store: VectorStore) -> dict[str, Any]:
    entity_count = 0
    chunk_count = 0
    for chunk_path in _chunk_files(Path(chunks_dir)):
        records = _read_chunk_file(chunk_path)
        if not records:
            continue
        first = records[0]
        store.upsert_entity(first.entity_name, first.type, records)
        entity_count += 1
        chunk_count += len(records)
    stats = store.stats()
    return {"entities_built": entity_count, "chunks_built": chunk_count, **stats}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and query the local Chroma vector store.")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_DIR, help="Chunk JSONL root.")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Chroma persistence directory.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH, help="Store manifest JSON path.")
    parser.add_argument("--encoder", choices=("sentence_transformers", "ollama"), default="sentence_transformers")
    parser.add_argument("--model-id", help="Override the encoder model id.")
    parser.add_argument("--build", action="store_true", help="Upsert all chunk files into Chroma.")
    parser.add_argument("--stats", action="store_true", help="Print store statistics.")
    parser.add_argument("--query", help="Run a similarity query.")
    parser.add_argument("--type", choices=("person", "place", "any"), default="any", help="Query type filter.")
    parser.add_argument("--entity", action="append", help="Restrict query to an entity name; repeat for multiple.")
    parser.add_argument("--k", type=int, default=6, help="Number of query results.")
    parser.add_argument("--reset", action="store_true", help="Clear the collection and manifest before other actions.")
    args = parser.parse_args(argv)

    encoder = get_encoder(args.encoder, model_id=args.model_id)
    store = VectorStore(args.persist_dir, encoder, manifest_path=args.manifest)

    if args.reset:
        store.reset()
        print("vector store reset")
    if args.build:
        report = build_store(args.chunks, store)
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.stats:
        print(json.dumps(store.stats(), indent=2, sort_keys=True))
    if args.query:
        results = store.query(args.query, k=args.k, type_filter=args.type, entity_filter=args.entity)
        for index, chunk in enumerate(results, start=1):
            print(f"{index}. {chunk.entity_name} [{chunk.type}] pos={chunk.position} sim={chunk.similarity:.3f}")
            print(chunk.text[:500].replace("\n", " "))
            print()
    if not any((args.reset, args.build, args.stats, args.query)):
        parser.error("pass at least one action: --build, --stats, --query, or --reset")
    return 0


def _chunk_files(root: Path) -> list[Path]:
    return sorted(root.glob("people/*.jsonl")) + sorted(root.glob("places/*.jsonl"))


def _read_chunk_file(path: Path) -> list[StoredChunk]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(_coerce_chunk(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
    return records


def _coerce_chunk(chunk: ChunkInput | Mapping[str, Any]) -> StoredChunk:
    if isinstance(chunk, Mapping):
        value = chunk
        return StoredChunk(
            chunk_id=_required_string(value, "chunk_id"),
            entity_name=_required_string(value, "entity_name"),
            type=_required_type(value),
            wikipedia_url=_required_string(value, "wikipedia_url"),
            position=_required_int(value, "position"),
            text=_required_string(value, "text"),
        )
    return StoredChunk(
        chunk_id=chunk.chunk_id,
        entity_name=chunk.entity_name,
        type=chunk.type,
        wikipedia_url=chunk.wikipedia_url,
        position=chunk.position,
        text=chunk.text,
    )


def _metadata(record: StoredChunk) -> dict[str, str | int]:
    return {
        "type": record.type,
        "entity_name": record.entity_name,
        "wikipedia_url": record.wikipedia_url,
        "position": record.position,
    }


def _where_clause(type_filter: TypeFilter, entity_filter: list[str] | None) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if type_filter != "any":
        clauses.append({"type": type_filter})
    if entity_filter:
        if len(entity_filter) == 1:
            clauses.append({"entity_name": entity_filter[0]})
        else:
            clauses.append({"entity_name": {"$in": entity_filter}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _retrieved_chunks(results: Mapping[str, Any]) -> list[RetrievedChunk]:
    ids = _first_result_list(results.get("ids"))
    documents = _first_result_list(results.get("documents"))
    metadatas = _first_result_list(results.get("metadatas"))
    distances = _first_result_list(results.get("distances"))
    chunks = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        if not isinstance(metadata, Mapping):
            raise ValueError("Chroma returned a result without metadata")
        distance_value = float(distance)
        chunks.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                entity_name=_required_string(metadata, "entity_name"),
                type=_required_type(metadata),
                wikipedia_url=_required_string(metadata, "wikipedia_url"),
                position=_required_int(metadata, "position"),
                text=str(document),
                distance=distance_value,
                similarity=1.0 - distance_value,
            )
        )
    return chunks


def _batches(records: Sequence[StoredChunk], size: int) -> list[Sequence[StoredChunk]]:
    return [records[index : index + size] for index in range(0, len(records), size)]


def _flat_list(value: object) -> list[Any]:
    if not isinstance(value, list):
        return []
    if value and isinstance(value[0], list):
        return [item for group in value for item in group]
    return value


def _first_result_list(value: object) -> list[Any]:
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    return first if isinstance(first, list) else value


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Chunk is missing non-empty {key!r}")
    return item.strip()


def _required_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int):
        raise ValueError(f"Chunk is missing integer {key!r}")
    return item


def _required_type(value: Mapping[str, Any]) -> EntityType:
    item = value.get("type")
    if item not in ("person", "place"):
        raise ValueError(f"Chunk has invalid type: {item!r}")
    return item


if __name__ == "__main__":
    raise SystemExit(main())
