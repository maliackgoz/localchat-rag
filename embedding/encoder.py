"""Local embedding encoder interfaces for MiniLM and Ollama."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal, Protocol


Backend = Literal["sentence_transformers", "ollama"]
DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    @property
    def dim(self) -> int:
        """Embedding vector dimension."""

    @property
    def model_id(self) -> str:
        """Stable model identifier for manifests and diagnostics."""


class SentenceTransformersEncoder:
    def __init__(self, model_id: str = DEFAULT_SENTENCE_TRANSFORMERS_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the default encoder. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self._model_id = model_id
        self._model = SentenceTransformer(model_id)
        model_dim = self._model.get_sentence_embedding_dimension()
        self._dim = int(model_dim) if model_dim is not None else 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = _to_float_rows(embeddings)
        normalized = normalize_vectors(vectors)
        if normalized and self._dim == 0:
            self._dim = len(normalized[0])
        return normalized

    @property
    def dim(self) -> int:
        if self._dim == 0:
            self.encode(["dimension probe"])
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_id


class OllamaEncoder:
    def __init__(self, model_id: str = DEFAULT_OLLAMA_MODEL) -> None:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                "ollama is required for the Ollama encoder. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self._model_id = model_id
        self._client = ollama.Client()
        self._dim = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embed(model=self._model_id, input=texts)
        except Exception as exc:
            raise RuntimeError(
                "Ollama embedding request failed. Ensure the Ollama daemon is running "
                f"and `{self._model_id}` is pulled."
            ) from exc

        embeddings = _response_value(response, "embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Ollama embedding response did not include an embeddings list")
        normalized = normalize_vectors(_to_float_rows(embeddings))
        if normalized and self._dim == 0:
            self._dim = len(normalized[0])
        return normalized

    @property
    def dim(self) -> int:
        if self._dim == 0:
            self.encode(["dimension probe"])
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_id


def get_encoder(
    backend: Backend = "sentence_transformers",
    *,
    model_id: str | None = None,
) -> Encoder:
    if backend == "sentence_transformers":
        return SentenceTransformersEncoder(model_id or DEFAULT_SENTENCE_TRANSFORMERS_MODEL)
    if backend == "ollama":
        return OllamaEncoder(model_id or DEFAULT_OLLAMA_MODEL)
    raise ValueError(f"Unsupported encoder backend: {backend!r}")


def normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    normalized = []
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise ValueError("Cannot normalize a zero-length embedding vector")
        normalized.append([value / norm for value in vector])
    return normalized


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe local embedding encoders.")
    parser.add_argument("--probe", action="store_true", help="Print model id, dim, norm, and chunk stats.")
    parser.add_argument(
        "--backend",
        choices=("sentence_transformers", "ollama"),
        default="sentence_transformers",
    )
    parser.add_argument("--model-id", help="Override the backend model id.")
    parser.add_argument("--chunks", default="data/chunks", help="Chunk directory for average stats.")
    args = parser.parse_args(argv)

    if not args.probe:
        parser.error("pass --probe to run the encoder probe")

    encoder = get_encoder(args.backend, model_id=args.model_id)
    sample = "Ada Lovelace wrote about Charles Babbage's analytical engine."
    vector = encoder.encode([sample])[0]
    print(f"model_id: {encoder.model_id}")
    print(f"dim: {encoder.dim}")
    print(f"sample_vector_norm: {vector_norm(vector):.6f}")

    stats = chunk_stats(Path(args.chunks))
    if stats["entities"] > 0:
        print(f"chunk_entities: {stats['entities']}")
        print(f"chunk_count: {stats['chunks']}")
        print(f"average_chunks_per_entity: {stats['average_chunks_per_entity']:.2f}")
    else:
        print(f"chunk_entities: 0")
    return 0


def chunk_stats(root: Path) -> dict[str, float | int]:
    entity_files = sorted(root.glob("people/*.jsonl")) + sorted(root.glob("places/*.jsonl"))
    chunks = 0
    for path in entity_files:
        with path.open("r", encoding="utf-8") as handle:
            chunks += sum(1 for line in handle if line.strip())
    entities = len(entity_files)
    average = chunks / entities if entities else 0.0
    return {"entities": entities, "chunks": chunks, "average_chunks_per_entity": average}


def _to_float_rows(values: object) -> list[list[float]]:
    rows = values.tolist() if hasattr(values, "tolist") else values
    if not isinstance(rows, list):
        raise TypeError("Embedding output must be a list of vectors")

    converted: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list):
            raise TypeError("Embedding output must be a list of vectors")
        converted.append([float(value) for value in row])
    return converted


def _response_value(response: object, key: str) -> object:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


if __name__ == "__main__":
    raise SystemExit(main())
