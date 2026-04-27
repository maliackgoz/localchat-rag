"""Sentence-aware sliding-window chunking for raw Wikipedia documents."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


EntityType = Literal["person", "place"]
DEFAULT_TARGET_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 60
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass(frozen=True)
class Chunk:
    position: int
    n_tokens: int
    text: str


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    entity_name: str
    type: EntityType
    wikipedia_url: str
    position: int
    n_tokens: int
    text: str


@dataclass(frozen=True)
class ChunkingReport:
    entities: int
    chunks: int
    output_dir: str

    @property
    def average_chunks_per_entity(self) -> float:
        if self.entities == 0:
            return 0.0
        return self.chunks / self.entities


def split_document(
    text: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split ``text`` into sentence-aware overlapping token windows."""
    _validate_window(target_tokens, overlap)
    units = _sentence_units(text, target_tokens)
    chunks: list[Chunk] = []
    start = 0

    while start < len(units):
        end = start
        token_total = 0
        chunk_parts: list[str] = []

        while end < len(units):
            unit_text, unit_tokens = units[end]
            if chunk_parts and token_total + unit_tokens > target_tokens:
                break
            chunk_parts.append(unit_text)
            token_total += unit_tokens
            end += 1

        chunk_text = _join_chunk_parts(chunk_parts)
        if chunk_text:
            chunks.append(Chunk(position=len(chunks), n_tokens=token_total, text=chunk_text))

        if end >= len(units):
            break
        start = _next_start(units, start, end, overlap)

    return chunks


def chunk_raw_documents(
    raw_dir: str,
    out_dir: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
) -> ChunkingReport:
    """Write chunk JSONL files for every raw document under ``raw_dir``."""
    _validate_window(target_tokens, overlap)
    raw_root = Path(raw_dir)
    output_root = Path(out_dir)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw document directory does not exist: {raw_root}")

    entities = 0
    chunk_count = 0
    for entity_type in ("person", "place"):
        input_dir = raw_root / _plural_type(entity_type)
        if not input_dir.exists():
            continue
        for raw_path in sorted(input_dir.glob("*.json")):
            records = chunk_entity_file(
                raw_path,
                entity_type,
                target_tokens=target_tokens,
                overlap=overlap,
            )
            out_path = output_root / _plural_type(entity_type) / f"{raw_path.stem}.jsonl"
            _write_jsonl_if_changed(out_path, [asdict(record) for record in records])
            entities += 1
            chunk_count += len(records)

    return ChunkingReport(entities=entities, chunks=chunk_count, output_dir=str(output_root))


def chunk_entity_file(
    path: Path,
    entity_type: EntityType,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
) -> list[ChunkRecord]:
    """Build chunk records for one raw Wikipedia JSON document."""
    document = _read_raw_document(path)
    expected_type = document.get("type")
    if expected_type != entity_type:
        raise ValueError(f"{path} has type {expected_type!r}, expected {entity_type!r}")

    entity_name = _required_string(document, "entity_name", path)
    wikipedia_url = _required_string(document, "wikipedia_url", path)
    text = _required_string(document, "text", path)
    slug = slugify(entity_name)
    prefix = f"{entity_name} — "
    body_target_tokens = max(1, target_tokens - count_tokens(prefix))

    records = []
    for chunk in split_document(text, target_tokens=body_target_tokens, overlap=overlap):
        stored_text = f"{prefix}{chunk.text}"
        records.append(
            ChunkRecord(
                chunk_id=f"{slug}__{chunk.position:04}",
                entity_name=entity_name,
                type=entity_type,
                wikipedia_url=wikipedia_url,
                position=chunk.position,
                n_tokens=count_tokens(stored_text),
                text=stored_text,
            )
        )
    if not records:
        raise ValueError(f"{path} produced no chunks")
    return records


def count_tokens(text: str) -> int:
    """Return the whitespace-token count used by the chunking contract."""
    return len(text.split())


def slugify(name: str) -> str:
    """Return a stable filesystem slug for an entity name."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    if not slug:
        raise ValueError(f"Cannot create slug for empty entity name: {name!r}")
    return slug


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk raw Wikipedia documents into JSONL files.")
    parser.add_argument("--raw", default="data/raw", help="Raw document directory.")
    parser.add_argument("--out", default="data/chunks", help="Output directory for chunk JSONL files.")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_TOKENS)
    args = parser.parse_args(argv)

    report = chunk_raw_documents(
        args.raw,
        args.out,
        target_tokens=args.target_tokens,
        overlap=args.overlap,
    )
    print(f"{report.entities} entities chunked")
    print(f"{report.chunks} chunks written")
    print(f"average chunks/entity: {report.average_chunks_per_entity:.2f}")
    print(f"output: {report.output_dir}")
    return 0


def _validate_window(target_tokens: int, overlap: int) -> None:
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= target_tokens:
        raise ValueError("overlap must be smaller than target_tokens")


def _sentence_units(text: str, target_tokens: int) -> list[tuple[str, int]]:
    units: list[tuple[str, int]] = []
    for paragraph in re.split(r"\n{2,}", text.strip()):
        normalized = " ".join(paragraph.split())
        if not normalized:
            continue
        for sentence in SENTENCE_BOUNDARY_RE.split(normalized):
            stripped = sentence.strip()
            if not stripped:
                continue
            units.extend(_split_oversized_sentence(stripped, target_tokens))
    return units


def _split_oversized_sentence(sentence: str, target_tokens: int) -> list[tuple[str, int]]:
    tokens = sentence.split()
    if len(tokens) <= target_tokens:
        return [(sentence, len(tokens))]
    return [
        (" ".join(tokens[index : index + target_tokens]), len(tokens[index : index + target_tokens]))
        for index in range(0, len(tokens), target_tokens)
    ]


def _join_chunk_parts(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def _next_start(
    units: list[tuple[str, int]],
    start: int,
    end: int,
    overlap: int,
) -> int:
    if overlap == 0:
        return end

    overlap_tokens = 0
    next_start = end
    while next_start > start and overlap_tokens < overlap:
        next_start -= 1
        overlap_tokens += units[next_start][1]
    return max(start + 1, next_start)


def _read_raw_document(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _required_string(document: dict[str, Any], key: str, path: Path) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} is missing non-empty {key!r}")
    return value.strip()


def _write_jsonl_if_changed(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") == encoded:
        return
    path.write_text(encoded, encoding="utf-8")


def _plural_type(entity_type: EntityType) -> str:
    return "people" if entity_type == "person" else "places"


if __name__ == "__main__":
    raise SystemExit(main())
