"""Fetch and cache raw Wikipedia documents for the project roster."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


EntityType = Literal["person", "place"]

SUMMARY_ENDPOINT = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
EXTRACT_ENDPOINT = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "localchat-rag/1.0 (educational)"
THROTTLE_SECONDS = 0.15


class WikipediaIngestError(RuntimeError):
    """Base error for ingestion failures."""


class DisambiguationError(WikipediaIngestError):
    """Raised when Wikipedia resolves an entity to a disambiguation page."""


class WikipediaFetchError(WikipediaIngestError):
    """Raised when a Wikipedia API request fails or returns unusable data."""


@dataclass(frozen=True)
class RawDocument:
    entity_name: str
    type: EntityType
    wikipedia_url: str
    fetched_at: str
    title: str
    summary: str
    text: str


@dataclass(frozen=True)
class IngestReport:
    fetched: int
    cached: int
    total: int
    manifest_path: str


def fetch_entity(name: str, entity_type: EntityType) -> RawDocument:
    """Fetch one Wikipedia entity and return its raw document payload."""
    if entity_type not in ("person", "place"):
        raise ValueError(f"Unsupported entity type: {entity_type!r}")

    summary = _fetch_summary(name)
    _guard_summary_disambiguation(name, summary)

    title = str(summary.get("title") or name).strip()
    if not title:
        raise WikipediaFetchError(f"Wikipedia summary for {name!r} did not include a title")

    full_page = _fetch_full_page(title)
    page_title = str(full_page.get("title") or title).strip()
    extract = _clean_text(str(full_page.get("extract") or ""))
    if not extract:
        raise WikipediaFetchError(f"Wikipedia article for {name!r} has empty text")

    _guard_extract_disambiguation(name, page_title, extract)

    summary_text = _clean_text(str(summary.get("extract") or ""))
    return RawDocument(
        entity_name=name,
        type=entity_type,
        wikipedia_url=_canonical_url(summary, page_title),
        fetched_at=_utc_now(),
        title=page_title,
        summary=summary_text,
        text=extract,
    )


def ingest_roster(roster_path: str, out_dir: str, *, force: bool = False) -> IngestReport:
    """Ingest every entity in the roster into ``out_dir`` idempotently."""
    roster = _load_roster(Path(roster_path))
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "people").mkdir(exist_ok=True)
    (output_root / "places").mkdir(exist_ok=True)

    fetched = 0
    cached = 0
    manifest_entries: list[dict[str, Any]] = []

    for entity_type, names in (("person", roster["people"]), ("place", roster["places"])):
        for name in names:
            path = _document_path(output_root, name, entity_type)
            if path.exists() and not force:
                document = _read_document(path)
                cached += 1
            else:
                document = fetch_entity(name, entity_type)
                _write_json_if_changed(path, asdict(document))
                fetched += 1

            manifest_entries.append(_manifest_entry(output_root, document))

    manifest_path = output_root / "_manifest.json"
    manifest = {"entities": sorted(manifest_entries, key=lambda row: (row["type"], row["entity_name"]))}
    _write_json_if_changed(manifest_path, manifest)
    return IngestReport(
        fetched=fetched,
        cached=cached,
        total=fetched + cached,
        manifest_path=str(manifest_path),
    )


def slugify(name: str) -> str:
    """Return a stable filesystem slug for a Wikipedia entity name."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    if not slug:
        raise ValueError(f"Cannot create slug for empty entity name: {name!r}")
    return slug


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and cache Wikipedia roster documents.")
    parser.add_argument("--roster", default="data/roster.json", help="Path to roster JSON.")
    parser.add_argument("--out", default="data/raw", help="Output directory for raw documents.")
    parser.add_argument("--force", action="store_true", help="Refetch every roster entity.")
    parser.add_argument("--refresh", help="Refetch one entity instead of the full roster.")
    parser.add_argument("--type", choices=("person", "place"), help="Entity type for --refresh if not in roster.")
    args = parser.parse_args(argv)

    if args.refresh:
        entity_type = args.type or _find_entity_type(Path(args.roster), args.refresh)
        output_root = Path(args.out)
        output_root.mkdir(parents=True, exist_ok=True)
        _document_subdir(output_root, entity_type).mkdir(exist_ok=True)
        document = fetch_entity(args.refresh, entity_type)
        _write_json_if_changed(_document_path(output_root, args.refresh, entity_type), asdict(document))
        _rebuild_manifest(output_root)
        print(f"1 fetched, 0 cached ({args.refresh})")
        return 0

    report = ingest_roster(args.roster, args.out, force=args.force)
    print(f"{report.fetched} fetched, {report.cached} cached")
    print(f"manifest: {report.manifest_path}")
    return 0


def _fetch_summary(name: str) -> dict[str, Any]:
    title = urllib.parse.quote(name.replace(" ", "_"), safe="")
    return _get_json(SUMMARY_ENDPOINT.format(title=title))


def _fetch_full_page(title: str) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": title,
            "format": "json",
        }
    )
    data = _get_json(f"{EXTRACT_ENDPOINT}?{params}")
    pages = data.get("query", {}).get("pages", {})
    if not isinstance(pages, dict) or not pages:
        raise WikipediaFetchError(f"Wikipedia extract response for {title!r} did not include pages")

    page = next(iter(pages.values()))
    if not isinstance(page, dict):
        raise WikipediaFetchError(f"Wikipedia extract response for {title!r} was malformed")
    if "missing" in page:
        raise WikipediaFetchError(f"Wikipedia article {title!r} does not exist")
    return page


def _get_json(url: str, *, retries: int = 3) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                payload = json.loads(response.read().decode(charset))
                if not isinstance(payload, dict):
                    raise WikipediaFetchError(f"Wikipedia returned non-object JSON for {url}")
                time.sleep(THROTTLE_SECONDS)
                return payload
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(THROTTLE_SECONDS * (2 ** (attempt + 1)))
    raise WikipediaFetchError(f"Wikipedia request failed for {url}: {last_error}") from last_error


def _guard_summary_disambiguation(name: str, summary: dict[str, Any]) -> None:
    if summary.get("type") == "disambiguation":
        raise DisambiguationError(f"Wikipedia entity {name!r} resolved to a disambiguation page")
    title = str(summary.get("title") or "")
    if title.lower().endswith("(disambiguation)"):
        raise DisambiguationError(f"Wikipedia entity {name!r} resolved to {title!r}")


def _guard_extract_disambiguation(name: str, title: str, text: str) -> None:
    lowered_title = title.lower()
    first_section = text[:800].lower()
    if lowered_title.endswith("(disambiguation)") or "may refer to:" in first_section:
        raise DisambiguationError(f"Wikipedia entity {name!r} resolved to a disambiguation page")


def _clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith(("{|", "|}", "|-", "{{", "}}")):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def _canonical_url(summary: dict[str, Any], title: str) -> str:
    page_url = summary.get("content_urls", {}).get("desktop", {}).get("page")
    if isinstance(page_url, str) and page_url:
        return page_url
    quoted_title = urllib.parse.quote(title.replace(" ", "_"), safe="()")
    return f"https://en.wikipedia.org/wiki/{quoted_title}"


def _load_roster(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    people = data.get("people")
    places = data.get("places")
    if not isinstance(people, list) or not isinstance(places, list):
        raise ValueError(f"Roster {path} must contain 'people' and 'places' lists")
    return {
        "people": _validate_names(people, "people"),
        "places": _validate_names(places, "places"),
    }


def _validate_names(values: list[Any], key: str) -> list[str]:
    names = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Roster key {key!r} contains a non-empty string requirement violation")
        names.append(value.strip())
    return names


def _find_entity_type(roster_path: Path, name: str) -> EntityType:
    roster = _load_roster(roster_path)
    if name in roster["people"]:
        return "person"
    if name in roster["places"]:
        return "place"
    raise ValueError(f"{name!r} is not in {roster_path}; pass --type person|place for --refresh")


def _document_path(output_root: Path, name: str, entity_type: EntityType) -> Path:
    return _document_subdir(output_root, entity_type) / f"{slugify(name)}.json"


def _document_subdir(output_root: Path, entity_type: EntityType) -> Path:
    return output_root / ("people" if entity_type == "person" else "places")


def _read_document(path: Path) -> RawDocument:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    document = RawDocument(**data)
    if not document.text.strip():
        raise WikipediaFetchError(f"Cached document {path} has empty text; rerun with --force")
    return document


def _manifest_entry(output_root: Path, document: RawDocument) -> dict[str, Any]:
    path = _document_path(output_root, document.entity_name, document.type)
    return {
        "entity_name": document.entity_name,
        "type": document.type,
        "slug": slugify(document.entity_name),
        "path": path.relative_to(output_root).as_posix(),
        "sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
        "fetched_at": document.fetched_at,
        "wikipedia_url": document.wikipedia_url,
        "title": document.title,
        "text_chars": len(document.text),
    }


def _rebuild_manifest(output_root: Path) -> None:
    entries = []
    for entity_type in ("person", "place"):
        for path in sorted(_document_subdir(output_root, entity_type).glob("*.json")):
            entries.append(_manifest_entry(output_root, _read_document(path)))
    manifest = {"entities": sorted(entries, key=lambda row: (row["type"], row["entity_name"]))}
    _write_json_if_changed(output_root / "_manifest.json", manifest)


def _write_json_if_changed(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == encoded:
        return
    path.write_text(encoded, encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
