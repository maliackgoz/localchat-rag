"""Deterministic query routing for person/place retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


Intent = Literal["person", "place", "both"]
EntityType = Literal["person", "place"]

DEFAULT_ROSTER_PATH = "data/roster.json"

PERSON_KEYWORDS = {
    "who",
    "born",
    "discovered",
    "invented",
    "wrote",
    "played",
    "painted",
    "compose",
    "composed",
    "composer",
}
PLACE_KEYWORDS = {
    "where",
    "located",
    "country",
    "mountain",
    "river",
    "built",
    "landmark",
    "city",
}
ENTITY_ALIAS_STOPWORDS = {
    "and",
    "athens",
    "big",
    "city",
    "great",
    "house",
    "mount",
    "opera",
    "pyramids",
    "tower",
    "wall",
}


@dataclass(frozen=True)
class Roster:
    people: list[str]
    places: list[str]

    def entity_type(self, entity_name: str) -> EntityType:
        if entity_name in self.people:
            return "person"
        if entity_name in self.places:
            return "place"
        raise ValueError(f"{entity_name!r} is not in the roster")

    @property
    def entities(self) -> list[tuple[str, EntityType]]:
        return [(name, "person") for name in self.people] + [(name, "place") for name in self.places]


def load_roster(path: str = DEFAULT_ROSTER_PATH) -> Roster:
    roster_path = Path(path)
    with roster_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"{roster_path} must contain a JSON object")
    return Roster(
        people=_required_string_list(data, "people", roster_path),
        places=_required_string_list(data, "places", roster_path),
    )


def classify_intent(query: str, roster: Roster) -> tuple[Intent, list[str]]:
    matched_entities = _matched_entities(query, roster)
    if matched_entities:
        matched_types = {roster.entity_type(name) for name in matched_entities}
        if matched_types == {"person"}:
            return "person", matched_entities
        if matched_types == {"place"}:
            return "place", matched_entities
        return "both", matched_entities

    tokens = set(re.findall(r"[a-z0-9]+", query.casefold()))
    has_person_signal = bool(tokens & PERSON_KEYWORDS)
    has_place_signal = bool(tokens & PLACE_KEYWORDS)
    if has_person_signal and has_place_signal:
        return "both", []
    if has_person_signal:
        return "person", []
    if has_place_signal:
        return "place", []
    return "both", []


def _matched_entities(query: str, roster: Roster) -> list[str]:
    matched: list[str] = []
    seen: set[str] = set()
    for alias, entity_name in _entity_aliases(roster):
        if entity_name in seen:
            continue
        if _contains_phrase(query, alias):
            matched.append(entity_name)
            seen.add(entity_name)
    return matched


def _entity_aliases(roster: Roster) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    token_counts: dict[str, int] = {}
    entity_tokens: dict[str, list[str]] = {}
    for entity_name, _ in roster.entities:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", entity_name.casefold())
            if len(token) >= 4 and token not in ENTITY_ALIAS_STOPWORDS
        ]
        entity_tokens[entity_name] = tokens
        for token in set(tokens):
            token_counts[token] = token_counts.get(token, 0) + 1

    for entity_name, _ in roster.entities:
        aliases.append((entity_name, entity_name))
        for token in entity_tokens[entity_name]:
            if token_counts[token] == 1:
                aliases.append((token, entity_name))
    return sorted(aliases, key=lambda item: (-len(item[0]), item[0], item[1]))


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = re.compile(rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)")
    return bool(pattern.search(text.casefold()))


def _required_string_list(data: Mapping[str, Any], key: str, path: Path) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must contain a non-empty {key!r} list")
    items = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{path} has an invalid {key!r} item: {item!r}")
        items.append(item.strip())
    return items
