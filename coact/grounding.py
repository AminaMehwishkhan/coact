"""Parsing an instruction into the constraints an agent can ground.

This module used to hold a world-model grounder too; that moved to
`coact.agent`, which grounds against detections in each camera's own image
instead of against the simulator's object list. What is left here is the
language side: turning an utterance into a constraint set, and merging a
spoken correction into the constraints already in play.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

SPATIAL_TERMS = {
    "left": "left",
    "leftmost": "left",
    "right": "right",
    "rightmost": "right",
    "near": "near",
    "nearest": "near",
    "closest": "near",
    "far": "far",
    "furthest": "far",
    "farthest": "far",
}
SIZE_TERMS = {"small": "small", "little": "small", "large": "large", "big": "large"}
COLOR_TERMS = ["red", "blue", "white", "green", "grey", "gray", "black"]
NOUNS = ["plate", "cup", "bowl", "glass"]


@dataclass
class Constraints:
    noun: str | None = None
    color: str | None = None
    size: str | None = None
    spatial: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def merged_with(self, other: "Constraints") -> "Constraints":
        """A correction overrides; unspecified fields are inherited."""
        return Constraints(
            noun=other.noun or self.noun,
            color=other.color or self.color,
            size=other.size or self.size,
            # An explicit attribute correction retires the spatial term that
            # caused the ambiguity in the first place.
            spatial=None if (other.size or other.color) else (other.spatial or self.spatial),
        )

    def summary(self) -> str:
        bits = [v for v in (self.size, self.color, self.spatial) if v]
        return " ".join(bits + [self.noun or "object"]).strip()


def parse_constraints(text: str, target_role: str = "destination") -> Constraints:
    """Extract the constraint set for the destination (the ambiguous part).

    "Put the cup on the plate on the left" -> destination clause is everything
    after the last ' on '. "No, the large plate" -> whole utterance.
    """
    t = text.lower().strip()
    clause = t
    if target_role == "destination":
        # Split on the FIRST 'on' only. "put the cup on the plate on the left"
        # -> "the plate on the left", which keeps both the head noun and the
        # spatial modifier. Splitting on the last 'on' loses the noun.
        parts = re.split(r"\bon\b", t, maxsplit=1)
        if len(parts) > 1:
            clause = parts[1]
    tokens = re.findall(r"[a-z]+", clause)

    noun = next((n for n in NOUNS if n in tokens), None)
    color = next((c for c in COLOR_TERMS if c in tokens), None)
    size = next((SIZE_TERMS[w] for w in tokens if w in SIZE_TERMS), None)
    spatial = next((SPATIAL_TERMS[w] for w in tokens if w in SPATIAL_TERMS), None)
    return Constraints(noun=noun, color=color, size=size, spatial=spatial)


def parse_source(text: str) -> Constraints:
    """The object to be picked up — clause before the first ' on '."""
    t = text.lower().strip()
    clause = re.split(r"\bon\b", t)[0]
    tokens = re.findall(r"[a-z]+", clause)
    return Constraints(
        noun=next((n for n in NOUNS if n in tokens), None),
        color=next((c for c in COLOR_TERMS if c in tokens), None),
        size=next((SIZE_TERMS[w] for w in tokens if w in SIZE_TERMS), None),
        spatial=next((SPATIAL_TERMS[w] for w in tokens if w in SPATIAL_TERMS), None),
    )
