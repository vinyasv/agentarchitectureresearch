"""Task and shard definitions.

A task is one fully specified request with a definitive, machine-checkable outcome.
It is split into *shards*: shard 1 is the high-level intent, every later shard adds
exactly one requirement the final answer depends on. Together the shards are
equivalent to the fully specified instruction (the sharding method of Laban et al.,
"LLMs Get Lost in Multi-Turn Conversation", 2025).

A shard may be a *revision* of an earlier shard (the user changes their mind). A
revision can only be revealed after the shard it revises, and in the fully specified
prompt the original shard is replaced by the revision's `full_text`.

A shard may be a *distractor*: true but irrelevant context ("it's for our gym app"), as
in Laban et al.'s BFCL shards. Distractors do not change the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Shard:
    id: int
    text: str
    revises: int | None = None
    # Consolidated wording used in the fully specified prompt (revision shards only).
    full_text: str | None = None
    distractor: bool = False


@dataclass
class Task:
    id: str
    domain: str  # "code" | "sql" | "action" | "math"
    title: str
    shards: list[Shard]
    gold: Any
    # Domain-specific grading payload (tests, ordered flag, arg rules, ...).
    grading: dict = field(default_factory=dict)
    # Shard-necessity evidence used by scripts/validate_tasks.py.
    ablations: dict = field(default_factory=dict)
    # Per-task overrides (used by imported tasks). None means "use the domain default".
    context: str | None = None        # assistant system-prompt context
    deliverable: str | None = None    # deliverable-format instruction
    full_text: str | None = None      # original fully specified instruction
    source: str = "ours"

    def shard(self, sid: int) -> Shard:
        return next(s for s in self.shards if s.id == sid)

    @property
    def n_shards(self) -> int:
        return len(self.shards)

    def full_prompt(self) -> str:
        """The fully specified single-turn instruction: the original text when known,
        otherwise the shards concatenated (Laban's CONCAT form)."""
        if self.full_text is not None:
            return self.full_text
        return self.concat_prompt()

    def concat_prompt(self) -> str:
        revised = {s.revises: s for s in self.shards if s.revises is not None}
        details = []
        for s in self.shards[1:]:
            if s.revises is not None:
                continue
            details.append(revised[s.id].full_text if s.id in revised else s.text)
        bullets = "\n".join(f"- {d}" for d in details)
        return f"{self.shards[0].text}\n\nDetails:\n{bullets}"

    def available_shards(self, revealed: set[int]) -> list[int]:
        """Unrevealed shards the user may share next."""
        if 1 not in revealed:
            return [1]
        out = []
        for s in self.shards:
            if s.id in revealed:
                continue
            if s.revises is not None and s.revises not in revealed:
                continue
            out.append(s.id)
        return out
