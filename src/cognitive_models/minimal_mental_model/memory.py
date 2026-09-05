from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class Chunk:
    """A declarative-memory chunk containing cueable slots and a payload."""

    chunk_id: int
    kind: str
    slots: dict[str, Any]
    payload: dict[str, Any]
    created_at: float
    use_times: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class Retrieval:
    chunk: Chunk
    activation: float


class DeclarativeMemory:
    """Deterministic minimal version of ACT-R declarative memory."""

    def __init__(self, retrieval_threshold: float = -2.0, decay: float = 0.5) -> None:
        self.retrieval_threshold = float(retrieval_threshold)
        self.decay = float(decay)
        self.clock = 1.0
        self._chunks: list[Chunk] = []

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    def advance(self, amount: float = 1.0) -> float:
        if amount <= 0:
            raise ValueError("Memory time must advance by a positive amount")
        self.clock += float(amount)
        return self.clock

    def store(
        self,
        kind: str,
        slots: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> Chunk:
        chunk = Chunk(
            chunk_id=len(self._chunks),
            kind=kind,
            slots=dict(slots),
            payload=dict(payload),
            created_at=self.clock,
            use_times=[self.clock],
        )
        self._chunks.append(chunk)
        return chunk

    def activation(self, chunk: Chunk, cue: Mapping[str, Any] | None = None) -> float:
        cue = cue or {}
        base = math.log(
            sum(max(self.clock - use_time, 1.0) ** (-self.decay) for use_time in chunk.use_times)
        )
        mismatches = sum(chunk.slots.get(slot) != value for slot, value in cue.items())
        return float(base - mismatches)

    def retrieve_many(
        self,
        kind: str | None = None,
        cue: Mapping[str, Any] | None = None,
        *,
        exact: bool = False,
        limit: int | None = None,
    ) -> list[Retrieval]:
        cue = cue or {}
        candidates: list[Retrieval] = []
        for chunk in self._chunks:
            if kind is not None and chunk.kind != kind:
                continue
            if exact and any(chunk.slots.get(slot) != value for slot, value in cue.items()):
                continue
            activation = self.activation(chunk, cue)
            if activation >= self.retrieval_threshold:
                candidates.append(Retrieval(chunk=chunk, activation=activation))
        candidates.sort(key=lambda item: (item.activation, item.chunk.chunk_id), reverse=True)
        return candidates if limit is None else candidates[:limit]

    def retrieve_one(
        self,
        kind: str | None = None,
        cue: Mapping[str, Any] | None = None,
        *,
        exact: bool = False,
    ) -> Retrieval | None:
        matches = self.retrieve_many(kind=kind, cue=cue, exact=exact, limit=1)
        if not matches:
            return None
        self.touch(matches[0].chunk)
        return matches[0]

    def touch(self, chunk: Chunk) -> None:
        chunk.use_times.append(self.clock)

