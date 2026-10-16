"""Thin async wrapper over the Anthropic Messages API with usage/cost accounting.

Every agent keeps an append-only history and replays full `response.content`
(thinking blocks included), which is what preserved thinking requires.
`StubLLM` replaces the API for offline dry runs.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

# $ per 1M tokens (input, output). Cache reads/writes are not used here.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.00, 20.00),
}

DEFAULT_AGENT_MODEL = "claude-sonnet-5-5"
DEFAULT_USER_MODEL = "claude-haiku-5-5"


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    refusals: int = 0

    def add(self, model: str, inp: int, out: int) -> None:
        pi, po = PRICES.get(model, (0.0, 0.0))
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        self.cost_usd += inp / 1e6 * pi + out / 1e6 * po

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class Reply:
    content: list[Any]  # blocks to append to history verbatim
    text: str           # concatenated text blocks
    stop_reason: str | None = None

    def json(self) -> dict:
        return json.loads(self.text)


@dataclass
class LLM:
    """One model + effort setting. `usage` is shared per role by the caller."""

    model: str
    effort: str = "medium"
    client: anthropic.AsyncAnthropic | None = None
    semaphore: asyncio.Semaphore | None = None
    max_tokens: int = 16000

    async def chat(self, system: str, messages: list[dict], usage: Usage,
                   schema: dict | None = None) -> Reply:
        output_config: dict = {"effort": self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=messages, output_config=output_config)
        async with self.semaphore:
            resp = await self.client.messages.create(**kwargs)
        usage.add(self.model, resp.usage.input_tokens, resp.usage.output_tokens)
        if resp.stop_reason == "refusal":
            usage.refusals += 1
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Reply(content=list(resp.content), text=text, stop_reason=resp.stop_reason)


@dataclass
class StubLLM:
    """Offline stand-in: `responder(system, messages, schema)` returns the reply text."""

    responder: Callable[[str, list[dict], dict | None], str]
    model: str = "stub"
    calls: list = field(default_factory=list)

    async def chat(self, system: str, messages: list[dict], usage: Usage,
                   schema: dict | None = None) -> Reply:
        text = self.responder(system, messages, schema)
        usage.add(self.model, 0, 0)
        return Reply(content=[{"type": "text", "text": text}], text=text, stop_reason="end_turn")


def make_client(max_retries: int = 8) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(max_retries=max_retries, timeout=300.0)


def content_to_jsonable(content: list[Any]) -> list[dict]:
    out = []
    for b in content:
        if isinstance(b, dict):
            out.append(b)
        else:
            d = b.model_dump(mode="json", exclude_none=True)
            if d.get("type") == "thinking":
                d.pop("signature", None)
            out.append(d)
    return out
