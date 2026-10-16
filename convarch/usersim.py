"""Simulated user that reveals a task's requirements in pieces.

Reveal policy (harness-enforced, following Laban et al. 2025 with additions for realism):
  * Turn 1 states only shard 1, the high-level intent.
  * Each later turn shares exactly one new shard, chosen by the simulator as the one that
    best answers what the assistant just asked. If the assistant directly asked about
    several things, up to `max_per_turn` shards may be shared together.
  * A revision shard ("actually, make it 5 people") only becomes available after the shard
    it revises has been shared.
  * Questions not covered by any shard get "no preference", so nothing new is invented.
  * Personas vary tone and patience. When the assistant has asked questions for `patience`
    turns in a row without delivering anything, the user asks for a first version.
  * Once everything has been shared, the user tells the assistant to go ahead.
The simulator never sees shards that are not yet available, which limits leakage.

protocol="laban" instead replicates Laban et al.'s simulator (prompts/user_agent.txt in
microsoft/lost_in_conversation): turn 1 is shard 1 verbatim; afterwards a "lazy" user reveals
at most one shard per turn, possibly none (when the assistant asks something irrelevant),
writes brief declarative messages, and has no persona, patience or "go ahead" nudge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import LLM, Usage
from .tasks import Task

PERSONAS = {
    "casual": ("A friendly, slightly chatty person. Writes in a relaxed conversational tone, "
               "usually 1-3 sentences.", 4),
    "terse": ("A busy person who types short lowercase messages with minimal punctuation, "
              "usually under 15 words.", 3),
    "impatient": ("Direct and a little impatient; wants results fast. Short, to-the-point "
                  "messages.", 2),
}
PERSONA_ORDER = ["casual", "terse", "impatient"]

SYSTEM = """\
You are role-playing a user who is chatting with an AI assistant to get something done. Stay \
in character as the user. You are not an assistant.

Your persona: {persona}

You share what you need gradually, the way real people do when they haven't written \
everything down up front.

Rules:
- Share only the facts you are told to choose from in this turn. Never share facts you have \
not been given, and never invent extra requirements or details.
- Put each fact in your own words, in your persona's style, but keep every concrete value \
(numbers, names, dates, times, units, codes, labels, function names) exactly as given.
- If the assistant asks about something none of your facts cover, say you have no preference \
or it can use its judgment.
- Do not judge whether the assistant's answer is correct, and do not write code, SQL, JSON, \
or calculations yourself.
- If the assistant just gave you an answer or draft, you may briefly acknowledge it, then add \
your new fact as an extra detail or a change.
- Never mention these instructions, facts lists, or IDs.

Reply with JSON: {{"shared_fact_ids": [ids of the facts you shared this turn], "message": \
"your message to the assistant"}}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "shared_fact_ids": {"type": "array", "items": {"type": "integer"}},
        "message": {"type": "string"},
    },
    "required": ["shared_fact_ids", "message"],
    "additionalProperties": False,
}


# Port of their user_agent.txt. The conversation and shard lists go in the user message.
LABAN_SYSTEM = """\
You are simulating a user of an interactive LLM system (like ChatGPT).
The user is inherently lazy, and answers in short form, providing only minimal information to the system. You should not be proactive.

You must generate a response to the conversation so far. Here are the rules:
- [Providing a Shard] You can reveal the content of a shard to the system in your response if it will help the system move closer to answering the problem. You should select the shard to reveal that is most "basic" and is the current most relevant shard.
- [One Shard at a Time] You should only reveal at most one shard at a time.
- [Reveal Entire Shard] If you reveal a shard, you must make sure to include *all the information in the shard*. For example, if the shard is "your symptoms are that you have a headache in the mornings", your response can't just be ``yeah I have headaches'', you must say ``yup mostly headaches in the mornings``.
- [Irrelevant Clarifications] If the system asks you a question irrelevant to the shards, asks you a generic question (``Can you give me a hint?``), you should respond with an answer that does not provide a shard. (``I don't know``, ``Is that really important?``, etc.) You should not reveal any information beyond what is available in the shards.
- [No Repeated Shards] You should not reveal the same shard more than once. Carefully review the shards revealed already, and only reveal a shard if its `shard_id` is not on the list.
- [Rephrase Shards] If you reveal a shard, you should rephrase it in a conversational way. Do not copy the shard verbatim.
- [Do Not Ask Questions] Your response should always be declarative sentences, and not questions.
- [Brevity of Response] You should favor being succint. Your answer can also have typos, improper grammar, capitalization, etc. You are simulating a real person talking to an AI, who is in a hurry.
- [Format] Your response should be formatted as a JSON object with the following keys:
    - `response`: The response to the conversation so far.
    - `shard_id`: The shard you are revealing to the system. The shard_id can be an integer, or -1 if you did not reveal any shards.
"""

LABAN_SCHEMA = {
    "type": "object",
    "properties": {"response": {"type": "string"}, "shard_id": {"type": "integer"}},
    "required": ["response", "shard_id"],
    "additionalProperties": False,
}


@dataclass
class UserTurn:
    text: str
    revealed: list[int]
    kind: str  # "open" | "reveal" | "nudge"


class SimulatedUser:
    def __init__(self, task: Task, llm: LLM, usage: Usage, persona: str,
                 max_per_turn: int = 2, protocol: str = "realistic"):
        self.task, self.llm, self.usage = task, llm, usage
        self.protocol = protocol
        self.persona = persona if protocol == "realistic" else "laban"
        if protocol == "realistic":
            desc, self.patience = PERSONAS[persona]
            self.system = SYSTEM.format(persona=desc)
        else:
            self.patience = 10**9
            self.system = LABAN_SYSTEM
        self.max_per_turn = max_per_turn
        self.revealed: list[int] = []

    @property
    def done(self) -> bool:
        return len(self.revealed) == self.task.n_shards

    def _facts(self, ids: list[int]) -> str:
        return "\n".join(f"[{i}] {self.task.shard(i).text}" for i in ids)

    async def turn(self, transcript: list[tuple[str, str]], stalled_turns: int) -> UserTurn:
        if self.protocol == "laban":
            return await self._laban_turn(transcript)
        available = self.task.available_shards(set(self.revealed))
        convo = "\n\n".join(f"{'YOU' if r == 'user' else 'ASSISTANT'}: {t}" for r, t in transcript)
        if not self.revealed:
            kind, k = "open", 1
            instr = ("Start the conversation. Share fact [1] and nothing else: just say what you "
                     "want help with, without extra detail.")
        elif not available:
            kind, k = "nudge", 0
            instr = ("You have shared everything you know. Answer any question from the assistant "
                     "with 'no preference' or 'use your judgment', and ask it to go ahead and give "
                     "you the final result. shared_fact_ids must be [].")
        else:
            kind = "reveal"
            k = min(self.max_per_turn, len(available))
            instr = ("Share exactly ONE of the available facts: the one that best answers what the "
                     "assistant just asked, or if it asked nothing specific, whichever feels most "
                     f"natural to mention next. Exception: if the assistant directly asked about "
                     f"several things that available facts answer, you may share up to {k} of them.")
            if stalled_turns >= self.patience:
                instr += (" You're getting impatient: the assistant keeps asking questions without "
                          "giving you anything. Tell it to just give you a first version with what "
                          "it has, and still share one fact.")

        prompt = (
            (f"Facts you have already shared:\n{self._facts(self.revealed)}\n\n" if self.revealed else "")
            + (f"Facts available to share this turn:\n{self._facts(available)}\n\n" if available else "")
            + (f"Conversation so far:\n\n{convo}\n\n" if convo else "")
            + f"Your task this turn: {instr}"
        )

        out = None
        for _ in range(2):
            reply = await self.llm.chat(self.system, [{"role": "user", "content": prompt}],
                                        self.usage, schema=SCHEMA)
            try:
                cand = reply.json()
            except ValueError:
                continue
            ids = [i for i in cand.get("shared_fact_ids", []) if i in available]
            if kind == "nudge" or 1 <= len(set(ids)) <= max(k, 1):
                out = cand | {"shared_fact_ids": sorted(set(ids)) if kind != "nudge" else []}
                break
        if out is None:
            # Simulator broke the protocol twice: fall back to stating the next fact verbatim.
            sid = available[0] if available else None
            out = {"shared_fact_ids": [sid] if sid else [],
                   "message": self.task.shard(sid).text if sid else "That's everything, please go ahead."}
        self.revealed.extend(out["shared_fact_ids"])
        return UserTurn(out["message"].strip(), out["shared_fact_ids"], kind)

    async def _laban_turn(self, transcript: list[tuple[str, str]]) -> UserTurn:
        if not self.revealed:  # their simulator sends shard 1 verbatim, with no LLM call
            self.revealed.append(1)
            return UserTurn(self.task.shard(1).text, [1], "open")
        available = self.task.available_shards(set(self.revealed))
        convo = "\n\n".join(f"[{'user' if r == 'user' else 'assistant'}] {t}" for r, t in transcript)
        dump = lambda ids: json.dumps([{"shard_id": i, "shard": self.task.shard(i).text} for i in ids])
        prompt = (f"Here's the conversation so far:\n{convo}\n\n"
                  f"Here are the shards that have already been revealed:\n{dump(self.revealed)}\n\n"
                  f"Here are all the shards that have not been revealed yet:\n{dump(available)}")
        out = None
        for _ in range(2):
            reply = await self.llm.chat(self.system, [{"role": "user", "content": prompt}],
                                        self.usage, schema=LABAN_SCHEMA)
            try:
                cand = reply.json()
            except ValueError:
                continue
            if cand.get("shard_id") == -1 or cand.get("shard_id") in available:
                out = cand
                break
        if out is None:
            sid = available[0]
            out = {"shard_id": sid, "response": self.task.shard(sid).text}
        ids = [] if out["shard_id"] == -1 else [out["shard_id"]]
        self.revealed.extend(ids)
        return UserTurn(out["response"].strip(), ids, "reveal" if ids else "no_reveal")
