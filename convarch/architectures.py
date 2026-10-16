"""Conversational architectures under test.

All of them expose `respond(user_text) -> AssistantTurn`, so the episode runner
treats them the same. The text in AssistantTurn is exactly what the user sees.

  single          One agent, plain assistant prompt (the setup Laban et al. measure).
  single_clarify  One agent whose prompt asks it to gather requirements first and to
                  regenerate from scratch on changes. This is a prompt-only control:
                  any gain of two_agent over it comes from the architecture, not the
                  instructions.
  two_agent       Gatherer + executor. The gatherer talks to the user and rewrites a
                  consolidated requirements spec every turn; when it is ready, a fresh
                  executor (which never sees the conversation) builds the deliverable
                  from the spec alone.
  two_agent_verbatim  Same, but the executor also receives the user's messages
                  verbatim, which tests whether spec summarization loses information.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLM, Reply, Usage
from .tasks import Task, task_context, task_deliverable

DELIVERABLE_KIND = {
    "code": "a single Python function",
    "sql": "a single SQLite query",
    "action": "exactly one API call",
    "math": "a worked calculation ending in one numeric answer",
    "lic_code": "a single Python function",
    "lic_math": "a worked calculation ending in one numeric answer",
    "lic_actions": "one or more API function calls",
}

CLARIFY_GUIDANCE = (
    "Users often share requirements in pieces over several messages. Before producing the "
    "deliverable, make sure you understand what is needed: ask brief, specific questions "
    "(at most 2-3 at a time) about details that would change the result, and fill minor gaps "
    "with stated assumptions. When you produce the deliverable, base it on everything the user "
    "has said in the whole conversation. If they later add or change something, produce a "
    "complete updated version from scratch that reflects all requirements so far, rather than "
    "patching your previous answer."
)


@dataclass
class AssistantTurn:
    text: str
    meta: dict = field(default_factory=dict)


def _assistant_message(reply: Reply) -> dict:
    content = list(reply.content)
    if not any((b.get("type") if isinstance(b, dict) else b.type) == "text" for b in content):
        content.append({"type": "text", "text": "(no response)"})
    return {"role": "assistant", "content": content}


class SingleAgent:
    def __init__(self, task: Task, llm: LLM, usage: dict[str, Usage], clarify: bool = False):
        self.task, self.llm, self.usage = task, llm, usage
        parts = ["You are a helpful assistant.", task_context(task)] if task.context is None \
            else [task_context(task)]
        if clarify:
            parts.append(CLARIFY_GUIDANCE)
        if task_deliverable(task):
            parts.append(task_deliverable(task))
        self.system = "\n\n".join(parts)
        self.history: list[dict] = []

    async def respond(self, user_text: str) -> AssistantTurn:
        self.history.append({"role": "user", "content": user_text})
        reply = await self.llm.chat(self.system, self.history, self.usage["assistant"])
        self.history.append(_assistant_message(reply))
        return AssistantTurn(reply.text or "(no response)", {"stop_reason": reply.stop_reason})


# ------------------------------------------------------------------ two-agent

GATHERER_PROMPT = """\
You are the requirements lead in a two-agent assistant. You talk with the user; a separate \
executor agent produces the deliverable. The executor never sees this conversation, only the \
spec you write, so the spec must carry everything that matters.

{context}

The executor will produce: {kind}.

Every turn, return JSON with:
- spec: the complete current requirements, rewritten in full every turn.
  - objective: one or two sentences on what the user wants.
  - requirements: every requirement the user has stated, each as one precise, self-contained \
statement. Keep exact values (numbers, names, dates, units, formats, codes). When the user \
changes their mind, replace the old requirement; never keep both.
  - assumptions: choices the user has not specified that the executor should make. Never \
present an assumption as a requirement.
  - open_questions: unknowns whose answer could change the deliverable.
- ready_to_execute: true to have the executor produce (or re-produce) the deliverable now, \
from the spec.
- message_to_user: what you say to the user this turn.

How to run the conversation:
- Users often give requirements in pieces. Ask brief, specific questions (at most 2-3 at a \
time) about details that would change the result.
- Early drafts are cheap: each execution starts fresh from the spec, so a draft never \
constrains later versions. Set ready_to_execute when the spec is good enough that the \
deliverable would likely be accepted, not when every conceivable detail is known. Fill minor \
gaps with assumptions instead of asking. If the user asks for a result, or says they have \
nothing to add, set ready_to_execute to true.
- When ready_to_execute is true, message_to_user is a short lead-in; the executor's \
deliverable is shown to the user right after it.
- After a deliverable has been shown, if the user adds or changes anything, update the spec \
and set ready_to_execute to true again so the executor regenerates it from the full spec. \
Never write or patch the deliverable yourself."""

EXECUTOR_PROMPT = """\
You are the executor in a two-agent assistant. You receive a requirements spec written by a \
colleague who talked with the user. Produce the deliverable so that it satisfies every \
requirement exactly.

{context}

{deliverable}

Rules:
- Every requirement is binding, with its exact values. Where the spec is silent, follow its \
assumptions; otherwise make the most standard choice.
- You cannot ask questions. Produce the deliverable now.
- Reply with the deliverable, then, only if you made choices the spec does not state, a short \
"Assumptions:" list of at most 3 bullets. No other commentary.{verbatim_rule}"""

VERBATIM_RULE = (
    "\n- The user's own messages are included for reference. If they conflict with the spec, "
    "the user's most recent statement wins."
)

_STR_LIST = {"type": "array", "items": {"type": "string"}}
GATHERER_SCHEMA = {
    "type": "object",
    "properties": {
        "spec": {
            "type": "object",
            "properties": {
                "objective": {"type": "string"},
                "requirements": _STR_LIST,
                "assumptions": _STR_LIST,
                "open_questions": _STR_LIST,
            },
            "required": ["objective", "requirements", "assumptions", "open_questions"],
            "additionalProperties": False,
        },
        "ready_to_execute": {"type": "boolean"},
        "message_to_user": {"type": "string"},
    },
    "required": ["spec", "ready_to_execute", "message_to_user"],
    "additionalProperties": False,
}


def render_spec(spec: dict) -> str:
    def bullets(items):
        return "\n".join(f"- {x}" for x in items) or "- (none)"

    return (
        f"Objective: {spec.get('objective', '')}\n\n"
        f"Requirements:\n{bullets(spec.get('requirements', []))}\n\n"
        f"Assumptions (not confirmed by the user):\n{bullets(spec.get('assumptions', []))}\n\n"
        f"Open questions (unresolved; use your best judgment):\n{bullets(spec.get('open_questions', []))}"
    )


class TwoAgent:
    def __init__(self, task: Task, gatherer: LLM, executor: LLM, usage: dict[str, Usage],
                 verbatim: bool = False):
        self.task, self.gatherer, self.executor, self.usage = task, gatherer, executor, usage
        ctx = task_context(task)
        self.g_system = GATHERER_PROMPT.format(context=ctx, kind=DELIVERABLE_KIND[task.domain])
        self.e_system = EXECUTOR_PROMPT.format(
            context=ctx, deliverable=task_deliverable(task),
            verbatim_rule=VERBATIM_RULE if verbatim else "")
        self.verbatim = verbatim
        self.history: list[dict] = []
        self.user_messages: list[str] = []
        self.pending_delivery: str | None = None
        self.spec: dict = {}
        self.executions = 0

    async def _gather(self) -> dict:
        last_err = None
        for _ in range(2):
            reply = await self.gatherer.chat(self.g_system, self.history, self.usage["gatherer"],
                                             schema=GATHERER_SCHEMA)
            try:
                out = reply.json()
                self.history.append(_assistant_message(reply))
                return out
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e  # unparseable reply is discarded, history stays append-only
        raise RuntimeError(f"gatherer returned invalid JSON twice: {last_err}")

    async def respond(self, user_text: str) -> AssistantTurn:
        self.user_messages.append(user_text)
        if self.pending_delivery is not None:
            content = (f"[Shown to the user after your last message: the executor's deliverable]\n"
                       f"{self.pending_delivery}\n\n[User]\n{user_text}")
            self.pending_delivery = None
        else:
            content = f"[User]\n{user_text}"
        self.history.append({"role": "user", "content": content})

        out = await self._gather()
        self.spec = out["spec"]
        message = out["message_to_user"].strip()
        meta = {"spec": self.spec, "ready": out["ready_to_execute"]}
        if not out["ready_to_execute"]:
            return AssistantTurn(message, meta)

        exec_input = "# Requirements spec\n\n" + render_spec(self.spec)
        if self.verbatim:
            msgs = "\n".join(f"{i}. {m}" for i, m in enumerate(self.user_messages, 1))
            exec_input += f"\n\n# The user's messages, verbatim and in order\n\n{msgs}"
        reply = await self.executor.chat(self.e_system, [{"role": "user", "content": exec_input}],
                                         self.usage["executor"])
        self.executions += 1
        deliverable = reply.text.strip() or "(executor returned nothing)"
        self.pending_delivery = deliverable
        meta.update({"executor_input": exec_input, "executor_output": deliverable})
        return AssistantTurn(f"{message}\n\n{deliverable}" if message else deliverable, meta)
