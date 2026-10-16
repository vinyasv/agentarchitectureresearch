"""Run episodes (task x condition x seed) and append results to a JSONL file."""

from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from .architectures import SingleAgent, TwoAgent
from .graders import cumulative_calls_correct, extract, gold_answer_text, grade
from .llm import LLM, StubLLM, Usage, make_client
from .tasks import Task
from .usersim import PERSONA_ORDER, SimulatedUser

CONDITIONS = ["full", "single", "single_clarify", "two_agent", "two_agent_verbatim"]
ROLES = ["assistant", "gatherer", "executor", "user"]


@dataclass
class Config:
    agent_model: str
    user_model: str
    agent_effort: str = "medium"
    user_effort: str = "low"
    max_per_turn: int = 2
    extra_turns: int = 3  # assistant turns allowed beyond the number of shards
    protocol: str = "realistic"  # "realistic" | "laban"
    dry_run: bool = False


def _make_llms(cfg: Config, client, sem):
    agent = LLM(cfg.agent_model, cfg.agent_effort, client, sem)
    user = LLM(cfg.user_model, cfg.user_effort, client, sem)
    return agent, user


def _build_agent(task: Task, condition: str, agent_llm, usage):
    if condition in ("full", "single"):
        return SingleAgent(task, agent_llm, usage)
    if condition == "single_clarify":
        return SingleAgent(task, agent_llm, usage, clarify=True)
    if condition == "two_agent":
        return TwoAgent(task, agent_llm, agent_llm, usage)
    if condition == "two_agent_verbatim":
        return TwoAgent(task, agent_llm, agent_llm, usage, verbatim=True)
    raise ValueError(condition)


async def run_episode(task: Task, condition: str, seed: int, cfg: Config, agent_llm, user_llm) -> dict:
    usage = {r: Usage() for r in ROLES}
    persona = PERSONA_ORDER[seed % len(PERSONA_ORDER)]
    user = SimulatedUser(task, user_llm, usage["user"], persona, cfg.max_per_turn, cfg.protocol)
    persona = user.persona
    if cfg.dry_run:
        agent_llm, user_llm = _stubs(task, user)
        user.llm = user_llm
    agent = _build_agent(task, condition, agent_llm, usage)

    turns: list[dict] = []
    pairs: list[tuple[str, str]] = []

    async def assistant_turn(user_text: str):
        at = await agent.respond(user_text)
        answer = extract(task, at.text)
        g = grade(task, answer) if answer is not None else None
        rec = {"role": "assistant", "text": at.text, "is_attempt": answer is not None,
               "shards_known": len(user.revealed) if condition != "full" else task.n_shards,
               "grade": asdict(g) if g else None, "meta": at.meta}
        turns.append(rec)
        pairs.append(("assistant", at.text))
        return rec

    if condition == "full":
        user.revealed = [s.id for s in task.shards]
        text = task.full_prompt()
        turns.append({"role": "user", "text": text, "revealed": [s.id for s in task.shards], "kind": "full"})
        await assistant_turn(text)
    elif cfg.protocol == "laban":
        # Their loop: stop at the first correct answer, or once every shard has been revealed
        # and the assistant has answered the turn that revealed the last one.
        max_turns = 2 * task.n_shards + 2
        ut = await user.turn(pairs, 0)
        while True:
            turns.append({"role": "user", "text": ut.text, "revealed": ut.revealed, "kind": ut.kind})
            pairs.append(("user", ut.text))
            rec = await assistant_turn(ut.text)
            if rec["is_attempt"] and rec["grade"]["correct"]:
                break
            if user.done or sum(t["role"] == "assistant" for t in turns) >= max_turns:
                break
            ut = await user.turn(pairs, 0)
    else:
        stalled, nudged = 0, False
        max_turns = task.n_shards + cfg.extra_turns
        ut = await user.turn(pairs, stalled)
        while True:
            turns.append({"role": "user", "text": ut.text, "revealed": ut.revealed, "kind": ut.kind})
            pairs.append(("user", ut.text))
            rec = await assistant_turn(ut.text)
            stalled = 0 if rec["is_attempt"] else stalled + 1
            n_asst = sum(t["role"] == "assistant" for t in turns)
            if user.done and (rec["is_attempt"] or nudged):
                break
            if n_asst >= max_turns:
                break
            ut = await user.turn(pairs, stalled)
            nudged = nudged or ut.kind == "nudge"

    return _summarize(task, condition, seed, persona, user, turns, usage)


def attempt_stats(task: Task, turns: list[dict]) -> dict:
    asst = [t for t in turns if t["role"] == "assistant"]
    attempts = [(i, t) for i, t in enumerate(asst, 1) if t["is_attempt"]]
    last = attempts[-1][1]["grade"] if attempts else None
    first = attempts[0] if attempts else None
    return {
        "correct": bool(last and last["correct"]),
        "lenient_correct": bool(last and (last["correct"] or last["detail"].get("extra_cols_ok"))),
        "score": last["score"] if last else 0.0,
        "n_attempts": len(attempts),
        "n_premature_attempts": sum(1 for _, t in attempts if t["shards_known"] < task.n_shards),
        "first_attempt_turn": first[0] if first else None,
        "first_attempt_correct": bool(first and first[1]["grade"]["correct"]),
        "first_attempt_shards_known": first[1]["shards_known"] if first else None,
        "ever_correct": any(t["grade"]["correct"] for _, t in attempts),
        "n_assistant_turns": len(asst),
        "n_question_turns": len(asst) - len(attempts),
        "cumulative_correct": (cumulative_calls_correct(task, [t["text"] for t in asst])
                               if task.domain == "lic_actions" else None),
    }


def regrade(task: Task, rec: dict) -> dict:
    """Re-extract and re-grade every assistant turn of a saved episode (no API calls).

    The conversation itself is not replayed, so if the extraction rule changes whether a
    turn counts as an attempt, the episode's ending is still the one the old rule produced.
    """
    for t in rec["turns"]:
        if t["role"] == "assistant":
            answer = extract(task, t["text"])
            t["is_attempt"] = answer is not None
            t["grade"] = asdict(grade(task, answer)) if answer is not None else None
    rec.update(attempt_stats(task, rec["turns"]))
    return rec


def _summarize(task, condition, seed, persona, user, turns, usage) -> dict:
    total_cost = sum(u.cost_usd for u in usage.values())
    agent_cost = total_cost - usage["user"].cost_usd
    return {
        "task_id": task.id, "domain": task.domain, "condition": condition, "seed": seed,
        "persona": persona, "n_shards": task.n_shards,
        "has_revision": any(s.revises is not None for s in task.shards),
        **attempt_stats(task, turns),
        "all_revealed": len(user.revealed) == task.n_shards or condition == "full",
        "nudged": any(t.get("kind") == "nudge" for t in turns),
        "refusals": sum(u.refusals for u in usage.values()),
        "usage": {r: u.to_dict() for r, u in usage.items()},
        "cost_usd": round(total_cost, 6),
        "agent_cost_usd": round(agent_cost, 6),
        "agent_output_tokens": sum(usage[r].output_tokens for r in ROLES if r != "user"),
        "turns": turns,
    }


# ------------------------------------------------------------------ dry-run stubs

def _stubs(task: Task, user: SimulatedUser):
    """Scripted stand-ins that exercise every code path without the API."""
    try:
        gold = gold_answer_text(task)
    except ValueError:  # imported code tasks have no reference solution
        gold = "```python\ndef stub():\n    return None\n```"

    def user_resp(system, messages, schema):
        prompt = messages[-1]["content"]
        if "shard_id" in schema["properties"]:  # laban protocol
            pending = json.loads(prompt.rsplit("not been revealed yet:\n", 1)[1])
            return json.dumps({"shard_id": pending[0]["shard_id"], "response": pending[0]["shard"]})
        avail = re.search(r"Facts available to share this turn:\n((?:\[\d+\].*\n?)+)", prompt)
        if not avail:
            return json.dumps({"shared_fact_ids": [], "message": "That's all I have, go ahead."})
        sid, text = re.match(r"\[(\d+)\] (.*)", avail.group(1)).groups()
        return json.dumps({"shared_fact_ids": [int(sid)], "message": text})

    def agent_resp(system, messages, schema):
        if schema is not None:  # gatherer
            return json.dumps({"spec": {"objective": "stub", "requirements": [], "assumptions": [],
                                        "open_questions": []},
                               "ready_to_execute": user.done, "message_to_user": "Here you go."})
        if system.startswith("You are the executor"):
            return gold
        n_asst = sum(m["role"] == "assistant" for m in messages)
        if n_asst == 1:  # a premature wrong attempt, to exercise "last attempt counts"
            return "Draft:\n```python\ndef nope(): pass\n```\n```sql\nSELECT 1\n```\n" \
                   '```json\n{"name": "x", "arguments": {}}\n```\nANSWER: -1'
        return gold if user.done else "Got it. Anything else I should know?"

    return StubLLM(agent_resp), StubLLM(user_resp)


# ------------------------------------------------------------------ batch runner

def episode_key(task_id: str, condition: str, seed: int) -> str:
    return f"{task_id}|{condition}|{seed}"


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    done.add(episode_key(r["task_id"], r["condition"], r["seed"]))
    return done


async def run_batch(tasks: list[Task], conditions: list[str], seeds: list[int], cfg: Config,
                    out_path: Path, concurrency: int = 8, limit: int | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    # Interleave domains so a --limit pilot samples every domain and condition.
    by_domain: dict[str, list[Task]] = {}
    for t in tasks:
        by_domain.setdefault(t.domain, []).append(t)
    interleaved = [ts[i] for i in range(max(map(len, by_domain.values())))
                   for ts in by_domain.values() if i < len(ts)]
    todo = [(t, c, s) for s in seeds for t in interleaved for c in conditions
            if episode_key(t.id, c, s) not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(done)} episodes already done, {len(todo)} to run -> {out_path}")
    if not todo:
        return

    client = None if cfg.dry_run else make_client()
    api_sem = asyncio.Semaphore(concurrency * 2)
    agent_llm, user_llm = _make_llms(cfg, client, api_sem)
    ep_sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    stats = {"n": 0, "cost": 0.0, "correct": 0, "t0": time.monotonic()}

    async def one(task, cond, seed):
        async with ep_sem:
            try:
                rec = await run_episode(task, cond, seed, cfg, agent_llm, user_llm)
            except Exception as e:
                rec = {"task_id": task.id, "domain": task.domain, "condition": cond, "seed": seed,
                       "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
            rec["config"] = asdict(cfg)
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                stats["n"] += 1
                stats["cost"] += rec.get("cost_usd", 0.0)
                stats["correct"] += bool(rec.get("correct"))
                status = "ERROR " + rec["error"][:80] if rec.get("error") else (
                    "ok" if rec["correct"] else "wrong")
                el = time.monotonic() - stats["t0"]
                print(f"[{stats['n']}/{len(todo)}] {task.id:24s} {cond:19s} s{seed} {status:6s} "
                      f"| total ${stats['cost']:.2f} | {el:.0f}s", flush=True)

    await asyncio.gather(*(one(*x) for x in todo))
