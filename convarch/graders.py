"""Deterministic answer extraction and grading.

An assistant message is an *answer attempt* if it contains the domain's deliverable
(a ```python / ```sql / ```json block, or an `ANSWER:` line). The episode score is
the grade of the last attempt in the conversation.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import math
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from .tasks import Task
from .tasks.action_tasks import API
from .tasks.sqldb import run_query

FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)
ANSWER_RE = re.compile(r"ANSWER:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)")


@dataclass
class Grade:
    correct: bool
    score: float  # partial credit in [0, 1] (fraction of tests / args); correct => 1.0
    detail: dict = field(default_factory=dict)


def _blocks(text: str, langs: set[str]) -> list[str]:
    return [body for lang, body in FENCE.findall(text) if lang.lower() in langs]


def extract(task: Task, text: str) -> str | None:
    """Return the deliverable in `text`, or None if this message is not an attempt."""
    if task.domain == "code":
        # Only a block that defines the function is a deliverable; usage examples are not.
        entry = task.grading["entry"]
        blocks = _blocks(text, {"python", "py"})
        defining = [b for b in blocks if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", b, re.M)]
        return defining[-1] if defining else None
    if task.domain == "sql":
        blocks = _blocks(text, {"sql", "sqlite"})
        return blocks[-1].strip() if blocks else None
    if task.domain == "action":
        # Only a JSON object naming a function is a call; other JSON (examples, fragments) is not.
        calls = []
        for b in _blocks(text, {"json"}):
            try:
                obj = json.loads(b)
            except json.JSONDecodeError:
                calls.append(b.strip())  # malformed call attempt: still an attempt, graded wrong
                continue
            if isinstance(obj, dict) and "name" in obj:
                calls.append(b.strip())
        return calls[-1] if calls else None
    if task.domain in ("math", "lic_math"):
        m = ANSWER_RE.findall(text)
        return m[-1].replace(",", "") if m else None
    if task.domain == "lic_code":
        blocks = [b for b in _blocks(text, {"python", "py", ""}) if re.search(r"^\s*def\s+\w+\s*\(", b, re.M)]
        if blocks:
            return blocks[-1]
        if "```" not in text and re.search(r"^def\s+\w+\s*\(", text, re.M):
            return text
        return None
    if task.domain == "lic_actions":
        return extract_calls(text)
    raise ValueError(task.domain)


def extract_calls(text: str) -> str | None:
    """A BFCL-style answer: `[f(a=1), g(b=2)]`, possibly fenced. Returns the call string or None."""
    from .vendor.lic_bfcl import ast_parse, is_function_calling_format_output

    candidates = [text] + [b for _, b in reversed(FENCE.findall(text))]
    candidates += [ln for ln in reversed(text.splitlines()) if ln.strip().startswith("[")]
    for c in candidates:
        try:
            calls = ast_parse(c.strip(), "Python")
        except Exception:
            continue
        if calls and is_function_calling_format_output(calls):
            return c.strip()
    return None


def grade(task: Task, answer: str | None) -> Grade:
    if answer is None:
        return Grade(False, 0.0, {"error": "no answer"})
    fn = {"code": grade_code, "sql": grade_sql, "action": grade_action, "math": grade_math,
          "lic_code": grade_lic_code, "lic_math": grade_lic_math, "lic_actions": grade_lic_actions}[task.domain]
    try:
        return fn(task, answer)
    except Exception as e:  # grader must never crash an episode
        return Grade(False, 0.0, {"error": f"grader exception: {e!r}"})


# ---------------------------------------------------------------- code

_HARNESS = textwrap.dedent('''
    import json, sys
    def raises(exc, thunk):
        try:
            thunk()
        except exc:
            return True
        except Exception:
            return False
        return False
    src = sys.stdin.read()
    payload = json.loads(src)
    ns = {}
    try:
        exec(compile(payload["code"], "candidate.py", "exec"), ns)
        f = ns[payload["entry"]]
    except Exception as e:
        print(json.dumps({"load_error": repr(e)}))
        sys.exit(0)
    out = []
    for t in payload["tests"]:
        try:
            ok = bool(eval(t["expr"], {"f": f, "raises": raises}))
            out.append({"ok": ok})
        except Exception as e:
            out.append({"ok": False, "error": repr(e)[:200]})
    print(json.dumps({"results": out}))
''')


def grade_code(task: Task, code: str) -> Grade:
    tests = task.grading["tests"]
    payload = json.dumps({"code": code, "entry": task.grading["entry"], "tests": tests})
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.py"
        harness.write_text(_HARNESS)
        try:
            proc = subprocess.run([sys.executable, "-I", str(harness)], input=payload, cwd=tmp,
                                  capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired:
            return Grade(False, 0.0, {"error": "timeout"})
    try:
        res = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return Grade(False, 0.0, {"error": "harness crashed", "stderr": proc.stderr[-500:]})
    if "load_error" in res:
        return Grade(False, 0.0, {"error": res["load_error"]})
    oks = [r["ok"] for r in res["results"]]
    failed_shards = sorted({t["shard"] for t, ok in zip(tests, oks) if not ok})
    return Grade(all(oks), sum(oks) / len(oks), {"passed": sum(oks), "total": len(oks),
                                                 "failed_shards": failed_shards})


# ---------------------------------------------------------------- sql

def _norm_cell(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, str):
        return v.strip()
    return v


def _rows(rows) -> list[tuple]:
    return [tuple(_norm_cell(c) for c in r) for r in rows]


def sql_results_match(gold_rows, cand_rows, ordered: bool) -> bool:
    gold, cand = _rows(gold_rows), _rows(cand_rows)
    if len(gold) != len(cand):
        return False
    if not gold:
        return True
    ncols = len(gold[0])
    if any(len(r) != ncols for r in cand):
        return False
    # Column order is not part of the spec: accept any permutation of the columns.
    for perm in itertools.permutations(range(ncols)):
        permuted = [tuple(r[i] for i in perm) for r in cand]
        if ordered and permuted == gold:
            return True
        if not ordered and sorted(map(repr, permuted)) == sorted(map(repr, gold)):
            return True
    return False


def grade_sql(task: Task, sql: str) -> Grade:
    gold_rows = run_query(task.gold)
    try:
        rows = run_query(sql)
    except Exception as e:
        return Grade(False, 0.0, {"error": f"sql error: {e}"})
    ordered = task.grading.get("ordered", False)
    ok = sql_results_match(gold_rows, rows, ordered)
    detail = {"n_rows": len(rows), "gold_rows": len(gold_rows)}
    if not ok and rows and gold_rows and len(rows[0]) > len(gold_rows[0]):
        # Lenient check: right rows once extra columns are dropped ("answer bloat").
        k = len(gold_rows[0])
        detail["extra_cols_ok"] = any(
            sql_results_match(gold_rows, [tuple(r[i] for i in cols) for r in rows], ordered)
            for cols in itertools.combinations(range(len(rows[0])), k))
    return Grade(ok, 1.0 if ok else 0.0, detail)


# ---------------------------------------------------------------- action

def _norm_str(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().casefold()


def _parse_dt(v) -> dt.datetime:
    s = str(v).strip().replace(" ", "T")
    s = re.sub(r"(Z|[+-]\d\d:?\d\d)$", "", s)
    return dt.datetime.fromisoformat(s)


def _parse_time(v) -> tuple[int, int]:
    s = str(v).strip().lower()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm)?", s)
    if not m:
        raise ValueError(s)
    h, mi = int(m.group(1)), int(m.group(2))
    if m.group(3) == "pm" and h < 12:
        h += 12
    if m.group(3) == "am" and h == 12:
        h = 0
    return h, mi


def arg_equal(typ: str, gold, cand, contains: str | None = None, aliases: list[str] | None = None) -> bool:
    if gold is None or cand is None:
        if typ.startswith("set") and not gold and not cand:
            return True
        return gold is None and cand is None
    if typ == "free":
        return bool(str(cand).strip())
    if contains is not None:
        return contains in _norm_str(cand)
    if aliases is not None:
        c = _norm_str(cand)
        return c in aliases or any(len(a) > 3 and a in c for a in aliases)
    if typ in ("text", "code") or typ.startswith("enum"):
        return _norm_str(gold) == _norm_str(cand)
    if typ.startswith("set"):
        if not isinstance(cand, list):
            return False
        return {_norm_str(x) for x in gold} == {_norm_str(x) for x in cand}
    if typ in ("int", "number"):
        try:
            return math.isclose(float(gold), float(cand), abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    if typ == "bool":
        return bool(gold) == bool(cand) and isinstance(cand, bool)
    if typ == "datetime":
        return _parse_dt(gold).replace(second=0) == _parse_dt(cand).replace(second=0)
    if typ == "date":
        return _parse_dt(gold).date() == _parse_dt(cand).date()
    if typ == "time":
        return _parse_time(gold) == _parse_time(cand)
    raise ValueError(typ)


def grade_call(task: Task, call: dict) -> Grade:
    gold = task.gold
    if not isinstance(call, dict) or call.get("name") != gold["name"]:
        return Grade(False, 0.0, {"error": f"wrong function: {call.get('name') if isinstance(call, dict) else call}"})
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        return Grade(False, 0.0, {"error": "arguments not an object"})
    params = API[gold["name"]]["params"]
    unknown = sorted(set(args) - set(params))
    contains = task.grading.get("contains", {})
    aliases = task.grading.get("aliases", {})
    null_ok = task.grading.get("aliases_null_ok", {})  # gold is null, but these values mean the same
    wrong = []
    for p, (typ, _req, default, _d) in params.items():
        g = gold["arguments"].get(p, default)
        c = args.get(p, default)
        if g is None and c is not None and p in null_ok:
            c = None if arg_equal(typ, "x", c, None, null_ok[p]) else c
        try:
            ok = arg_equal(typ, g, c, contains.get(p), aliases.get(p))
        except (ValueError, TypeError):
            ok = False
        if not ok:
            wrong.append(p)
    n = len(params)
    correct = not wrong and not unknown
    return Grade(correct, 1.0 if correct else (n - len(wrong)) / n,
                 {"wrong_args": wrong, "unknown_args": unknown})


def grade_action(task: Task, answer: str) -> Grade:
    try:
        call = json.loads(answer)
    except json.JSONDecodeError as e:
        return Grade(False, 0.0, {"error": f"invalid json: {e}"})
    return grade_call(task, call)


# ---------------------------------------------------------------- math

def grade_math(task: Task, answer: str) -> Grade:
    val = float(answer)
    ok = math.isclose(val, float(task.gold), abs_tol=0.005)
    return Grade(ok, 1.0 if ok else 0.0, {"value": val, "gold": task.gold})


# ---------------------------------------------------------------- imported (Laban et al.) tasks

# Same standard-library imports their code evaluator prepends (tasks/code/eval_code.py).
_LIC_IMPORTS = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string, re, datetime, collections, heapq, bisect, copy, math, random, statistics
import itertools, functools, operator, io, sys, json
sys.setrecursionlimit(50000)
"""

_LIC_HARNESS = textwrap.dedent('''
    import json, signal, sys
    payload = json.loads(sys.stdin.read())
    class _Timeout(Exception):
        pass
    def _alarm(*_):
        raise _Timeout()
    signal.signal(signal.SIGALRM, _alarm)
    ns = {}
    try:
        exec(compile(payload["code"], "candidate.py", "exec"), ns)
        target = ns["Solution"]() if "class Solution" in payload["code"] else None
        fn = getattr(target, payload["fn_name"]) if target is not None else ns[payload["fn_name"]]
    except Exception as e:
        print(json.dumps({"load_error": repr(e)[:300]}))
        sys.exit(0)
    passed = 0
    for inp, out in zip(payload["inputs"], payload["outputs"]):
        try:
            args = [json.loads(line) for line in inp.split("\\n")]
            expected = json.loads(out)
            signal.alarm(6)
            pred = fn(*args)
            signal.alarm(0)
            if isinstance(pred, tuple):
                pred = list(pred)
            if pred != expected:
                print(json.dumps({"passed": passed, "fail": "wrong answer", "input": inp[:200],
                                  "got": repr(pred)[:200], "expected": out[:200]}))
                sys.exit(0)
            passed += 1
        except BaseException as e:
            signal.alarm(0)
            print(json.dumps({"passed": passed, "fail": repr(e)[:200], "input": inp[:200]}))
            sys.exit(0)
    print(json.dumps({"passed": passed}))
''')


def grade_lic_code(task: Task, code: str) -> Grade:
    """Their protocol: rename the function to the expected name, prepend standard imports
    (plus any imports from the HumanEval prompt), run every public and private test, compare
    with == after converting tuples to lists."""
    g = task.grading
    code = code.replace("```python", "").replace("```", "")
    if "def " not in code:
        return Grade(False, 0.0, {"error": "no function"})
    old = code.split("def ")[1].split("(")[0].strip()
    code = code.replace(old, g["fn_name"])
    extra = "\n".join(ln for ln in g.get("prompt_imports", "").splitlines()
                      if ln.startswith(("import ", "from ")))
    payload = json.dumps({"code": _LIC_IMPORTS + extra + "\n\n" + code, "fn_name": g["fn_name"],
                          "inputs": g["inputs"], "outputs": g["outputs"]})
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.py"
        harness.write_text(_LIC_HARNESS)
        try:
            proc = subprocess.run([sys.executable, "-I", str(harness)], input=payload, cwd=tmp,
                                  capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return Grade(False, 0.0, {"error": "timeout"})
    try:
        res = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return Grade(False, 0.0, {"error": "harness crashed", "stderr": proc.stderr[-300:]})
    if "load_error" in res:
        return Grade(False, 0.0, {"error": res["load_error"]})
    n = len(g["inputs"])
    ok = res["passed"] == n
    return Grade(ok, res["passed"] / n if n else 0.0, res | {"total": n})


def grade_lic_math(task: Task, answer: str) -> Grade:
    """Their GSM8K normalization (tasks/math/task_math.py): flexible number regex, strip , $ and
    a trailing period, then exact string match with the gold answer."""
    gold = task.gold.lower()
    try:
        m = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", answer.strip())[-1]
        ext = [x for x in m if x][0]
    except IndexError:
        return Grade(False, 0.0, {"error": "no number"})
    if "$" in ext and ext.endswith(".00"):
        ext = ext[:-3]
    for rx in [",", "\\$", "(?s).*#### ", "\\.$"]:
        ext = re.sub(rx, "", ext)
        gold = re.sub(rx, "", gold)
    # A model writing 18.0 for 18 is right; their exact-string rule would mark it wrong.
    ok = ext == gold or (_is_num(ext) and _is_num(gold) and float(ext) == float(gold))
    return Grade(ok, 1.0 if ok else 0.0, {"value": ext, "gold": gold})


def _is_num(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def cumulative_calls_correct(task: Task, assistant_texts: list[str]) -> bool:
    """Lenient actions score: every reference call was issued in *some* turn. This credits the
    incremental style (one new call per turn, as an agent executing calls would), which the
    strict last-message rule marks wrong. It ignores extra or superseded calls, so it is an
    upper bound."""
    from .vendor.lic_bfcl import ast_parse, find_description, simple_function_checker

    made = []
    for text in assistant_texts:
        calls = extract_calls(text)
        if calls:
            made += ast_parse(calls, "Python")
    fn = task.grading["function"]
    return all(any(simple_function_checker(find_description(fn, list(ref)[0]), m, ref, "Python", "gpt-4o")["valid"]
                   for m in made) for ref in task.gold)


def grade_lic_actions(task: Task, answer: str) -> Grade:
    from .vendor.lic_bfcl import ast_checker, ast_parse

    g = task.grading
    try:
        calls = ast_parse(answer.strip(), g["language"])
    except Exception as e:
        return Grade(False, 0.0, {"error": f"unparseable: {e!r}"[:200]})
    res = ast_checker(g["function"], calls, task.gold, g["language"], g["test_category"], "gpt-4o")
    return Grade(bool(res["valid"]), 1.0 if res["valid"] else 0.0,
                 {"error": json.dumps(res.get("error"), default=str)[:300]})


def gold_answer_text(task: Task) -> str:
    """Render the gold answer as a correctly formatted assistant deliverable."""
    if task.domain == "code":
        return f"```python\n{task.gold.strip()}\n```"
    if task.domain == "sql":
        return f"```sql\n{task.gold.strip()}\n```"
    if task.domain == "action":
        return f"```json\n{json.dumps(task.gold, indent=2)}\n```"
    if task.domain == "lic_actions":
        return reference_calls(task)
    if task.domain == "lic_code":
        raise ValueError("imported code tasks ship tests but no reference solution")
    return f"ANSWER: {task.gold}"


def reference_calls(task: Task) -> str:
    """Render BFCL reference answers as a call string, taking the first acceptable value per
    parameter and omitting parameters whose first acceptable value is '' (optional)."""
    def first(v):
        if isinstance(v, dict):  # nested dicts also list acceptable values per key
            return {k: vv[0] for k, vv in v.items() if vv and vv[0] != ""}
        return v

    parts = []
    for entry in task.gold:
        (name, params), = entry.items()
        args = ", ".join(f"{k}={first(v[0])!r}" for k, v in params.items() if v and v[0] != "")
        parts.append(f"{name}({args})")
    return "[" + ", ".join(parts) + "]"
