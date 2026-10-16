"""Sharded instructions from Laban et al. (2025), "LLMs Get Lost in Multi-Turn Conversation".

Source: github.com/microsoft/lost_in_conversation (MIT, see data/lic/LICENSE), commit
c865793f, data/sharded_instructions_600.json, restricted to the code (HumanEval and
LiveCodeBench), math (GSM8K) and actions (BFCL) tasks.

The assistant system prompts and fully specified prompts below are theirs, verbatim. One
thing is added for deterministic grading: a deliverable-format line (a ```python block for
code, an `ANSWER:` line for math). They use an LLM classifier and extractor instead; their
actions prompt already fixes the output format, so nothing is added there.
"""

from __future__ import annotations

import base64
import io
import json
import pickle
import zlib
from functools import lru_cache
from pathlib import Path

from .schema import Shard, Task

DATA = Path(__file__).resolve().parents[2] / "data" / "lic" / "sharded_code_math_actions.json"

LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question (problem specification) and "
    "will generate a correct Python program that matches the specification and passes all tests.\n\n"
    "Format:\n- [Standalone] Make sure that your answer consists of only one Python function at the "
    "top level. Do not wrap with a class or split into multiple functions."
)
MATH_SYSTEM = "As an expert problem solver solve step by step the following mathematical questions."
ACTIONS_SYSTEM = (
    "You are an expert in composing functions. You are given a question and a set of possible "
    "functions. Based on the question, you will need to make one or more function/tool calls to "
    "achieve the purpose.\nIf none of the functions can be used, point it out. If the given question "
    "lacks the parameters required by the function, also point it out.\nYou should only return the "
    "function calls in your response.\n\nIf you decide to invoke any of the function(s), you MUST put "
    "it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), "
    "func_name2(params)]\nYou SHOULD NOT include any other text in the response.\n\nHere is a list of "
    "functions in JSON format that you can invoke.\n\n{functions}"
)

LCB_FULL = "### Question:\n{question}\n\n### Format: {preamble}\n{formatting}\n\n### Answer: (use the provided format with backticks)\n"
LCB_PREAMBLE = ("You will use the following starter code to write the solution to the problem and "
                "enclose your code within delimiters.")
HUMANEVAL_FULL = (
    "You must generate a valid Python function that satisfies the following instruction:\n\n"
    "Instruction:\n\n{instruction}\n\nRules:\n- [Format] You must only return the Python function, "
    "matching the signature exactly.\n- [Validity] The Python function must be valid and executable. "
    "Do not include any other text or comments. (```python```)"
)

CODE_DELIVERABLE = ("Deliverable format: whenever you provide the code, put the complete function in a "
                    "single ```python fenced block.")
MATH_DELIVERABLE = ("Deliverable format: whenever you give a final numeric answer, end your message with "
                    "a line of the form `ANSWER: <number>`.")


class _NoGlobalsUnpickler(pickle.Unpickler):
    """Loads plain data only. Any class or function lookup is refused, so unpickling the
    downloaded test payload cannot run code."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _decode_private_tests(blob: str) -> list[dict]:
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        raw = zlib.decompress(base64.b64decode(blob.encode("utf-8")))
        return json.loads(_NoGlobalsUnpickler(io.BytesIO(raw)).load())


def _shards(sample: dict) -> list[Shard]:
    return [Shard(s["shard_id"], s["shard"]) for s in sorted(sample["shards"], key=lambda s: s["shard_id"])]


def _code_task(s: dict) -> Task:
    tests = json.loads(s["public_test_cases"])
    if s.get("private_test_cases"):
        tests += _decode_private_tests(s["private_test_cases"])
    if s["source"] == "humaneval":
        full = HUMANEVAL_FULL.format(
            instruction=f"Complete the following incomplete function signature:\n```python\n{s['prompt']}\n```")
    else:
        full = LCB_FULL.format(question=s["question_content"], preamble=LCB_PREAMBLE,
                               formatting=f"```python\n{s['starter_code'] or '# YOUR CODE HERE'}\n```")
    return Task(
        id=s["task_id"], domain="lic_code", title=s["task_id"], shards=_shards(s), gold=None,
        grading={"fn_name": s["metadata"]["func_name"],
                 "inputs": [t["input"] for t in tests], "outputs": [t["output"] for t in tests],
                 "prompt_imports": s.get("prompt", "")},
        context=LCB_SYSTEM, deliverable=CODE_DELIVERABLE, full_text=full, source=s["source"])


def _math_task(s: dict) -> Task:
    return Task(
        id=s["task_id"], domain="lic_math", title=s["task_id"], shards=_shards(s),
        gold=s["answer"].split("####")[1].strip(), grading={},
        context=MATH_SYSTEM, deliverable=MATH_DELIVERABLE, full_text=f"Q: {s['question']}\nA:",
        source="gsm8k")


def _actions_task(s: dict) -> Task:
    return Task(
        id=s["task_id"], domain="lic_actions", title=s["task_id"], shards=_shards(s),
        gold=s["reference_answer"],
        grading={"function": s["function"], "language": s["language"], "test_category": s["test_category"]},
        context=ACTIONS_SYSTEM.format(functions="{}".format(s["function"])), deliverable="",
        full_text=s["fully_specified_question"][0][0]["content"], source="bfcl")


@lru_cache(maxsize=1)
def load() -> list[Task]:
    build = {"code": _code_task, "math": _math_task, "actions": _actions_task}
    return [build[s["task"]](s) for s in json.loads(DATA.read_text())]
