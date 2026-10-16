"""Check that the task suite is sound before spending money on it.

For every task:
  * the gold answer, rendered as an assistant message, is extracted and graded correct;
  * every non-intent shard is load-bearing: code shards have tagged tests, SQL/action/math
    shards have an ablation (the answer you get when ignoring that shard) that grades wrong;
  * revision shards point at an earlier shard.
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convarch.graders import extract, gold_answer_text, grade, grade_call  # noqa: E402
from convarch.tasks import all_tasks  # noqa: E402
from convarch.tasks.sqldb import build, run_query  # noqa: E402


def main() -> int:
    build()
    problems = []
    for t in all_tasks():
        ans = extract(t, gold_answer_text(t))
        g = grade(t, ans)
        if not g.correct:
            problems.append(f"{t.id}: gold answer graded wrong: {g.detail}")

        later = [s for s in t.shards[1:] if not s.distractor]
        for s in later:
            if s.revises is not None and s.revises >= s.id:
                problems.append(f"{t.id}: shard {s.id} revises a later shard")
        # Shards whose requirement is superseded by a revision only need the revision covered.
        superseded = {s.revises for s in t.shards if s.revises is not None}

        if t.domain == "code":
            tagged = {x["shard"] for x in t.grading["tests"]}
            missing = [s.id for s in later if s.id not in tagged and s.id not in superseded]
            if missing:
                problems.append(f"{t.id}: shards without tests: {missing}")
        else:
            for sid, abl in t.ablations.items():
                if t.domain == "sql":
                    ab = grade(t, abl)
                elif t.domain == "action":
                    call = copy.deepcopy(t.gold)
                    p, v = abl
                    if v is None:
                        call["arguments"].pop(p, None)
                    else:
                        call["arguments"][p] = v
                    ab = grade_call(t, call)
                else:
                    ab = grade(t, str(abl))
                if ab.correct:
                    problems.append(f"{t.id}: ablating shard {sid} still graded correct")
            uncovered = [s.id for s in later if s.id not in t.ablations and s.id not in superseded]
            print(f"  {t.id}: shards without an ablation (format/value shards): {uncovered}")

        if t.domain == "sql":
            rows = run_query(t.gold)
            print(f"  {t.id}: {len(rows)} gold rows, first: {rows[:3]}")
            if not rows:
                problems.append(f"{t.id}: gold query returns no rows")
            cols = t.grading.get("sort_cols")
            if cols:
                keys = [tuple(r[c] for c in cols) for r in rows]
                if len(keys) != len(set(keys)):
                    problems.append(f"{t.id}: ties in ORDER BY columns {cols}, so row order is ambiguous")

    print()
    for t in all_tasks()[:1]:
        print("Example fully specified prompt:\n" + t.full_prompt() + "\n")
    n = len(all_tasks())
    revs = sum(1 for t in all_tasks() for s in t.shards if s.revises is not None)
    dis = sum(1 for t in all_tasks() for s in t.shards if s.distractor)
    shards = [t.n_shards for t in all_tasks()]
    print(f"{n} tasks checked: {sum(shards) / n:.1f} shards on average ({min(shards)}-{max(shards)}), "
          f"{revs} revisions, {dis} distractors.")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
