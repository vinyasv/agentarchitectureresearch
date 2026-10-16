"""Re-grade a run's saved transcripts with the current extractor/graders (no API calls).

Usage: python3 scripts/regrade.py <run-name>   (rewrites results/<run>/episodes.jsonl, keeps a .bak)
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convarch.analyze import analyze  # noqa: E402
from convarch.runner import regrade  # noqa: E402
from convarch.tasks import all_tasks, lic_tasks  # noqa: E402

run_dir = Path("results") / sys.argv[1]
path = run_dir / "episodes.jsonl"
shutil.copy(path, path.with_suffix(".jsonl.bak"))
tasks = {t.id: t for t in all_tasks() + lic_tasks()}
out, changed = [], 0
for line in path.read_text().splitlines():
    rec = json.loads(line)
    if not rec.get("error"):
        before = rec["correct"]
        regrade(tasks[rec["task_id"]], rec)
        if rec["correct"] != before:
            changed += 1
            print(f"changed: {rec['task_id']} / {rec['condition']} / s{rec['seed']}: {before} -> {rec['correct']}")
    out.append(json.dumps(rec))
path.write_text("\n".join(out) + "\n")
print(f"{changed} episode outcomes changed")
print(analyze(run_dir)["markdown"])
