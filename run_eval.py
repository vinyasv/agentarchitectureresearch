"""CLI: run the sharded-conversation eval and write a report.

Examples:
  python3 run_eval.py --dry-run                         # offline plumbing check, no API calls
  python3 run_eval.py --run pilot --seeds 1 --limit 20  # small paid pilot to measure cost
  python3 run_eval.py --run main --seeds 3              # full comparison
  python3 run_eval.py --run main --analyze-only
"""

import argparse
import asyncio
import os
import random
from pathlib import Path


def load_dotenv(path: Path = Path(__file__).with_name(".env")) -> None:
    """Set KEY=VALUE lines from .env, without overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


load_dotenv()

from convarch.analyze import analyze
from convarch.llm import DEFAULT_AGENT_MODEL, DEFAULT_USER_MODEL
from convarch.runner import CONDITIONS, Config, run_batch
from convarch.tasks import get_suite
from convarch.tasks.sqldb import ensure_db


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run name; results go to results/<run>/")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=3, help="runs per task and condition; seed picks the persona")
    ap.add_argument("--suite", default="ours", choices=["ours", "lic"],
                    help="ours: the 24 hand-built tasks; lic: Laban et al.'s code/math/actions instructions")
    ap.add_argument("--protocol", default="realistic", choices=["realistic", "laban"],
                    help="user simulation and stopping rules (see convarch/usersim.py)")
    ap.add_argument("--tasks", default="", help="comma-separated task-id prefixes or domains to include")
    ap.add_argument("--per-domain", type=int, default=None, help="random sample of N tasks per domain")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    ap.add_argument("--user-model", default=DEFAULT_USER_MODEL)
    ap.add_argument("--agent-effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--user-effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--max-per-turn", type=int, default=2, help="max shards the user shares in one turn when asked")
    ap.add_argument("--concurrency", type=int, default=6, help="episodes in flight")
    ap.add_argument("--limit", type=int, default=None, help="run at most N new episodes (pilots)")
    ap.add_argument("--dry-run", action="store_true", help="scripted stubs instead of the API")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()

    run = a.run or ("dryrun" if a.dry_run else "main")
    run_dir = Path("results") / run
    if not a.analyze_only:
        ensure_db()
        conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
        unknown = set(conditions) - set(CONDITIONS)
        if unknown:
            ap.error(f"unknown conditions: {sorted(unknown)}")
        tasks = get_suite(a.suite)
        if a.tasks:
            keys = [k.strip() for k in a.tasks.split(",")]
            tasks = [t for t in tasks if t.domain in keys or any(t.id.startswith(k) for k in keys)]
        if a.per_domain:
            rng = random.Random(a.sample_seed)
            by_dom: dict[str, list] = {}
            for t in tasks:
                by_dom.setdefault(t.domain, []).append(t)
            tasks = [t for d in sorted(by_dom) for t in rng.sample(by_dom[d], min(a.per_domain, len(by_dom[d])))]
        cfg = Config(agent_model=a.agent_model, user_model=a.user_model, agent_effort=a.agent_effort,
                     user_effort=a.user_effort, max_per_turn=a.max_per_turn, protocol=a.protocol,
                     dry_run=a.dry_run)
        asyncio.run(run_batch(tasks, conditions, list(range(a.seeds)), cfg, run_dir / "episodes.jsonl",
                              a.concurrency, a.limit))
    print()
    print(analyze(run_dir)["markdown"])


if __name__ == "__main__":
    main()
