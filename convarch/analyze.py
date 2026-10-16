"""Aggregate episodes.jsonl into a comparison report.

Headline metric: accuracy of the final delivered answer. Also reported:
  * paired difference vs. the `single` baseline (same task and seed, so the same persona),
    plus two_agent vs. single_clarify. All CIs resample whole tasks (cluster bootstrap),
    because repeated runs of one task are correlated;
  * Laban-style aptitude (P90 of per-run scores per task) and unreliability (P90 - P10),
    averaged over tasks; these need at least ~5 seeds to mean much;
  * failure-mode metrics: premature attempts, how much was known at the first attempt,
    answers that were right at some point but wrong at the end, question turns;
  * cost per episode and per correct answer.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .runner import CONDITIONS

BASELINE = "single"


def load(path: Path) -> tuple[list[dict], list[dict]]:
    ok, err = [], []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            (err if r.get("error") else ok).append(r)
    # Keep the latest record per episode key (re-runs after errors append new lines).
    latest = {}
    for r in ok:
        latest[(r["task_id"], r["condition"], r["seed"])] = r
    return list(latest.values()), err


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _cluster_ci(groups: dict, n=4000, seed=0):
    """95% CI of the pooled mean, resampling whole tasks (clusters) with replacement.
    Runs of the same task are correlated, so resampling episodes would overstate confidence."""
    keys = list(groups)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        vals = [v for k in rng.choices(keys, k=len(keys)) for v in groups[k]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def _by_task(rows, value):
    groups = defaultdict(list)
    for r in rows:
        groups[r["task_id"]].append(value(r))
    return groups


def _paired(rows_a, rows_b) -> dict | None:
    """Paired difference a - b in accuracy over shared (task, seed) episodes."""
    base = {(r["task_id"], r["seed"]): float(r["correct"]) for r in rows_b}
    groups = defaultdict(list)
    for r in rows_a:
        k = (r["task_id"], r["seed"])
        if k in base:
            groups[r["task_id"]].append(float(r["correct"]) - base[k])
    diffs = [d for v in groups.values() for d in v]
    if not diffs:
        return None
    task_means = [sum(v) / len(v) for v in groups.values()]
    return {"n_pairs": len(diffs), "n_tasks": len(groups), "delta": _mean(diffs), "ci95": _cluster_ci(groups),
            "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
            "tasks_better": sum(m > 0 for m in task_means), "tasks_worse": sum(m < 0 for m in task_means)}


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(rows: list[dict]) -> dict:
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    conds = [c for c in CONDITIONS if c in by_cond]

    out = {"conditions": {}, "paired_vs_baseline": {}, "by_domain": {}, "n_episodes": len(rows)}
    for c in conds:
        rs = by_cond[c]
        acc = [float(r["correct"]) for r in rs]
        per_task = defaultdict(list)
        for r in rs:
            per_task[r["task_id"]].append(r["score"])
        apt = _mean([_pct(v, 0.9) for v in per_task.values()])
        unrel = _mean([_pct(v, 0.9) - _pct(v, 0.1) for v in per_task.values()])
        n_correct = sum(r["correct"] for r in rs)
        cost = sum(r["agent_cost_usd"] for r in rs)
        attempts = [r for r in rs if r["n_attempts"]]
        out["conditions"][c] = {
            "n": len(rs),
            "accuracy": _mean(acc),
            "accuracy_ci95": _cluster_ci(_by_task(rs, lambda r: float(r["correct"]))),
            "mean_score": _mean([r["score"] for r in rs]),
            "aptitude_p90": apt,
            "unreliability_p90_p10": unrel,
            "first_attempt_accuracy": _mean([float(r["first_attempt_correct"]) for r in attempts]),
            "ever_correct": _mean([float(r["ever_correct"]) for r in rs]),
            "lost_after_correct": _mean([float(r["ever_correct"] and not r["correct"]) for r in rs]),
            "no_answer": _mean([float(r["n_attempts"] == 0) for r in rs]),
            "attempts_per_episode": _mean([r["n_attempts"] for r in rs]),
            "premature_attempts_per_episode": _mean([r["n_premature_attempts"] for r in rs]),
            "frac_shards_known_at_first_attempt": _mean(
                [r["first_attempt_shards_known"] / r["n_shards"] for r in attempts]),
            "assistant_turns": _mean([r["n_assistant_turns"] for r in rs]),
            "question_turns": _mean([r["n_question_turns"] for r in rs]),
            "agent_cost_per_episode": cost / len(rs),
            "agent_cost_per_correct": cost / n_correct if n_correct else float("nan"),
            "agent_output_tokens_per_episode": _mean([r["agent_output_tokens"] for r in rs]),
            "user_sim_cost_per_episode": _mean([r["usage"]["user"]["cost_usd"] for r in rs]),
            "refusals": sum(r["refusals"] for r in rs),
        }
        for d in sorted({r["domain"] for r in rs}):
            dr = [r for r in rs if r["domain"] == d]
            out["by_domain"].setdefault(d, {})[c] = {"n": len(dr), "accuracy": _mean([float(r["correct"]) for r in dr])}
        out["conditions"][c]["lenient_accuracy"] = _mean([float(r.get("lenient_correct", r["correct"])) for r in rs])
        cum = [r for r in rs if r.get("cumulative_correct") is not None]
        out["conditions"][c]["actions_cumulative_accuracy"] = _mean([float(r["cumulative_correct"]) for r in cum])
        out["conditions"][c]["actions_strict_accuracy"] = _mean([float(r["correct"]) for r in cum])
        rev = [r for r in rs if r["has_revision"]]
        out["conditions"][c]["accuracy_revision_tasks"] = _mean([float(r["correct"]) for r in rev])

    if BASELINE in by_cond:
        for c in conds:
            if c != BASELINE:
                p = _paired(by_cond[c], by_cond[BASELINE])
                if p:
                    out["paired_vs_baseline"][c] = p
    out["paired_other"] = {}
    for a, b in [("two_agent", "single_clarify"), ("two_agent_verbatim", "two_agent")]:
        if a in by_cond and b in by_cond:
            p = _paired(by_cond[a], by_cond[b])
            if p:
                out["paired_other"][f"{a} vs {b}"] = p
    return out


def _f(x, pct=False, money=False):
    if x != x:  # nan
        return "n/a"
    if money:
        return f"${x:.4f}"
    return f"{100 * x:.1f}%" if pct else f"{x:.2f}"


def render_markdown(s: dict, errors: list[dict], title: str) -> str:
    conds = list(s["conditions"])
    L = [f"# {title}", "", f"{s['n_episodes']} completed episodes, {len(errors)} errored.", ""]
    L += ["## Accuracy of the final answer", "",
          "| condition | n | accuracy | 95% CI (task bootstrap) | Δ vs single (paired) | 95% CI | wins/losses |",
          "|---|---|---|---|---|---|---|"]
    for c in conds:
        m = s["conditions"][c]
        p = s["paired_vs_baseline"].get(c)
        lo, hi = m["accuracy_ci95"]
        if p:
            plo, phi = p["ci95"]
            pd = f"{100 * p['delta']:+.1f} pts | [{100 * plo:+.1f}, {100 * phi:+.1f}] | {p['wins']}/{p['losses']}"
        else:
            pd = "baseline | | " if c == BASELINE else "| | "
        L.append(f"| {c} | {m['n']} | {_f(m['accuracy'], True)} | [{_f(lo, True)}, {_f(hi, True)}] | {pd} |")

    if s.get("paired_other"):
        L += ["", "## Other paired comparisons", "",
              "| comparison | pairs | tasks | Δ accuracy | 95% CI (task bootstrap) | episode wins/losses | tasks better/worse |",
              "|---|---|---|---|---|---|---|"]
        for name, p in s["paired_other"].items():
            lo, hi = p["ci95"]
            L.append(f"| {name} | {p['n_pairs']} | {p['n_tasks']} | {100 * p['delta']:+.1f} pts | "
                     f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {p['wins']}/{p['losses']} | {p['tasks_better']}/{p['tasks_worse']} |")
    L += ["", "## By domain (accuracy)", "", "| domain | " + " | ".join(conds) + " |",
          "|---|" + "---|" * len(conds)]
    for d, cs in s["by_domain"].items():
        L.append(f"| {d} | " + " | ".join(_f(cs[c]["accuracy"], True) if c in cs else "" for c in conds) + " |")

    rows = [
        ("Aptitude (P90 score per task)", "aptitude_p90", "pct"),
        ("Unreliability (P90 - P10)", "unreliability_p90_p10", "pct"),
        ("Accuracy on tasks with a revision", "accuracy_revision_tasks", "pct"),
        ("Accuracy allowing extra SQL columns", "lenient_accuracy", "pct"),
        ("Imported actions: strict (last message has every call)", "actions_strict_accuracy", "pct"),
        ("Imported actions: every call issued in some turn", "actions_cumulative_accuracy", "pct"),
        ("First-attempt accuracy", "first_attempt_accuracy", "pct"),
        ("Correct at some point", "ever_correct", "pct"),
        ("Correct earlier, wrong at the end", "lost_after_correct", "pct"),
        ("Never answered", "no_answer", "pct"),
        ("Answer attempts / episode", "attempts_per_episode", "num"),
        ("Premature attempts / episode", "premature_attempts_per_episode", "num"),
        ("Share of shards known at first attempt", "frac_shards_known_at_first_attempt", "pct"),
        ("Assistant turns / episode", "assistant_turns", "num"),
        ("Question-only turns / episode", "question_turns", "num"),
        ("Agent output tokens / episode", "agent_output_tokens_per_episode", "num"),
        ("Agent cost / episode", "agent_cost_per_episode", "money"),
        ("Agent cost / correct answer", "agent_cost_per_correct", "money"),
        ("User-simulator cost / episode", "user_sim_cost_per_episode", "money"),
        ("Refusals", "refusals", "num"),
    ]
    L += ["", "## Behavior and cost", "", "| metric | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for label, key, kind in rows:
        vals = [s["conditions"][c][key] for c in conds]
        L.append(f"| {label} | " + " | ".join(
            _f(v, pct=kind == "pct", money=kind == "money") for v in vals) + " |")
    if errors:
        L += ["", "## Errors", ""] + [f"- {e['task_id']} / {e['condition']} / s{e['seed']}: {e['error'][:160]}"
                                      for e in errors[:20]]
    return "\n".join(L) + "\n"


def analyze(run_dir: Path) -> dict:
    rows, errors = load(run_dir / "episodes.jsonl")
    s = summarize(rows)
    (run_dir / "summary.json").write_text(json.dumps(s, indent=2))
    md = render_markdown(s, errors, f"Results: {run_dir.name}")
    (run_dir / "report.md").write_text(md)
    return {"summary": s, "markdown": md}
