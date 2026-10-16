from . import action_tasks, code_tasks, math_tasks, sql_tasks
from .schema import Shard, Task

DOMAINS = {
    "code": code_tasks,
    "sql": sql_tasks,
    "action": action_tasks,
    "math": math_tasks,
}


def all_tasks() -> list[Task]:
    return [t for mod in DOMAINS.values() for t in mod.TASKS]


def lic_tasks() -> list[Task]:
    """Instructions imported from Laban et al.'s released data (code, math, actions)."""
    from .lic_tasks import load
    return load()


def get_suite(name: str) -> list[Task]:
    if name == "ours":
        return all_tasks()
    if name == "lic":
        return lic_tasks()
    raise ValueError(f"unknown suite {name!r}")


def task_context(task: Task) -> str:
    return task.context if task.context is not None else DOMAINS[task.domain].CONTEXT


def task_deliverable(task: Task) -> str:
    return task.deliverable if task.deliverable is not None else DOMAINS[task.domain].DELIVERABLE


__all__ = ["Shard", "Task", "DOMAINS", "all_tasks", "lic_tasks", "get_suite", "task_context",
           "task_deliverable"]
