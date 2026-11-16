"""Core orchestration: plan -> schedule subtasks in parallel -> aggregate.

Pure stdlib asyncio (no web framework) so it can be tested on its own.
A Run records every state change as an event; the API streams those events
to the dashboard over SSE.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field, asdict

from llm import BlockerError, TransientError

MAX_CONCURRENCY = 4   # how many subtasks run at once
MAX_ATTEMPTS = 3      # 1 try + 2 retries for transient errors

# status values: pending | running | done | failed | blocked
#   pending -> waiting in the queue (deps not finished or no free worker)
#   blocked -> cannot ever run (an upstream task failed) or worker reported a blocker


@dataclass
class Task:
    id: str
    title: str
    description: str
    deps: list[str] = field(default_factory=list)
    status: str = "pending"
    result: str | None = None
    error: str | None = None
    attempts: int = 0
    started_at: float | None = None
    finished_at: float | None = None


@dataclass
class Run:
    goal: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "planning"  # planning | running | done | failed
    tasks: dict[str, Task] = field(default_factory=dict)
    summary: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    events: list[dict] = field(default_factory=list)
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status,
            "summary": self.summary,
            "error": self.error,
            "created_at": self.created_at,
            "tasks": [asdict(t) for t in self.tasks.values()],
        }

    async def emit(self, type_: str, **data):
        async with self._cond:
            self.events.append({"i": len(self.events), "t": time.time(), "type": type_, **data})
            self._cond.notify_all()

    async def stream(self, start: int = 0):
        """Async generator of events from index `start`; ends after run_finished."""
        i = start
        while True:
            async with self._cond:
                await self._cond.wait_for(lambda: len(self.events) > i)
                batch = self.events[i:]
            for ev in batch:
                yield ev
                i = ev["i"] + 1
                if ev["type"] == "run_finished":
                    return


# --------------------------------------------------------------------------- #
def validate_plan(raw: list[dict]) -> list[Task]:
    """Clean up whatever the planner returned: unique ids, valid deps, no cycles."""
    tasks: dict[str, Task] = {}
    for n, item in enumerate(raw, 1):
        tid = str(item.get("id") or f"t{n}")
        if tid in tasks:
            tid = f"{tid}_{n}"
        tasks[tid] = Task(
            id=tid,
            title=str(item.get("title") or tid)[:80],
            description=str(item.get("description") or ""),
            deps=[str(d) for d in item.get("deps", [])],
        )
    if not tasks:
        raise ValueError("planner returned no tasks")
    for t in tasks.values():  # drop unknown / self deps
        t.deps = [d for d in dict.fromkeys(t.deps) if d in tasks and d != t.id]
    # cycle check (Kahn); if cyclic, drop deps of the stuck nodes
    indeg = {i: len(t.deps) for i, t in tasks.items()}
    ready = [i for i, d in indeg.items() if d == 0]
    seen = 0
    while ready:
        cur = ready.pop()
        seen += 1
        for t in tasks.values():
            if cur in t.deps:
                indeg[t.id] -= 1
                if indeg[t.id] == 0:
                    ready.append(t.id)
    if seen != len(tasks):
        for i, d in indeg.items():
            if d > 0:
                tasks[i].deps = []
    return list(tasks.values())


class Orchestrator:
    def __init__(self, llm, max_concurrency: int = MAX_CONCURRENCY):
        self.llm = llm
        self.max_concurrency = max_concurrency

    async def execute(self, run: Run) -> None:
        try:
            await run.emit("run_started", goal=run.goal, llm=self.llm.name)
            # 1) plan
            await run.emit("log", level="info", msg="Planner: breaking goal into subtasks...")
            plan = validate_plan(await self.llm.plan(run.goal))
            run.tasks = {t.id: t for t in plan}
            run.status = "running"
            await run.emit("plan_ready", tasks=[asdict(t) for t in plan])
            await run.emit("log", level="info", msg=f"Planner: created {len(plan)} subtasks")

            # 2) schedule
            await self._schedule(run)

            # 3) aggregate whatever finished
            done = {t.id: t.result for t in run.tasks.values() if t.status == "done"}
            if not done:
                raise RuntimeError("no subtask completed successfully")
            await run.emit("log", level="info", msg="Aggregator: combining results...")
            run.summary = await self.llm.aggregate(run.goal, done)
            failed = [t for t in run.tasks.values() if t.status in ("failed", "blocked")]
            run.status = "done" if not failed else "failed"
            if failed:
                run.error = f"{len(failed)} subtask(s) did not complete"
        except Exception as e:  # planner or aggregator failure
            run.status = "failed"
            run.error = f"{type(e).__name__}: {e}"
            await run.emit("log", level="error", msg=run.error)
        finally:
            await run.emit("run_finished", status=run.status, summary=run.summary, error=run.error)

    # ------------------------------------------------------------------ #
    async def _schedule(self, run: Run) -> None:
        sem = asyncio.Semaphore(self.max_concurrency)
        running: dict[str, asyncio.Task] = {}

        while True:
            # mark tasks whose upstream failed/blocked as blocked
            for t in run.tasks.values():
                if t.status == "pending":
                    bad = [d for d in t.deps if run.tasks[d].status in ("failed", "blocked")]
                    if bad:
                        await self._set(run, t, "blocked", error=f"upstream {', '.join(bad)} did not finish")

            # start everything whose deps are all done
            for t in run.tasks.values():
                if t.status == "pending" and t.id not in running and all(
                    run.tasks[d].status == "done" for d in t.deps
                ):
                    running[t.id] = asyncio.create_task(self._run_task(run, t, sem))

            if not running:
                if any(t.status == "pending" for t in run.tasks.values()):  # safety net
                    for t in run.tasks.values():
                        if t.status == "pending":
                            await self._set(run, t, "blocked", error="unschedulable")
                return

            finished, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)
            for tid in [k for k, v in running.items() if v in finished]:
                running.pop(tid)

    async def _run_task(self, run: Run, t: Task, sem: asyncio.Semaphore) -> None:
        async with sem:
            dep_results = {d: run.tasks[d].result or "" for d in t.deps}
            while t.attempts < MAX_ATTEMPTS:
                t.attempts += 1
                await self._set(run, t, "running")
                try:
                    t.result = await self.llm.work(run.goal, asdict(t), dep_results)
                    await self._set(run, t, "done")
                    return
                except BlockerError as e:
                    await self._set(run, t, "blocked", error=str(e))
                    return
                except TransientError as e:
                    if t.attempts >= MAX_ATTEMPTS:
                        await self._set(run, t, "failed", error=f"gave up after {t.attempts} attempts: {e}")
                        return
                    await run.emit("log", level="warn", task=t.id,
                                   msg=f"{t.title}: {e} - retrying ({t.attempts}/{MAX_ATTEMPTS})")
                    await self._set(run, t, "pending", error=str(e))
                    await asyncio.sleep(min(2 ** t.attempts * 0.5, 8))
                except Exception as e:
                    await self._set(run, t, "failed", error=f"{type(e).__name__}: {e}")
                    return

    async def _set(self, run: Run, t: Task, status: str, error: str | None = None) -> None:
        t.status = status
        if error is not None:
            t.error = error
        if status == "running":
            t.started_at = time.time()
            t.error = None
        if status in ("done", "failed", "blocked"):
            t.finished_at = time.time()
        await run.emit("task_update", task=asdict(t))
        label = {"running": "started", "done": "finished", "failed": "FAILED", "blocked": "BLOCKED"}.get(status)
        if label:
            level = "error" if status in ("failed", "blocked") else "info"
            extra = f" - {t.error}" if t.error and status in ("failed", "blocked") else ""
            await run.emit("log", level=level, task=t.id, msg=f"{t.title}: {label}{extra}")
