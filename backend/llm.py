"""LLM adapters: MockLLM (no API key, for demos/tests) and GeminiLLM (real).

Every adapter exposes three async methods used by the orchestrator:
    plan(goal)                     -> list[dict]   subtasks (id, title, description, deps)
    work(goal, task, dep_results)  -> str          result text for one subtask
    aggregate(goal, results)       -> str          final combined answer

LangSmith tracing is optional: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY
and every call below shows up as a trace. If `langsmith` isn't installed the
decorator is a no-op.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re

try:  # optional tracing
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*dargs, **dkwargs):
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return lambda fn: fn


class TransientError(Exception):
    """Raised by an LLM call that is worth retrying (rate limit, timeout...)."""


class BlockerError(Exception):
    """Raised when a task cannot proceed (missing info, bad input...). Not retried."""


# --------------------------------------------------------------------------- #
# Mock LLM
# --------------------------------------------------------------------------- #
class MockLLM:
    name = "mock"

    def __init__(self, speed: float = 1.0, seed: int | None = None):
        self.speed = speed
        self.rng = random.Random(seed)
        self._failed_once: set[str] = set()

    @traceable(name="mock_plan")
    async def plan(self, goal: str) -> list[dict]:
        await asyncio.sleep(0.8 * self.speed)
        return [
            {"id": "t1", "title": "Clarify scope", "description": f"Define what '{goal}' needs to cover.", "deps": []},
            {"id": "t2", "title": "Research background", "description": "Collect key facts and prior art.", "deps": ["t1"]},
            {"id": "t3", "title": "List constraints", "description": "Budget, time, tools, risks.", "deps": ["t1"]},
            {"id": "t4", "title": "Draft outline", "description": "Structure the deliverable.", "deps": ["t2", "t3"]},
            {"id": "t5", "title": "Write first draft", "description": "Fill the outline with content.", "deps": ["t4"]},
            {"id": "t6", "title": "Find examples", "description": "Pull concrete examples to illustrate.", "deps": ["t2"]},
            {"id": "t7", "title": "Review & polish", "description": "Merge draft and examples, fix gaps.", "deps": ["t5", "t6"]},
        ]

    @traceable(name="mock_work")
    async def work(self, goal: str, task: dict, dep_results: dict[str, str]) -> str:
        await asyncio.sleep(self.rng.uniform(1.2, 3.2) * self.speed)
        # Demo the retry path: t3 fails once with a transient error.
        if task["id"] == "t3" and task["id"] not in self._failed_once:
            self._failed_once.add(task["id"])
            raise TransientError("simulated rate limit (429)")
        return f"[mock] Finished '{task['title']}' using {len(dep_results)} upstream result(s)."

    @traceable(name="mock_aggregate")
    async def aggregate(self, goal: str, results: dict[str, str]) -> str:
        await asyncio.sleep(0.8 * self.speed)
        lines = "\n".join(f"- {k}: {v}" for k, v in results.items())
        return f"Goal: {goal}\n\nCombined output from {len(results)} subtasks:\n{lines}"


# --------------------------------------------------------------------------- #
# Gemini LLM
# --------------------------------------------------------------------------- #
PLAN_PROMPT = """You are a planning agent. Break the user's goal into 4-10 subtasks that
can be executed by independent worker agents. Subtasks that don't depend on each
other should have no shared dependency so they can run in parallel.

Return ONLY JSON, no prose, in this shape:
{{"tasks": [{{"id": "t1", "title": "short title", "description": "what to do", "deps": []}}]}}
Rules: ids are t1, t2, ...; deps list ids of subtasks that must finish first; no cycles.

Goal: {goal}"""

WORK_PROMPT = """You are a worker agent completing ONE subtask of a larger goal.

Overall goal: {goal}
Your subtask: {title}
Details: {description}

Results from subtasks you depend on:
{deps}

Do the subtask now. Be concrete and concise (under 200 words). If you cannot proceed
because required information is missing, reply with exactly: BLOCKED: <reason>"""

AGG_PROMPT = """You are the final aggregator agent. Combine the subtask results below
into one coherent, well-structured answer to the goal.

Goal: {goal}

Subtask results:
{results}"""


class GeminiLLM:
    name = "gemini"

    def __init__(self):
        from google import genai  # pip install google-genai

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or run with LLM_PROVIDER=mock)")
        self.client = genai.Client(api_key=key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    
    async def _call(self, prompt: str, json_mode: bool = False) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            response_mime_type="application/json" if json_mode else "text/plain"
        )

        for attempt in range(4):
            try:
                resp = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=cfg
                )
                return (resp.text or "").strip()

            except Exception as e:
                message = str(e)
                status = (
                    getattr(e, "code", None)
                    or getattr(e, "status_code", None)
                )

                retryable = (
                    status in (429, 500, 502, 503, 504)
                    or any(
                        code in message.upper()
                        for code in (
                            "429", "500", "502", "503", "504",
                            "UNAVAILABLE", "RESOURCE_EXHAUSTED"
                        )
                    )
                )

                if not retryable or attempt == 3:
                    raise TransientError(message) from e

                delay = 2 ** (attempt + 1)
                await asyncio.sleep(delay)


    @traceable(name="gemini_plan")
    async def plan(self, goal: str) -> list[dict]:
        raw = await self._call(PLAN_PROMPT.format(goal=goal), json_mode=True)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        data = json.loads(raw)
        return data["tasks"] if isinstance(data, dict) else data

    @traceable(name="gemini_work")
    async def work(self, goal: str, task: dict, dep_results: dict[str, str]) -> str:
        deps = "\n".join(f"- {k}: {v}" for k, v in dep_results.items()) or "(none)"
        text = await self._call(
            WORK_PROMPT.format(goal=goal, title=task["title"], description=task["description"], deps=deps)
        )
        if text.upper().startswith("BLOCKED:"):
            raise BlockerError(text[8:].strip() or "worker reported it is blocked")
        return text

    @traceable(name="gemini_aggregate")
    async def aggregate(self, goal: str, results: dict[str, str]) -> str:
        joined = "\n\n".join(f"## {k}\n{v}" for k, v in results.items())
        return await self._call(AGG_PROMPT.format(goal=goal, results=joined))


def make_llm():
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    if provider == "gemini":
        return GeminiLLM()
    return MockLLM(speed=float(os.getenv("MOCK_SPEED", "1.0")))
