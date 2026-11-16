"""FastAPI app: starts runs, streams events (SSE), serves the dashboard.

Run:  uvicorn main:app --reload --port 8000     (from the backend/ folder)
Open: http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

try:  # load .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception:
    pass

from llm import make_llm
from orchestrator import Orchestrator, Run

app = FastAPI(title="TaskFlow")
RUNS: dict[str, Run] = {}
BG: set[asyncio.Task] = set()  # keep references so tasks aren't garbage collected
FRONTEND = Path(__file__).parent.parent / "frontend"


class NewRun(BaseModel):
    goal: str = Field(min_length=3, max_length=2000)


@app.post("/api/runs")
async def create_run(body: NewRun):
    run = Run(goal=body.goal.strip())
    RUNS[run.id] = run
    task = asyncio.create_task(Orchestrator(make_llm()).execute(run))
    BG.add(task)
    task.add_done_callback(BG.discard)
    return {"id": run.id}


@app.get("/api/runs")
async def list_runs():
    runs = sorted(RUNS.values(), key=lambda r: r.created_at, reverse=True)
    return [{"id": r.id, "goal": r.goal, "status": r.status, "created_at": r.created_at} for r in runs]


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run.snapshot()


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, last: int = 0):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")

    async def gen():
        async for ev in run.stream(start=last):
            yield f"id: {ev['i']}\ndata: {json.dumps(ev)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/health")
async def health():
    return {"ok": True, "provider": os.getenv("LLM_PROVIDER", "mock")}


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html")
