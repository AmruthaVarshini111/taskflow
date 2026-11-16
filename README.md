# TaskFlow - multi-agent task executor with a live dashboard

Give it a goal. A **planner** agent splits it into subtasks with dependencies, **worker**
agents run them in parallel (up to 4 at once), and an **aggregator** merges the results.
The dashboard shows the queue, running tasks, blockers, retries, live logs and the final answer.

```
goal -> Planner -> [subtask DAG] -> parallel Workers (retry on transient errors)
                                        -> Aggregator -> final answer
                       |
                       +-- every state change = event -> SSE -> dashboard
```

## Run it

```bash
cd taskflow
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cp .env.example .env                                    # leave LLM_PROVIDER=mock to start
cd backend && uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 and click a suggested goal. In mock mode `t3` fails once on purpose
so you can see the retry path.

### Use real Gemini
1. Get a free key at https://aistudio.google.com/apikey
2. In `.env`: `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...`
3. Restart uvicorn.

### Turn on LangSmith
In `.env` set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...`. Every planner / worker /
aggregator call appears as a trace at smith.langchain.com.

## Layout
| File | What it does |
|---|---|
| `backend/orchestrator.py` | Plan validation (cycles, bad deps), dependency-aware parallel scheduler, retries, blocker propagation, event log |
| `backend/llm.py` | `MockLLM` and `GeminiLLM` behind the same 3 methods: `plan`, `work`, `aggregate` |
| `backend/main.py` | FastAPI: `POST /api/runs`, `GET /api/runs/{id}`, SSE `GET /api/runs/{id}/events` |
| `frontend/index.html` | Dashboard (no build step, no dependencies) |

## Ideas to extend (good portfolio upgrades)
1. **Dependency graph view** - draw the subtask DAG (React Flow, or SVG) next to the board.
2. **Persistence** - store runs/events in SQLite so history survives restarts.
3. **Human-in-the-loop** - when a worker returns `BLOCKED:`, let the user type the missing info and resume that task.
4. **Tools for workers** - web search, code execution, file reading.
5. **Cost tracking** - tokens and estimated cost per subtask.
6. **Move to LangGraph** - the scheduler in `orchestrator.py` maps directly onto a LangGraph state graph.
7. **Next.js frontend** - port `index.html` once you want routing and components.
