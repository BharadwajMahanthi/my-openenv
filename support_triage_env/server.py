"""FastAPI server exposing the OpsSupportEnv over HTTP."""
from __future__ import annotations

import os
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .env import OpsSupportEnv
from .tasks import TASK_REGISTRY

app = FastAPI(title="Ops Support Triage", version="1.0.0", description="OpenEnv support workflow simulator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_SESSIONS: Dict[str, OpsSupportEnv] = {}


def _ensure_session(session_id: str) -> OpsSupportEnv:
    env = _SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return env


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Ops Support Triage OpenEnv"}


@app.get("/tasks")
def list_tasks() -> Dict[str, list[str]]:
    return {"tasks": list(TASK_REGISTRY.keys())}


@app.post("/reset")
def reset(payload: Optional[Dict[str, str]] = None):
    task_id = (payload or {}).get("task_id") if payload else None
    env = OpsSupportEnv(task_id=task_id)
    observation = env.reset(task_id)
    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = env
    return {"session_id": session_id, "observation": observation.dict()}


@app.post("/step")
def step(payload: Dict):
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    env = _ensure_session(session_id)
    action = payload.get("action")
    if action is None:
        raise HTTPException(status_code=400, detail="action payload missing")
    result = env.step(action)
    if result.done:
        _SESSIONS.pop(session_id, None)
    return {
        "observation": result.observation.dict(),
        "reward": result.reward.dict(),
        "done": result.done,
        "info": {
            "task_score": result.info["task_score"],
            "requirements": [req.dict() for req in result.info["requirements"]],
        },
    }


@app.get("/state/{session_id}")
def get_state(session_id: str):
    env = _ensure_session(session_id)
    return env.state().dict()


def run_server() -> None:
    """Entry point compatible with setuptools console_scripts."""
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
