---
title: My Env
emoji: 🐠
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Ops Support Triage OpenEnv environment
---

# Ops Support Triage (OpenEnv)

Ops Support Triage is a real-world OpenEnv environment where an agent acts as an operations
specialist triaging healthcare SaaS support tickets. The agent must read an inbox, follow
policy snippets, craft compliant communications, and coordinate billing or incident work.
The environment models everyday work done by support engineers, success leads, and incident
commanders.

## Environment Highlights
- **Real workflows**: password recovery, enterprise billing disputes, and SEV-2 incident coordination.
- **OpenEnv-compliant API**: typed `Observation`, `Action`, and `Reward` models plus `step/ reset/ state` methods.
- **Three graded tasks**: easy -> medium -> hard, each with deterministic graders that emit scores in `[0, 1]`.
- **Shaped rewards**: incremental credit for covering rubric requirements, with penalties for empty/no-op moves.
- **Deployable**: packaged FastAPI server, Dockerfile, and Hugging Face Space entrypoint.

## Repository Structure
```
.
|-- app.py                     # HF Space entrypoint (uvicorn server)
|-- inference.py               # Baseline LLM-driven policy (OpenAI client)
|-- openenv.yaml               # Metadata consumed by openenv validators
|-- requirements.txt
|-- Dockerfile
|-- README.md
`-- support_triage_env/
    |-- __init__.py
    |-- env.py                 # OpsSupportEnv implementation
    |-- models.py              # Typed Observation/Action/Reward/State models
    |-- server.py              # FastAPI wiring (reset/step/state endpoints)
    `-- tasks.py               # Task registry + rubric-based graders
```

## Observation, Action, Reward Spaces
- **Observation** - `Observation` model exposes the task id, difficulty, inbox message history,
  policy snippets, remaining steps, and the agent's cumulative score.
- **Action** - Agents choose between `draft_reply`, `assign`, `resolve`, `clarify`, or `noop` and
  provide structured fields (reply text, assignment target, priority, tags, notes). Reply text is
  required for `draft_reply`/`resolve` per the validator in `Action`.
- **Reward** - Each `step` recomputes rubric coverage; the reward equals `Delta score - penalty` with
  components published under `shaped_components`. Penalties discourage `noop`, early resolves, or
  missing assignment targets.
- **State** - `state()` serializes the episode, enabling fast checkpointing for evaluators.

## Tasks & Difficulty
| Task ID | Difficulty | Scenario | Key Skills |
| --- | --- | --- | --- |
| `easy_password_reset` | Easy | Recover SSO-blocked access while enforcing MFA and identity verification | Empathy, identity verification, clear timeline |
| `medium_seat_billing` | Medium | Explain a $1,740 seat credit and route it for approval | Financial math, queue assignment, preventive guidance |
| `hard_incident_coordination` | Hard | Coordinate SEV-2 incident INC-4471 across regions | Incident command, broadcast cadence, multi-team plan |

Graders live in `support_triage_env/tasks.py` and evaluate keyword coverage, required tags, and
negative phrases. Each rubric weight sums to 1.0 so the total task score always falls in `[0, 1]`.

## Baseline Policy
`inference.py` runs a lightweight baseline that:
1. Instantiates `OpsSupportEnv` for each task
2. Crafts a primary reply using the OpenAI client (falling back to deterministic templates when the
   library/credentials are missing)
3. Issues follow-up assignment or resolve actions for medium/hard tasks

### Required environment variables
Set the following before running the real LLM baseline:
```
export API_BASE_URL="https://api.openai.com/v1"      # or router endpoint
export MODEL_NAME="gpt-4.1-mini"
export OPENAI_API_KEY="sk-..."
# or provide HF_TOKEN / API_KEY as described in the brief
```

### Run baseline locally
```
pip install -r requirements.txt
python3 inference.py                # Runs all three tasks
python3 inference.py --task easy_password_reset
```
Sample output (fallback heuristics):
```
{
  "scores": [
    {"task_id": "easy_password_reset", "score": 0.838},
    {"task_id": "medium_seat_billing", "score": 1.0},
    {"task_id": "hard_incident_coordination", "score": 0.883}
  ],
  "average": 0.907
}
```

## Docker & Hugging Face Space
1. Build and test locally:
   ```
   docker build -t ops-support-triage .
   docker run -p 7860:7860 ops-support-triage
   curl http://localhost:7860/reset              # obtain session + observation
   ```
2. Push to a Hugging Face Space with tag `openenv`. Spaces will run `python app.py`, which exposes
   `/reset`, `/step`, `/state/{session_id}`, and `/tasks`.

## Validation Checklist
- `openenv.yaml` describes metadata, typed models, and tasks.
- `support_triage_env.env.OpsSupportEnv` implements `reset`, `step`, and `state` with shaped rewards.
- Three deterministic graders yield scores between 0 and 1.
- `inference.py` reproduces the published baseline scores and honors `API_BASE_URL`, `MODEL_NAME`,
  and `OPENAI_API_KEY` / `HF_TOKEN`.
- Docker image + FastAPI server respond to probes on port 7860.

## License
MIT. See `openenv.yaml` for attribution metadata.
