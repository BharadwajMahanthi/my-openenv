"""Typed models for the Ops Support Triage OpenEnv environment."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class TicketMessage(BaseModel):
    """Single inbound or outbound ticket event."""

    sender: Literal["customer", "internal", "system", "agent"]
    subject: str
    body: str
    timestamp: str
    channel: Literal["email", "chat", "incident"] = "email"


class PolicySnippet(BaseModel):
    """Operational policy guidelines available to the agent."""

    title: str
    content: str


class RequirementProgress(BaseModel):
    """Tracks satisfaction of a single grading requirement."""

    requirement_id: str
    description: str
    weight: float
    met: bool
    coverage: float


class Observation(BaseModel):
    """Observation emitted to the agent."""

    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    scenario: str
    inbox: List[TicketMessage]
    policy_guidance: List[PolicySnippet]
    remaining_steps: int
    available_actions: List[str]
    last_action: Optional[str] = None
    partial_score: float = 0.0


class Action(BaseModel):
    """Action provided by the agent."""

    action_type: Literal[
        "draft_reply",
        "assign",
        "resolve",
        "clarify",
        "noop",
    ] = "draft_reply"
    reply: Optional[str] = None
    assign_to: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high"]] = None
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @validator("reply")
    def reply_required_for_draft(cls, value, values):  # type: ignore[override]
        if values.get("action_type") in {"draft_reply", "resolve"} and not value:
            raise ValueError("reply text required for draft_reply/resolve actions")
        return value


class Reward(BaseModel):
    """Reward signal with shaped components."""

    value: float
    shaped_components: Dict[str, float]


class EnvironmentState(BaseModel):
    """Serializable environment state."""

    task_id: str
    step_count: int
    max_steps: int
    cumulative_score: float
    done: bool
    latest_feedback: Optional[str] = None
    requirements: List[RequirementProgress] = Field(default_factory=list)


class StepOutput(BaseModel):
    """Full payload returned by step()."""

    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, float | str | List[RequirementProgress]]
