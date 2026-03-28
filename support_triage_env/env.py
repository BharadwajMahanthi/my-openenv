"""Core OpenEnv environment implementation."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import ValidationError

from .models import Action, EnvironmentState, Observation, Reward, StepOutput
from .tasks import TASK_REGISTRY, TaskConfig, TaskGrader

AVAILABLE_ACTIONS = ["draft_reply", "assign", "resolve", "clarify", "noop"]


class OpsSupportEnv:
    """Simulates ops/support workflows for OpenEnv agents."""

    def __init__(self, task_id: Optional[str] = None):
        self.task_id = task_id
        self.task: Optional[TaskConfig] = None
        self.grader: Optional[TaskGrader] = None
        self.step_count = 0
        self.reply_history: List[str] = []
        self.tags: List[str] = []
        self.cumulative_score = 0.0
        self.done = False
        self.last_action: Optional[str] = None
        self.reset(task_id)

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Reset environment for task_id (random if None)."""
        chosen_id = task_id or self.task_id or next(iter(TASK_REGISTRY))
        if chosen_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{chosen_id}'")
        self.task_id = chosen_id
        self.task = TASK_REGISTRY[chosen_id]
        self.grader = TaskGrader(self.task)
        self.step_count = 0
        self.reply_history = []
        self.tags = []
        self.cumulative_score = 0.0
        self.done = False
        self.last_action = None
        return self._build_observation()

    def _build_observation(self) -> Observation:
        assert self.task is not None
        remaining = max(self.task.max_steps - self.step_count, 0)
        return Observation(
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            scenario=self.task.scenario,
            inbox=self.task.inbox,
            policy_guidance=self.task.policies,
            remaining_steps=remaining,
            available_actions=AVAILABLE_ACTIONS,
            last_action=self.last_action,
            partial_score=round(self.cumulative_score, 3),
        )

    def step(self, action_dict: Dict | Action) -> StepOutput:
        """Apply an agent action."""
        if self.done:
            raise RuntimeError("Episode already finished; call reset().")
        try:
            action = action_dict if isinstance(action_dict, Action) else Action(**action_dict)
        except ValidationError as exc:
            raise ValueError(f"Invalid action payload: {exc}") from exc

        self.step_count += 1
        penalty = 0.0
        if action.action_type in {"draft_reply", "resolve"} and action.reply:
            self.reply_history.append(action.reply)
        if action.tags:
            self.tags.extend(action.tags)
        if action.action_type == "noop":
            penalty = 0.05
        elif action.action_type == "assign" and not action.assign_to:
            penalty = 0.1
        elif action.action_type == "resolve" and self.cumulative_score < 0.6:
            penalty = 0.2

        self.last_action = action.action_type

        assert self.grader is not None
        prev_score = self.cumulative_score
        score, breakdown = self.grader.score(self.reply_history, self.tags)
        delta = score - prev_score
        shaped = delta - penalty
        self.cumulative_score = score
        reward = Reward(
            value=round(shaped, 4),
            shaped_components={"delta": round(delta, 3), "penalty": penalty},
        )

        self.done = (
            self.step_count >= (self.task.max_steps if self.task else 0)
            or action.action_type == "resolve"
            or self.cumulative_score >= 0.95
        )

        observation = self._build_observation()
        info = {
            "task_score": round(self.cumulative_score, 3),
            "max_steps": self.task.max_steps if self.task else 0,
            "requirements": breakdown,
        }
        return StepOutput(observation=observation, reward=reward, done=self.done, info=info)

    def state(self) -> EnvironmentState:
        assert self.task is not None
        assert self.grader is not None
        _, breakdown = self.grader.score(self.reply_history, self.tags)
        return EnvironmentState(
            task_id=self.task.task_id,
            step_count=self.step_count,
            max_steps=self.task.max_steps,
            cumulative_score=round(self.cumulative_score, 3),
            done=self.done,
            latest_feedback=self.last_action,
            requirements=breakdown,
        )

    def available_tasks(self) -> List[str]:
        return list(TASK_REGISTRY.keys())
