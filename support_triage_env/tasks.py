"""Task registry and graders for the Ops Support Triage environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from pydantic import BaseModel, Field

from .models import PolicySnippet, RequirementProgress, TicketMessage


class TaskRequirement(BaseModel):
    """Structured grader rule."""

    requirement_id: str
    description: str
    keywords: List[str]
    weight: float
    must_not: List[str] = Field(default_factory=list)
    required_tags: List[str] = Field(default_factory=list)

    def coverage(self, text: str, tags: List[str]) -> float:
        if not self.keywords:
            return 0.0
        lower_text = text.lower()
        keyword_hits = sum(1 for kw in self.keywords if kw.lower() in lower_text)
        coverage = keyword_hits / len(self.keywords)
        if self.must_not:
            if any(block.lower() in lower_text for block in self.must_not):
                return 0.0
        if self.required_tags:
            if not set(tag.lower() for tag in self.required_tags).issubset(
                {tag.lower() for tag in tags}
            ):
                coverage *= 0.5
        return coverage


@dataclass
class TaskConfig:
    """Configuration for a single triage task."""

    task_id: str
    difficulty: str
    scenario: str
    inbox: List[TicketMessage]
    policies: List[PolicySnippet]
    max_steps: int
    requirements: List[TaskRequirement]


MIN_SCORE = 1e-3
MAX_SCORE = 1.0 - MIN_SCORE


class TaskGrader:
    """Deterministic grader computing progress and rewards."""

    def __init__(self, config: TaskConfig):
        self.config = config

    def score(self, reply_history: List[str], tags: List[str]) -> tuple[float, List[RequirementProgress]]:
        combined = "\n".join(reply_history).lower()
        breakdown: List[RequirementProgress] = []
        total = 0.0
        for req in self.config.requirements:
            cov = req.coverage(combined, tags)
            met = cov >= 0.8
            weighted = cov * req.weight
            total += weighted
            breakdown.append(
                RequirementProgress(
                    requirement_id=req.requirement_id,
                    description=req.description,
                    weight=req.weight,
                    met=met,
                    coverage=round(cov, 3),
                )
            )
        clamped = max(MIN_SCORE, min(total, MAX_SCORE))
        return clamped, breakdown


TASK_REGISTRY: Dict[str, TaskConfig] = {}


def register(task: TaskConfig) -> None:
    TASK_REGISTRY[task.task_id] = task


register(
    TaskConfig(
        task_id="easy_password_reset",
        difficulty="easy",
        scenario=(
            "Handle a healthcare admin who cannot log into the analytics dashboard after SSO "
            "settings changed. Provide a compliant, human-ready response and next steps."
        ),
        inbox=[
            TicketMessage(
                sender="customer",
                subject="Urgent: Cant sign in after Okta change",
                body=(
                    "We flipped Parnassus Clinics over to Okta yesterday. None of our charge nurses "
                    "can get back into the Insight dashboard, even though Okta shows the app "
                    "assigned. This blocks us from uploading billing. Need someone to fix ASAP."
                ),
                timestamp="2026-03-27T13:04:00Z",
                channel="email",
            )
        ],
        policies=[
            PolicySnippet(
                title="Identity re-verification",
                content=(
                    "When login failures follow an SSO cutover, require callers to confirm the last "
                    "invoice amount or ticket PIN before resetting MFA."
                ),
            ),
            PolicySnippet(
                title="Dashboard recovery steps",
                content=(
                    "Reset via Admin Portal > Directory > Force Password Reset, then send the new "
                    "magic link with 15-minute expiry. Mention that Insight users must re-enroll "
                    "MFA on first launch."
                ),
            ),
        ],
        max_steps=3,
        requirements=[
            TaskRequirement(
                requirement_id="ack",
                description="Empathetic acknowledgement of the outage impact",
                keywords=["sorry", "understand", "appreciate", "frustration"],
                weight=0.15,
            ),
            TaskRequirement(
                requirement_id="verify",
                description="Requests required identity verification signal",
                keywords=["verify", "last invoice", "ticket pin"],
                weight=0.2,
            ),
            TaskRequirement(
                requirement_id="steps",
                description="Shares the admin portal reset + MFA reenrollment steps",
                keywords=["admin portal", "force password reset", "magic link", "mfa"],
                weight=0.45,
            ),
            TaskRequirement(
                requirement_id="sla",
                description="Sets expectation for follow-up timing",
                keywords=["15", "minute", "update", "eta"],
                weight=0.2,
            ),
        ],
    )
)

register(
    TaskConfig(
        task_id="medium_seat_billing",
        difficulty="medium",
        scenario=(
            "Investigate a billing dispute after an HRIS sync failure inflated the active seat "
            "count. Calculate the precise credit, assign to the billing pod, and educate the "
            "customer on preventing repeat incidents."
        ),
        inbox=[
            TicketMessage(
                sender="customer",
                subject="Invoice 118771 charging 18 seats",
                body=(
                    "Larkin Health is showing 18 billed seats for February even though we offboarded "
                    "six care coordinators on Jan 5. Finance is upset that we paid $5,220 instead of "
                    "$3,480. Need proof this is fixed before month-close."
                ),
                timestamp="2026-03-26T18:22:00Z",
            ),
            TicketMessage(
                sender="system",
                subject="Seat audit",
                body="Usage export: 12 active seats after 2026-01-05, 18 billed seats until 2026-02-01",
                timestamp="2026-03-26T18:30:00Z",
            ),
        ],
        policies=[
            PolicySnippet(
                title="Pricing schedule",
                content=(
                    "$290 monthly per seat. Overages create refundable credits on the next invoice."
                ),
            ),
            PolicySnippet(
                title="Assignment rule",
                content="Send all >$1k adjustments to queue BILLING-RANGER with priority medium.",
            ),
        ],
        max_steps=4,
        requirements=[
            TaskRequirement(
                requirement_id="calc",
                description="Explains the math that leads to the $1,740 credit",
                keywords=["6", "seat", "$1,740", "$290"],
                weight=0.4,
            ),
            TaskRequirement(
                requirement_id="assignment",
                description="Assigns case to BILLING-RANGER with medium priority",
                keywords=["billing-ranger", "priority", "medium"],
                weight=0.2,
                required_tags=["billing"],
            ),
            TaskRequirement(
                requirement_id="prevention",
                description="Gives a preventive action referencing HRIS auto-sync",
                keywords=["okta", "auto-sync", "audit"],
                weight=0.2,
            ),
            TaskRequirement(
                requirement_id="timeline",
                description="Commits to credit appearing on the March 31 invoice",
                keywords=["march 31", "invoice", "credit"],
                weight=0.2,
            ),
        ],
    )
)

register(
    TaskConfig(
        task_id="hard_incident_coordination",
        difficulty="hard",
        scenario=(
            "Coordinate an SEV-2 incident (INC-4471) impacting remote chart uploads for three "
            "clinic groups. Build a cross-functional action plan, schedule broadcast updates, and "
            "reassure the requester."
        ),
        inbox=[
            TicketMessage(
                sender="customer",
                subject="INC-4471 uploads failing across regions",
                body=(
                    "Uploaders in Denver, Austin, and Fresno all get 504 errors when pushing overnight "
                    "ADT files. Clinics can't close charts and we triggered downtime procedures."
                ),
                timestamp="2026-03-27T05:12:00Z",
            ),
            TicketMessage(
                sender="internal",
                subject="PagerDuty - edge cache",
                body="Edge cache cluster 3 reports elevated 5xx after deploy 2026.12.7.",
                timestamp="2026-03-27T05:14:00Z",
                channel="incident",
            ),
            TicketMessage(
                sender="system",
                subject="SLA reminder",
                body="SEV-2 requires customer updates every 30 minutes until recovery.",
                timestamp="2026-03-27T05:15:00Z",
            ),
        ],
        policies=[
            PolicySnippet(
                title="Incident workflow",
                content=(
                    "Spin up #eng-incident room, assign comms lead, and update status page + health "
                    "RSS at least every 30 minutes."
                ),
            ),
            PolicySnippet(
                title="Broadcast template",
                content=(
                    "Use template: 'We are actively mitigating INC-#### impacting <systems>. Next "
                    "update at <time>.'"
                ),
            ),
        ],
        max_steps=5,
        requirements=[
            TaskRequirement(
                requirement_id="ref",
                description="References incident code INC-4471 and affected regions",
                keywords=["inc-4471", "denver", "austin", "fresno"],
                weight=0.25,
            ),
            TaskRequirement(
                requirement_id="plan",
                description="Outlines multi-step plan with owners (eng, comms, customer)",
                keywords=["#eng-incident", "status page", "comms", "customer"],
                weight=0.35,
            ),
            TaskRequirement(
                requirement_id="cadence",
                description="Commits to 30-minute update cadence",
                keywords=["30", "minute", "update", "cadence"],
                weight=0.2,
            ),
            TaskRequirement(
                requirement_id="broadcast",
                description="Drafts outward-facing message with next update time",
                keywords=["mitigating", "next update", "time"],
                weight=0.2,
            ),
        ],
    )
)
