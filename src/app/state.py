from __future__ import annotations

from typing import Annotated, Any, Literal
import operator
from typing_extensions import TypedDict


Status = Literal["ok", "clarification_needed", "not_found", "error"]


class WorkerTask(TypedDict, total=False):
    task: str
    context: str
    expected_output: str


class RouteDecision(TypedDict, total=False):
    status: Status
    needs_policy: bool
    needs_data: bool
    reason: str
    clarification_question: str | None
    policy_task: WorkerTask | None
    data_task: WorkerTask | None


class WorkerResult(TypedDict, total=False):
    status: Status
    summary: str
    facts: list[str]
    citations: list[str]
    tool_calls: list[dict[str, Any]]
    warnings: list[str]
    error: str | None


class TraceEvent(TypedDict, total=False):
    run_id: str
    case_id: str | None
    timestamp: str
    node: str
    event: str
    status: Status
    input: dict[str, Any]
    output: dict[str, Any]
    warnings: list[str]
    error: str | None
    latency_ms: int


class Recommendation(TypedDict):
    pain_point: str
    evidence: str
    feature: str
    improvement: str
    priority: Literal["high", "medium", "low"]


class ShoppingState(TypedDict, total=False):
    run_id: str
    case_id: str | None
    question: str
    route: RouteDecision
    policy_result: WorkerResult
    data_result: WorkerResult
    final_answer: str
    trace: Annotated[list[TraceEvent], operator.add]
