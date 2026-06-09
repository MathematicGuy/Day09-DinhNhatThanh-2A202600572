# Multi-Agent Observability and Recommendation Spec

## Goal

Build a shopping assistant that follows the Day 09 `Guide.md` architecture and proves that role separation makes the system easier to inspect, debug, and improve.

## Milestone 1: Observability

The system must keep the LangGraph flow synchronous:

User -> Supervisor -> Policy worker and/or Data worker -> Response worker -> Final answer

The trace must be append-only. Each node appends events to `trace`; no node deletes or rewrites earlier events.

State ownership:

- Supervisor reads `question`, writes `route`, and appends supervisor trace events.
- Policy worker reads only the policy task contract from `route`, writes `policy_result`, and appends policy trace events.
- Data worker reads only the data task contract from `route`, writes `data_result`, and appends data trace events.
- Response worker reads `route`, `policy_result`, and `data_result`, writes `final_answer`, and appends response trace events.

Trace events must capture:

- `run_id`
- `case_id`
- `timestamp`
- `node`
- `event`
- `status`
- `input`
- `output`
- `warnings`
- `error`
- `latency_ms`

The system must write JSON trace files for single-question runs and one trace file per case during batch runs.

## Milestone 2: Improvement Recommendations

The system must generate recommendations from batch summary and trace evidence.

Each recommendation must include:

- `pain_point`
- `evidence`
- `feature`
- `improvement`
- `priority`

Examples:

- If routing mismatches happen, recommend improving supervisor routing criteria or examples.
- If policy retrieval returns weak evidence, recommend improving chunking, query rewriting, or citation display.
- If worker status is `not_found` or `clarification_needed`, recommend clearer user input handling.
- If trace shows repeated failures, recommend fallback or retry limits.

## Non-Goals

This project does not build enterprise observability, distributed tracing, async orchestration, a web dashboard, or human review workflows. Human-in-the-loop is documented as a future extension point after the response worker.
