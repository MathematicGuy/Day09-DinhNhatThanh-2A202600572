# Observability and Recommendation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Day 09 multi-agent shopping assistant from `Guide.md`, with Milestone 1 focused on observable execution traces and Milestone 2 focused on trace-driven improvement recommendations.

**Architecture:** Keep the required synchronous LangGraph flow: Supervisor -> Policy worker and/or Data worker -> Response worker. Each node owns only its own state field, every node appends structured trace events, and batch execution writes per-case traces plus a summary used by the recommendation layer.

**Tech Stack:** Python, LangGraph, LangChain tools, Chroma, `sentence-transformers/all-MiniLM-L6-v2`, provider modules in `src/provider/`, pytest.

---

## File Structure

- Modify `SPECS.md`: rewrite the rough notes into clear Milestone 1 and Milestone 2 acceptance criteria.
- Modify `src/app/state.py`: define state ownership, route contract, worker result shapes, trace event shape, and recommendation shape.
- Create `src/app/tracing.py`: central trace event builders and JSON-safe serialization.
- Modify `src/app/data_access.py`: implement mock-data indexes and four small lookup tools.
- Modify `src/rag/parser.py`: parse policy markdown into H2/H3 chunks with citations.
- Modify `src/rag/vector_store.py`: implement Chroma indexing and search.
- Modify `src/app/prompts.py`: replace prompt notes with strict JSON contracts for supervisor and workers.
- Modify `src/app/graph.py`: implement synchronous graph nodes, A2A message contracts, trace capture, single-question execution, batch execution, and recommendation generation.
- Modify `src/app/cli.py`: support single-question, trace-file, batch, and recommendation output.
- Create tests under `tests/`: unit tests for data lookup, parser, trace schema, supervisor routing helpers, batch summary, and recommendations.
- Optional: create `pytest.ini` to register the `external` mark already used by `tests/test_model.py`.

## Milestone 1: Observability

Milestone 1 is complete when the lab runs end-to-end and every run can answer:

- Which agent ran, in what order.
- What input each node received.
- What output each node wrote.
- Which route the supervisor chose and why.
- Which tool calls were made and what status they returned.
- Which warnings, fallbacks, retries, or errors happened.

The trace is a list. Nodes append events and never overwrite existing trace data.

## Milestone 2: Improvement Recommendations

Milestone 2 is complete when batch execution produces a recommendation report that maps:

- Business analyst pain point.
- Observed evidence from trace or summary.
- Suggested feature or engineering improvement.
- Expected system improvement.

The recommendation layer must use trace and batch summary data. It must not rely on hidden state or manual interpretation.

---

## Task 1: Clarify `SPECS.md`

**Files:**
- Modify: `SPECS.md`

- [ ] **Step 1: Replace the rough spec with clear milestone language**

Use this structure:

```markdown
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
```

- [ ] **Step 2: Review the spec for ambiguity**

Run:

```powershell
Select-String -Path SPECS.md -Pattern 'PLACEHOLDER|\\[|\\]'
```

Expected: no placeholder matches.

- [ ] **Step 3: Commit**

```powershell
git add SPECS.md
git commit -m "docs: clarify observability milestones"
```

---

## Task 2: Define State and Trace Contracts

**Files:**
- Modify: `src/app/state.py`
- Create: `src/app/tracing.py`
- Test: `tests/test_tracing.py`

- [ ] **Step 1: Write trace contract tests**

Create `tests/test_tracing.py`:

```python
from app.tracing import make_trace_event


def test_make_trace_event_has_required_fields():
    event = make_trace_event(
        run_id="run-1",
        case_id="Q01",
        node="supervisor",
        event="route_decided",
        status="ok",
        input_payload={"question": "Chinh sach hoan tra?"},
        output_payload={"needs_policy": True},
        warnings=[],
        error=None,
        latency_ms=12,
    )

    assert event["run_id"] == "run-1"
    assert event["case_id"] == "Q01"
    assert event["node"] == "supervisor"
    assert event["event"] == "route_decided"
    assert event["status"] == "ok"
    assert event["input"]["question"] == "Chinh sach hoan tra?"
    assert event["output"]["needs_policy"] is True
    assert event["warnings"] == []
    assert event["error"] is None
    assert event["latency_ms"] == 12
    assert "timestamp" in event
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_tracing.py -q
```

Expected: fail because `app.tracing` does not exist.

- [ ] **Step 3: Implement typed state and trace helpers**

In `src/app/state.py`, define these `TypedDict` contracts:

```python
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
```

Create `src/app/tracing.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def make_trace_event(
    *,
    run_id: str,
    case_id: str | None,
    node: str,
    event: str,
    status: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
    warnings: list[str] | None = None,
    error: str | None = None,
    latency_ms: int = 0,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "case_id": case_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": node,
        "event": event,
        "status": status,
        "input": input_payload,
        "output": output_payload,
        "warnings": warnings or [],
        "error": error,
        "latency_ms": latency_ms,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
```

- [ ] **Step 4: Run test**

Run:

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_tracing.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add src/app/state.py src/app/tracing.py tests/test_tracing.py
git commit -m "feat: define observable state contracts"
```

---

## Task 3: Implement Mock Data Lookup Tools

**Files:**
- Modify: `src/app/data_access.py`
- Test: `tests/test_data_access.py`

- [ ] **Step 1: Write data lookup tests**

Create `tests/test_data_access.py`:

```python
from pathlib import Path

from app.data_access import ShoppingDataStore


DATA_PATH = Path("data/order_customer_mock_data.json")


def test_get_customer_by_id_ok():
    store = ShoppingDataStore(DATA_PATH)
    result = store.get_customer_by_id("C001")
    assert result["status"] == "ok"
    assert result["customer"]["customer_id"] == "C001"


def test_get_order_detail_by_order_id_not_found():
    store = ShoppingDataStore(DATA_PATH)
    result = store.get_order_detail_by_order_id("9999")
    assert result["status"] == "not_found"
    assert result["order_id"] == "9999"


def test_get_vouchers_by_customer_id_only_active():
    store = ShoppingDataStore(DATA_PATH)
    result = store.get_vouchers_by_customer_id("C001", only_active=True)
    assert result["status"] == "ok"
    assert all(voucher["status"] == "active" for voucher in result["vouchers"])
```

- [ ] **Step 2: Run failing tests**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_data_access.py -q
```

Expected: fail because the methods raise `NotImplementedError`.

- [ ] **Step 3: Implement data store and four tools**

Implementation requirements:

- Load JSON once in `ShoppingDataStore.__init__`.
- Build `customer_by_id`, `order_by_id`, `orders_by_customer_id`, and `vouchers_by_customer_id`.
- Return `{"status": "ok", ...}` or `{"status": "not_found", ...}` from every method.
- `build_data_tools()` must expose four LangChain tools with clear names:
  `get_customer_by_id`, `get_orders_by_customer_id`, `get_order_detail_by_order_id`, `get_vouchers_by_customer_id`.

- [ ] **Step 4: Run tests**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_data_access.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add src/app/data_access.py tests/test_data_access.py
git commit -m "feat: add shopping data lookup tools"
```

---

## Task 4: Implement Policy Parsing and Chroma Search

**Files:**
- Modify: `src/rag/parser.py`
- Modify: `src/rag/vector_store.py`
- Test: `tests/test_policy_rag.py`

- [ ] **Step 1: Write parser tests**

Create `tests/test_policy_rag.py`:

```python
from pathlib import Path

from rag.parser import parse_policy_markdown


def test_parse_policy_markdown_returns_h2_h3_chunks():
    markdown = Path("data/policy_mock_vi.md").read_text(encoding="utf-8")
    chunks = parse_policy_markdown(markdown)

    assert chunks
    assert all("section_h2" in chunk for chunk in chunks)
    assert all("section_h3" in chunk for chunk in chunks)
    assert all("citation" in chunk for chunk in chunks)
    assert all("rendered_text" in chunk for chunk in chunks)
    assert any("trả hàng" in chunk["rendered_text"].lower() for chunk in chunks)
```

- [ ] **Step 2: Run failing parser test**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_policy_rag.py::test_parse_policy_markdown_returns_h2_h3_chunks -q
```

Expected: fail because parser raises `NotImplementedError`.

- [ ] **Step 3: Implement parser**

Parser rules:

- A chunk starts at `###`.
- The active `##` is stored as `section_h2`.
- The active `###` is stored as `section_h3`.
- Content lines until the next `##` or `###` belong to the active chunk.
- `citation` is `"{section_h2} > {section_h3}"`.
- `rendered_text` is `"{section_h2}\n{section_h3}\n{content}"`.

- [ ] **Step 4: Run parser test**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_policy_rag.py::test_parse_policy_markdown_returns_h2_h3_chunks -q
```

Expected: pass.

- [ ] **Step 5: Implement Chroma store**

Implementation requirements:

- Initialize `chromadb.PersistentClient(path=str(persist_directory))`.
- Use `get_or_create_collection(collection_name)`.
- `ensure_index()` rebuilds only when collection count is zero.
- `rebuild()` deletes and recreates the collection, parses markdown, embeds `rendered_text`, and adds ids, documents, metadatas, and embeddings.
- `search()` embeds the query and returns hits with `citation`, `content`, `distance`, and section metadata.

- [ ] **Step 6: Add a lightweight fake embedding test for vector-store wiring**

Append to `tests/test_policy_rag.py`:

```python
from rag.vector_store import ChromaPolicyStore


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def test_chroma_policy_store_search_returns_citations(tmp_path):
    store = ChromaPolicyStore(tmp_path, FakeEmbeddings(), collection_name="test_policy")
    store.rebuild(Path("data/policy_mock_vi.md"))

    hits = store.search("hoan tra", top_k=2)

    assert len(hits) == 2
    assert "citation" in hits[0]
    assert "content" in hits[0]
    assert "distance" in hits[0]
```

- [ ] **Step 7: Run RAG tests**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_policy_rag.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```powershell
git add src/rag/parser.py src/rag/vector_store.py tests/test_policy_rag.py
git commit -m "feat: add policy parsing and chroma search"
```

---

## Task 5: Implement Supervisor Routing and A2A Contracts

**Files:**
- Modify: `src/app/prompts.py`
- Modify: `src/app/graph.py`
- Test: `tests/test_routing.py`

- [ ] **Step 1: Write routing tests**

Create `tests/test_routing.py`:

```python
from app.graph import route_question


def test_route_policy_question():
    route = route_question("Chính sách hoàn trả hàng ra sao?")
    assert route["status"] == "ok"
    assert route["needs_policy"] is True
    assert route["needs_data"] is False


def test_route_data_question():
    route = route_question("Đơn hàng 1971 bao giờ được giao?")
    assert route["status"] == "ok"
    assert route["needs_policy"] is False
    assert route["needs_data"] is True


def test_route_mixed_question():
    route = route_question("Đơn hàng 2058 còn trong thời gian trả hàng không?")	
    assert route["status"] == "ok"
    assert route["needs_policy"] is True
    assert route["needs_data"] is True


def test_route_clarification_needed():
    route = route_question("Voucher của tôi còn dùng được không?")
    assert route["status"] == "clarification_needed"
    assert route["needs_policy"] is False
    assert route["needs_data"] is False
    assert route["clarification_question"]
```

- [ ] **Step 2: Run failing routing tests**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_routing.py -q
```

Expected: fail because `route_question` does not exist.

- [ ] **Step 3: Implement deterministic routing helper**

Add `route_question(question: str) -> dict[str, Any]` in `src/app/graph.py`.

Routing requirements:

- Extract order ids with regex for `đơn hàng` followed by digits, and customer ids with regex `C\d{3}`.
- Policy-only if the question contains policy concepts such as `chính sách`, `hoàn trả`, `trả hàng`, `giao hàng`, `kiểm hàng`, or `voucher` and no specific order/customer lookup is required.
- Data-only if it asks about specific order/customer/voucher facts and contains an order id or customer id.
- Mixed if it asks whether a specific order can return, refund, refuse delivery, or relates an order to policy.
- Clarification if it asks about `tôi`, `đơn hàng của tôi`, or `voucher của tôi` without an order id or customer id.

Each `ok` route must include:

```python
{
    "status": "ok",
    "needs_policy": bool,
    "needs_data": bool,
    "reason": "...",
    "clarification_question": None,
    "policy_task": {"task": "...", "context": "...", "expected_output": "..."} | None,
    "data_task": {"task": "...", "context": "...", "expected_output": "..."} | None,
}
```

- [ ] **Step 4: Replace prompts with strict contracts**

`src/app/prompts.py` should describe exact JSON shapes for supervisor, policy worker, data worker, and response worker. The prompts must say workers should use only the task contract they receive.

- [ ] **Step 5: Run routing tests**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_routing.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add src/app/graph.py src/app/prompts.py tests/test_routing.py
git commit -m "feat: add supervisor routing contracts"
```

---

## Task 6: Implement Graph Nodes and Single-Question Execution

**Files:**
- Modify: `src/app/graph.py`
- Test: `tests/test_graph_execution.py`

- [ ] **Step 1: Write graph execution tests with fakes**

Create `tests/test_graph_execution.py`:

```python
from pathlib import Path

from app.graph import ShoppingAssistant
from app.config import Settings


def test_ask_clarification_writes_trace(tmp_path):
    settings = Settings.load()
    settings.traces_dir = tmp_path
    assistant = ShoppingAssistant(settings=settings)

    result = assistant.ask("Voucher của tôi còn dùng được không?", trace_file=tmp_path / "trace.json")

    assert result["route"]["status"] == "clarification_needed"
    assert "Status: clarification_needed" in result["final_answer"]
    assert result["trace"]
    assert (tmp_path / "trace.json").exists()


def test_ask_not_found_order_writes_data_result(tmp_path):
    settings = Settings.load()
    settings.traces_dir = tmp_path
    assistant = ShoppingAssistant(settings=settings)

    result = assistant.ask("Kiểm tra đơn hàng 9999 giúp tôi")

    assert result["route"]["needs_data"] is True
    assert result["data_result"]["status"] == "not_found"
    assert "Status: not_found" in result["final_answer"]
```

- [ ] **Step 2: Run failing tests**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_graph_execution.py -q
```

Expected: fail because `ShoppingAssistant.ask()` raises `NotImplementedError`.

- [ ] **Step 3: Implement `ShoppingAssistant.__init__`**

Initialize:

- `ShoppingDataStore(settings.orders_path)`.
- `SentenceTransformerEmbeddings(settings.embedding_model_name)`.
- `ChromaPolicyStore(settings.chroma_dir, embeddings)`.
- LLM provider from `src/provider/` only if response synthesis needs LLM.
- `self.graph = build_graph(self)`.

- [ ] **Step 4: Implement graph node methods**

Use synchronous node functions.

Node ownership:

- `supervisor_node` returns only `route` and `trace`.
- `worker_1_policy_node` returns only `policy_result` and `trace`.
- `worker_2_data_node` returns only `data_result` and `trace`.
- `worker_3_response_node` returns only `final_answer` and `trace`.

For Milestone 1, deterministic worker behavior is acceptable when it is enough to pass `data/test.json`. Use the LLM provider for wording only after data and policy facts are already structured.

- [ ] **Step 5: Implement graceful failures**

Failure behavior:

- Worker exceptions become `{"status": "error", "error": "..."}` in that worker result.
- Response worker returns successful partial evidence if one worker succeeds and the other fails.
- Trace event includes `status: "error"` and the error string.
- No node raises uncaught errors for normal `not_found` or `clarification_needed` cases.

- [ ] **Step 6: Implement `ask()`**

`ask()` must:

- Create a `run_id`.
- Initialize state with `run_id`, `case_id`, `question`, and empty trace.
- Optionally rebuild the Chroma index.
- Invoke graph.
- Write trace JSON when `trace_file` is passed.
- Return `route`, `policy_result`, `data_result`, `final_answer`, and `trace`.

- [ ] **Step 7: Run execution tests**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_graph_execution.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```powershell
git add src/app/graph.py tests/test_graph_execution.py
git commit -m "feat: execute observable shopping graph"
```

---

## Task 7: Implement CLI, Batch Summary, and Trace Files

**Files:**
- Modify: `src/app/cli.py`
- Modify: `src/app/graph.py`
- Test: `tests/test_batch.py`

- [ ] **Step 1: Write batch tests**

Create `tests/test_batch.py`:

```python
import json
from pathlib import Path

from app.config import Settings
from app.graph import ShoppingAssistant


def test_run_batch_writes_summary_and_traces(tmp_path):
    settings = Settings.load()
    settings.traces_dir = tmp_path / "traces"
    assistant = ShoppingAssistant(settings=settings)

    summary = assistant.run_batch(Path("data/test.json"), tmp_path)

    assert summary["total"] == 22
    assert "route_accuracy" in summary
    assert "status_accuracy" in summary
    assert (tmp_path / "summary.json").exists()
    assert len(list((tmp_path / "traces").glob("*.json"))) == 22

    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["total"] == 22
```

- [ ] **Step 2: Run failing test**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_batch.py -q
```

Expected: fail because `run_batch()` raises `NotImplementedError`.

- [ ] **Step 3: Implement `run_batch()`**

`run_batch()` must:

- Read `data/test.json`.
- For each case, call `ask(question, trace_file=output_dir / "traces" / "{id}.json")`.
- Compare `route` to `expected_route`.
- Compare final status to `expected_status`.
- Check `expected_contains` strings when present.
- Write `summary.json`.

Summary shape:

```json
{
  "total": 22,
  "route_accuracy": 0.0,
  "status_accuracy": 0.0,
  "contains_accuracy": 0.0,
  "cases": [
    {
      "id": "Q01",
      "question": "...",
      "expected_route": ["policy"],
      "actual_route": ["policy"],
      "expected_status": "ok",
      "actual_status": "ok",
      "route_ok": true,
      "status_ok": true,
      "contains_ok": true,
      "trace_file": "traces/Q01.json"
    }
  ]
}
```

- [ ] **Step 4: Implement CLI**

CLI behavior:

- `python -m app.cli --question "..."` prints the final answer.
- `python -m app.cli --question "..." --trace-file trace.json` writes trace JSON.
- `python -m app.cli --batch --test-file data/test.json` writes output to `src/artifacts/traces` by default and prints summary metrics.

- [ ] **Step 5: Run batch tests**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_batch.py -q
```

Expected: pass.

- [ ] **Step 6: Run CLI smoke checks**

```powershell
$env:PYTHONPATH='src'; python -m app.cli --question "Voucher của tôi còn dùng được không?"
$env:PYTHONPATH='src'; python -m app.cli --batch --test-file data/test.json
```

Expected:

- First command prints `Status: clarification_needed`.
- Second command writes `summary.json` and per-case traces.

- [ ] **Step 7: Commit**

```powershell
git add src/app/cli.py src/app/graph.py tests/test_batch.py
git commit -m "feat: add batch traces and summary"
```

---

## Task 8: Add Milestone 2 Recommendations

**Files:**
- Modify: `src/app/graph.py`
- Modify: `src/app/cli.py`
- Test: `tests/test_recommendations.py`

- [ ] **Step 1: Write recommendation tests**

Create `tests/test_recommendations.py`:

```python
from app.graph import recommend_improvements


def test_recommend_routing_improvement_from_summary():
    summary = {
        "route_accuracy": 0.8,
        "status_accuracy": 1.0,
        "contains_accuracy": 1.0,
        "cases": [
            {
                "id": "Q11",
                "route_ok": False,
                "status_ok": True,
                "contains_ok": True,
                "expected_route": ["data", "policy"],
                "actual_route": ["data"],
            }
        ],
    }

    recommendations = recommend_improvements(summary)

    assert recommendations
    assert recommendations[0]["priority"] == "high"
    assert "routing" in recommendations[0]["feature"].lower()


def test_recommend_answer_quality_from_contains_failures():
    summary = {
        "route_accuracy": 1.0,
        "status_accuracy": 1.0,
        "contains_accuracy": 0.5,
        "cases": [
            {
                "id": "Q01",
                "route_ok": True,
                "status_ok": True,
                "contains_ok": False,
                "expected_contains": ["15 ngày", "trả hàng"],
            }
        ],
    }

    recommendations = recommend_improvements(summary)

    assert any("evidence" in item["feature"].lower() or "citation" in item["feature"].lower() for item in recommendations)
```

- [ ] **Step 2: Run failing tests**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_recommendations.py -q
```

Expected: fail because `recommend_improvements` does not exist.

- [ ] **Step 3: Implement `recommend_improvements(summary)`**

Rules:

- If `route_accuracy < 0.95`, add high-priority recommendation for supervisor routing examples and route criteria.
- If `status_accuracy < 0.95`, add high-priority recommendation for `clarification_needed`, `not_found`, or graceful failure handling.
- If `contains_accuracy < 0.95`, add medium-priority recommendation for response evidence quality and citation use.
- If cases include `actual_status == "error"`, add high-priority recommendation for fallback and retry handling.
- If no issues are found, return one low-priority recommendation for cost and latency instrumentation.

Each recommendation has:

```python
{
    "pain_point": "...",
    "evidence": "...",
    "feature": "...",
    "improvement": "...",
    "priority": "high" | "medium" | "low",
}
```

- [ ] **Step 4: Add CLI recommendation flag**

Add `--recommend` to `src/app/cli.py`.

Behavior:

- `python -m app.cli --batch --test-file data/test.json --recommend` writes `recommendations.json`.
- The terminal prints recommendation count and top priority.

- [ ] **Step 5: Run recommendation tests**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_recommendations.py -q
```

Expected: pass.

- [ ] **Step 6: Run batch with recommendations**

```powershell
$env:PYTHONPATH='src'; python -m app.cli --batch --test-file data/test.json --recommend
```

Expected: `summary.json`, per-case traces, and `recommendations.json` are written.

- [ ] **Step 7: Commit**

```powershell
git add src/app/graph.py src/app/cli.py tests/test_recommendations.py
git commit -m "feat: recommend improvements from traces"
```

---

## Task 9: Final Verification and Cleanup

**Files:**
- Optional: `pytest.ini`
- Verify all implementation files.

- [ ] **Step 1: Register pytest mark if warning still appears**

Create `pytest.ini` only if `PytestUnknownMarkWarning` appears:

```ini
[pytest]
markers =
    external: live external-provider smoke tests
```

- [ ] **Step 2: Run syntax check**

```powershell
python -m py_compile src/app/*.py src/provider/*.py src/rag/*.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run local tests**

```powershell
$env:PYTHONPATH='src'; python -m pytest tests -q
```

Expected: all local deterministic tests pass. External smoke tests may skip unless `RUN_EXTERNAL_PROVIDER_SMOKE=1`.

- [ ] **Step 4: Run final lab smoke commands**

```powershell
$env:PYTHONPATH='src'; python -m app.cli --question "Đơn hàng 1971 có được hoàn trả không?" --trace-file src/artifacts/traces/manual-Q11.json
$env:PYTHONPATH='src'; python -m app.cli --batch --test-file data/test.json --recommend
```

Expected:

- Manual command returns an answer with policy and order evidence.
- Batch command writes `summary.json`, per-case trace files, and `recommendations.json`.

- [ ] **Step 5: Review generated artifacts**

Open one trace file and confirm it contains ordered events from:

- `supervisor`
- `worker_1_policy` when policy is needed
- `worker_2_data` when data is needed
- `worker_3_response`

- [ ] **Step 6: Commit final cleanup**

```powershell
git add .
git commit -m "test: verify observable multi-agent assistant"
```

---

## Acceptance Criteria

- `SPECS.md` clearly separates Milestone 1 observability from Milestone 2 recommendations.
- The assistant follows the required synchronous Guide.md flow.
- State ownership is explicit and enforced by node return values.
- Worker context follows the A2A need-to-know contract.
- Trace is append-only and JSON serializable.
- Single-question runs can write a trace file.
- Batch runs write one trace per case plus `summary.json`.
- Recommendations are generated from batch summary and trace-visible outcomes.
- `clarification_needed`, `not_found`, and worker errors are graceful user-facing states.
- Tests cover data lookup, policy parsing, routing, trace shape, graph execution, batch summary, and recommendations.
