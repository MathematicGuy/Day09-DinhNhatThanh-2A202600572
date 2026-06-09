# Test Suite Usage Guide

This folder contains the pytest tests for the shopping assistant. The tests are mostly offline and deterministic: they use local mock data, local policy markdown, and small fake objects instead of live LLM calls.

## Quick Start

Run the safe local smoke set from the repository root:

```powershell
& 'D:\CODE\Anaconda\envs\vin\python.exe' -m pytest `
  tests\test_routing.py `
  tests\test_graph_execution.py `
  tests\test_streamlit_chat.py `
  tests\test_tracing.py `
  tests\test_batch.py `
  tests\test_data_access.py `
  tests\test_recommendations.py `
  -q --basetemp .pytest_tmp
```

Use `--basetemp .pytest_tmp` on this Windows workspace because the default system temp directory can be blocked by sandbox permissions.

`test_policy_rag.py` imports ChromaDB. In some Windows/Conda environments, the native NumPy/Chroma stack can crash the Python process instead of failing cleanly. Run it separately when you are specifically working on policy RAG:

```powershell
& 'D:\CODE\Anaconda\envs\vin\python.exe' -m pytest tests\test_policy_rag.py -q --basetemp .pytest_tmp
```

Run one file:

```powershell
& 'D:\CODE\Anaconda\envs\vin\python.exe' -m pytest tests\test_routing.py -q --basetemp .pytest_tmp
```

Run one test:

```powershell
& 'D:\CODE\Anaconda\envs\vin\python.exe' -m pytest tests\test_routing.py::test_route_mixed_question -q --basetemp .pytest_tmp
```

## Mental Model

The app has three main layers:

1. Data and policy helpers
   - Read mock JSON data.
   - Parse/search local policy markdown.

2. Supervisor-worker graph
   - `route_question()` decides which workers are needed.
   - `ShoppingAssistant.ask()` runs the graph and returns route, worker results, answer, and trace.

3. Streamlit UI
   - `app.py` renders chat, trace views, batch dashboard, and About page flowchart.
   - UI tests use fake Streamlit/session objects where possible.

Most tests check one layer at a time. That makes failures easier to understand.

## What Each Test File Covers

### `test_data_access.py`

Tests `ShoppingDataStore` in `src/app/data_access.py`.

Use this file when you change:

- customer lookup
- order lookup
- voucher filtering
- `data/order_customer_mock_data.json` shape

Typical failure meaning: the local mock data access contract changed.

### `test_policy_rag.py`

Tests policy parsing and vector-store search.

It has two parts:

- `parse_policy_markdown()` should turn policy markdown into H2/H3 chunks with citations.
- `ChromaPolicyStore` should rebuild/search with fake embeddings.

Use this file when you change:

- `data/policy_mock_vi.md`
- `src/rag/parser.py`
- `src/rag/vector_store.py`

### `test_routing.py`

Tests supervisor routing only, without running the full graph.

This is the fastest place to test:

- policy-only route
- data-only route
- mixed policy + data route
- clarification-needed route
- guardrail-blocked route for abusive or hateful input

If a user question is being sent to the wrong worker, start here.

### `test_graph_execution.py`

Tests `ShoppingAssistant.ask()` end-to-end through the synchronous graph.

It checks that:

- clarification responses write traces
- missing orders become `not_found`
- guardrail-blocked comments skip policy/data workers

Use this when routing alone is not enough and you need to verify graph execution, final answer, and trace together.

### `test_batch.py`

Tests `ShoppingAssistant.run_batch()`.

It runs `data/test.json`, writes a summary, and writes one trace file per case.

Use this when you change:

- batch evaluation
- output summary format
- trace output paths
- expected case count in `data/test.json`

### `test_tracing.py`

Tests `make_trace_event()` in `src/app/tracing.py`.

It protects the trace event schema:

- run ID
- case ID
- node
- event
- status
- input/output payload
- warnings/error
- latency
- timestamp

Use this before changing trace fields because Streamlit and batch outputs depend on this structure.

### `test_recommendations.py`

Tests `recommend_improvements()` in `src/app/graph.py`.

It verifies that weak batch metrics produce useful improvement recommendations.

Use this when you change:

- recommendation wording
- priority logic
- batch metric interpretation

### `test_streamlit_chat.py`

Tests Streamlit-facing helpers in `app.py` without launching a browser.

It uses fake modules and fake session state to test:

- chat history initialization
- appending one chat turn
- default supervisor model configuration
- loading the Mermaid flowchart from the report
- rendering only Mermaid code into the Streamlit HTML component

Use this when you change:

- chat tab behavior
- About tab flowchart behavior
- helper functions in `app.py`

### `test_model.py`

Optional live external-provider smoke tests.

These are skipped unless `RUN_EXTERNAL_PROVIDER_SMOKE=1` is set. They may call real services and require API keys:

- Jina
- OpenRouter
- OpenAI
- Mistral

Do not run these as part of normal local verification unless you intentionally want live API smoke tests.

Example:

```powershell
$env:RUN_EXTERNAL_PROVIDER_SMOKE='1'
$env:OPENAI_API_KEY='...'
& 'D:\CODE\Anaconda\envs\vin\python.exe' -m pytest tests\test_model.py::test_openai_model_smoke -q -s
```

### `test_langsmith.py`

This file is more like a manual experiment than a normal unit test. It creates and invokes a LangChain agent at import time.

Treat it carefully:

- It may require provider credentials.
- It may make live model calls.
- It is not part of the recommended safe local test set above.

## Common Workflows

### I changed supervisor routing

Run:

```powershell
& 'D:\CODE\Anaconda\envs\vin\python.exe' -m pytest tests\test_routing.py tests\test_graph_execution.py -q --basetemp .pytest_tmp
```

Start with `test_routing.py` because it isolates `route_question()`. Then use `test_graph_execution.py` to confirm the whole graph still behaves.

### I changed the Streamlit app

Run:

```powershell
& 'D:\CODE\Anaconda\envs\vin\python.exe' -m pytest tests\test_streamlit_chat.py -q --basetemp .pytest_tmp
& 'D:\CODE\Anaconda\envs\vin\python.exe' -m py_compile app.py
```

The tests do not launch Streamlit. They test helper functions and state transitions directly.

### I changed data or policy files

Run:

```powershell
& 'D:\CODE\Anaconda\envs\vin\python.exe' -m pytest tests\test_data_access.py tests\test_batch.py -q --basetemp .pytest_tmp
& 'D:\CODE\Anaconda\envs\vin\python.exe' -m pytest tests\test_policy_rag.py -q --basetemp .pytest_tmp
```

This checks local JSON lookup, policy parsing/search, and batch output.

### I changed trace format

Run:

```powershell
& 'D:\CODE\Anaconda\envs\vin\python.exe' -m pytest tests\test_tracing.py tests\test_graph_execution.py tests\test_streamlit_chat.py -q --basetemp .pytest_tmp
```

Trace format affects graph output and Streamlit display.

## How To Add A New Test

Pick the narrowest layer that proves the behavior:

- If you only changed route decisions, add to `test_routing.py`.
- If you need final answer and trace, add to `test_graph_execution.py`.
- If you changed local data lookup, add to `test_data_access.py`.
- If you changed Streamlit helpers, add to `test_streamlit_chat.py`.

Prefer tests shaped like this:

```python
def test_specific_behavior():
    result = function_under_test("input")

    assert result["status"] == "expected_status"
```

Avoid tests that require a browser, real API key, or live LLM unless the test is explicitly marked as an external smoke test.

## Reading A Failure

When a test fails, first identify the layer:

- `test_routing.py` failed: route classifier logic is wrong.
- `test_graph_execution.py` failed: graph path, final answer, or trace is wrong.
- `test_streamlit_chat.py` failed: UI helper/state behavior is wrong.
- `test_data_access.py` failed: local JSON lookup is wrong.
- `test_policy_rag.py` failed: policy parsing or retrieval is wrong.

Then run only the failing test with `-q` removed for more detail:

```powershell
& 'D:\CODE\Anaconda\envs\vin\python.exe' -m pytest tests\test_routing.py::test_route_mixed_question --basetemp .pytest_tmp
```

Small, focused runs are faster and easier to debug than rerunning everything.
