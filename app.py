from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
__path__ = [str(SRC_DIR / "app")]

from app.graph import ShoppingAssistant, recommend_improvements  # noqa: E402


TEST_FILE = ROOT_DIR / "data" / "test.json"
OUTPUT_DIR = ROOT_DIR / "src" / "artifacts" / "streamlit_demo"
st: Any = None


def main() -> None:
    global get_assistant, load_cases, st
    import streamlit as streamlit

    st = streamlit
    get_assistant = st.cache_resource(_build_assistant)
    load_cases = st.cache_data(_load_cases)

    st.set_page_config(
        layout="wide",
        page_title="Supervisor-Worker Observability Demo",
    )
    initialize_state()

    st.title("Supervisor-Worker Observability Demo")
    st.caption(
        "Run shopping-support questions through the supervisor-worker graph and inspect routing, worker output, and traces."
    )

    ask_tab, batch_tab, about_tab = st.tabs(["Ask & Trace", "Batch Dashboard", "About"])
    with ask_tab:
        render_ask_tab()
    with batch_tab:
        render_batch_tab()
    with about_tab:
        render_about_tab()


def initialize_state() -> None:
    defaults = {
        "single_result": None,
        "single_question": "",
        "batch_summary": None,
        "batch_recommendations": None,
        "selected_trace": None,
        "selected_case_id": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_assistant() -> ShoppingAssistant:
    return _build_assistant()


def load_cases() -> list[dict[str, Any]]:
    return _load_cases()


def _build_assistant() -> ShoppingAssistant:
    return ShoppingAssistant()


def _load_cases() -> list[dict[str, Any]]:
    return json.loads(TEST_FILE.read_text(encoding="utf-8"))


def render_ask_tab() -> None:
    cases = load_cases()
    sample_options = [case["question"] for case in cases]
    default_question = "Đơn hàng 1971 có được hoàn trả không?"
    selected_sample = st.selectbox(
        "Sample question",
        sample_options,
        index=sample_options.index(default_question) if default_question in sample_options else 0,
    )
    question = st.text_input(
        "Question",
        value=st.session_state.single_question or selected_sample,
        placeholder="Nhập câu hỏi cần kiểm tra...",
    )

    run_col, clear_col = st.columns([1, 5])
    with run_col:
        run_clicked = st.button("Run question", type="primary", use_container_width=True)
    with clear_col:
        if st.button("Use selected sample"):
            st.session_state.single_question = selected_sample
            st.rerun()

    if run_clicked:
        if not question.strip():
            st.warning("Enter a question before running.")
        else:
            with st.spinner("Running graph..."):
                result = get_assistant().ask(question.strip())
            st.session_state.single_question = question.strip()
            st.session_state.single_result = result
            st.session_state.selected_trace = result.get("trace", [])
            st.session_state.selected_case_id = None

    result = st.session_state.single_result
    if not result:
        st.info("Run a question to inspect the supervisor decision, worker outputs, and node trace.")
        return

    st.subheader("Final Answer")
    st.code(result.get("final_answer", ""), language="text")

    route = result.get("route", {})
    render_route_summary(route)

    st.subheader("Worker Results")
    policy_col, data_col = st.columns(2)
    with policy_col:
        render_worker_card("Policy worker", result.get("policy_result", {}))
    with data_col:
        render_worker_card("Data worker", result.get("data_result", {}))

    trace = result.get("trace", [])
    render_trace_section(trace)
    render_json_download("Download raw trace JSON", "single_trace.json", trace)


def render_route_summary(route: dict[str, Any]) -> None:
    st.subheader("Route Decision")
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", route.get("status", "-"))
    col2.metric("Needs policy", str(bool(route.get("needs_policy"))))
    col3.metric("Needs data", str(bool(route.get("needs_data"))))
    st.write(route.get("reason", "No route reason recorded."))
    if route.get("clarification_question"):
        st.info(route["clarification_question"])


def render_worker_card(title: str, result: dict[str, Any]) -> None:
    if not result:
        st.container(border=True).write(f"{title}: not invoked")
        return

    with st.container(border=True):
        top_col, latency_col = st.columns([3, 1])
        top_col.markdown(f"**{title}**")
        latency_col.markdown(f"`{result.get('status', '-')}`")
        if result.get("summary"):
            st.write(result["summary"])
        if result.get("facts"):
            st.markdown("**Facts**")
            for fact in result["facts"]:
                st.markdown(f"- {fact}")
        if result.get("citations"):
            st.markdown("**Citations**")
            st.write(", ".join(result["citations"]))
        if result.get("tool_calls"):
            with st.expander("Tool calls", expanded=False):
                st.json(result["tool_calls"])
        if result.get("warnings"):
            st.warning("\n".join(result["warnings"]))
        if result.get("error"):
            st.error(result["error"])


def render_batch_tab() -> None:
    st.write(f"Dataset: `{TEST_FILE.relative_to(ROOT_DIR)}`")
    if st.button("Run batch", type="primary"):
        with st.spinner("Running test cases..."):
            summary = get_assistant().run_batch(TEST_FILE, OUTPUT_DIR)
            recommendations = recommend_improvements(summary)
            write_recommendations(recommendations)
        st.session_state.batch_summary = summary
        st.session_state.batch_recommendations = recommendations
        st.session_state.selected_case_id = None
        st.session_state.selected_trace = None

    summary = st.session_state.batch_summary
    if not summary:
        st.info("Run the batch to see accuracy metrics, case results, recommendations, and trace drill-down.")
        return

    render_batch_metrics(summary)
    render_case_table(summary)
    render_batch_trace_picker(summary)
    render_recommendations(st.session_state.batch_recommendations or [])

    summary_path = OUTPUT_DIR / "summary.json"
    recommendations_path = OUTPUT_DIR / "recommendations.json"
    download_col1, download_col2 = st.columns(2)
    with download_col1:
        render_file_download("Download summary.json", summary_path)
    with download_col2:
        render_file_download("Download recommendations.json", recommendations_path)


def render_batch_metrics(summary: dict[str, Any]) -> None:
    st.subheader("Batch Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", summary.get("total", 0))
    col2.metric("Route accuracy", format_accuracy(summary.get("route_accuracy")))
    col3.metric("Status accuracy", format_accuracy(summary.get("status_accuracy")))
    col4.metric("Contains accuracy", format_accuracy(summary.get("contains_accuracy")))


def render_case_table(summary: dict[str, Any]) -> None:
    st.subheader("Case Results")
    rows = []
    for case in summary.get("cases", []):
        rows.append(
            {
                "id": case.get("id"),
                "question": case.get("question"),
                "expected_route": ", ".join(case.get("expected_route", [])),
                "actual_route": ", ".join(case.get("actual_route", [])),
                "expected_status": case.get("expected_status"),
                "actual_status": case.get("actual_status"),
                "route_ok": case.get("route_ok"),
                "status_ok": case.get("status_ok"),
                "contains_ok": case.get("contains_ok"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_batch_trace_picker(summary: dict[str, Any]) -> None:
    st.subheader("Trace Drill-Down")
    cases = summary.get("cases", [])
    case_ids = [case["id"] for case in cases]
    if not case_ids:
        st.info("No cases available.")
        return

    selected = st.selectbox(
        "Case ID",
        case_ids,
        index=case_ids.index(st.session_state.selected_case_id)
        if st.session_state.selected_case_id in case_ids
        else 0,
    )
    case = next(item for item in cases if item["id"] == selected)
    trace_path = OUTPUT_DIR / case["trace_file"]
    if st.button("Load selected trace"):
        st.session_state.selected_case_id = selected
        st.session_state.selected_trace = json.loads(trace_path.read_text(encoding="utf-8"))

    if st.session_state.selected_trace and st.session_state.selected_case_id:
        st.caption(f"Showing trace for `{st.session_state.selected_case_id}`")
        render_trace_section(st.session_state.selected_trace)
        render_json_download(
            "Download selected trace JSON",
            f"{st.session_state.selected_case_id}_trace.json",
            st.session_state.selected_trace,
        )


def render_recommendations(recommendations: list[dict[str, Any]]) -> None:
    st.subheader("Recommendations")
    for item in recommendations:
        priority = item.get("priority", "unknown").upper()
        with st.container(border=True):
            st.markdown(f"**{priority}: {item.get('feature', 'Recommendation')}**")
            st.write(item.get("pain_point", ""))
            st.caption(item.get("evidence", ""))
            st.write(item.get("improvement", ""))


def render_trace_section(trace: list[dict[str, Any]]) -> None:
    st.subheader("Node Timeline")
    if not trace:
        st.info("No trace events recorded.")
        return

    timeline_rows = [
        {
            "node": event.get("node"),
            "event": event.get("event"),
            "status": event.get("status"),
            "latency_ms": event.get("latency_ms"),
            "warnings": len(event.get("warnings") or []),
            "error": bool(event.get("error")),
        }
        for event in trace
    ]
    st.dataframe(timeline_rows, use_container_width=True, hide_index=True)

    for node, events in group_trace_by_node(trace).items():
        with st.expander(f"{node} ({len(events)} event{'s' if len(events) != 1 else ''})", expanded=True):
            for event in events:
                st.markdown(
                    f"**{event.get('event', 'event')}** "
                    f"`{event.get('status', '-')}` `{event.get('latency_ms', 0)} ms`"
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Input")
                    st.json(event.get("input", {}))
                with col2:
                    st.caption("Output")
                    st.json(event.get("output", {}))
                if event.get("warnings"):
                    st.warning("\n".join(event["warnings"]))
                if event.get("error"):
                    st.error(event["error"])


def render_about_tab() -> None:
    st.subheader("Flow")
    st.code("User -> Supervisor -> Policy worker and/or Data worker -> Response worker", language="text")
    st.write(
        "The demo uses the current synchronous graph and JSON trace events. "
        "It is intended for local classroom inspection of routing, worker outputs, and batch quality signals."
    )
    st.write("No LangGraph Studio integration or graph behavior changes are included.")


def group_trace_by_node(trace: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in trace:
        grouped.setdefault(str(event.get("node", "unknown")), []).append(event)
    return grouped


def format_accuracy(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "-"


def render_json_download(label: str, filename: str, payload: Any) -> None:
    st.download_button(
        label,
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name=filename,
        mime="application/json",
    )


def render_file_download(label: str, path: Path) -> None:
    if not path.exists():
        st.download_button(label, data="{}", file_name=path.name, mime="application/json", disabled=True)
        return
    st.download_button(
        label,
        data=path.read_text(encoding="utf-8"),
        file_name=path.name,
        mime="application/json",
    )


def write_recommendations(recommendations: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "recommendations.json").write_text(
        json.dumps(recommendations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
