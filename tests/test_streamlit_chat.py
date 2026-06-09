from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def load_streamlit_app(monkeypatch):
    fake_graph = ModuleType("app.graph")
    fake_graph.ShoppingAssistant = object
    fake_graph.recommend_improvements = lambda summary: []
    monkeypatch.setitem(sys.modules, "app.graph", fake_graph)

    spec = importlib.util.spec_from_file_location("streamlit_app_under_test", Path("app.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class FakeStreamlit:
    def __init__(self) -> None:
        self.session_state = FakeSessionState()


class FakeAssistant:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def ask(self, question: str) -> dict:
        self.questions.append(question)
        return {
            "final_answer": f"Answer for {question}",
            "route": {"status": "ok"},
            "policy_result": {},
            "data_result": {},
            "trace": [{"node": "supervisor", "event": "route_decided"}],
        }


def test_initialize_state_creates_empty_chat_history(monkeypatch):
    module = load_streamlit_app(monkeypatch)
    module.st = FakeStreamlit()

    module.initialize_state()

    assert module.st.session_state.chat_messages == []


def test_append_chat_turn_runs_assistant_and_stores_trace(monkeypatch):
    module = load_streamlit_app(monkeypatch)
    module.st = FakeStreamlit()
    module.initialize_state()
    assistant = FakeAssistant()
    monkeypatch.setattr(module, "get_assistant", lambda: assistant)

    turn = module.append_chat_turn("Đơn hàng 1971 có được hoàn trả không?")

    assert assistant.questions == ["Đơn hàng 1971 có được hoàn trả không?"]
    assert turn == module.st.session_state.chat_messages[0]
    assert turn["question"] == "Đơn hàng 1971 có được hoàn trả không?"
    assert turn["answer"] == "Answer for Đơn hàng 1971 có được hoàn trả không?"
    assert turn["trace"] == [{"node": "supervisor", "event": "route_decided"}]
    assert turn["result"]["route"]["status"] == "ok"


def test_configure_streamlit_supervisor_model_defaults_to_gpt_54_nano(monkeypatch):
    module = load_streamlit_app(monkeypatch)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    module.configure_streamlit_supervisor_model()

    assert os.environ["LLM_PROVIDER"] == "openai"
    assert os.environ["LLM_MODEL"] == "gpt-5.4-nano"


def test_load_report_flowchart_extracts_mermaid_block(monkeypatch, tmp_path):
    module = load_streamlit_app(monkeypatch)
    report = tmp_path / "report.md"
    report.write_text(
        "# Report\n\n```mermaid\nflowchart TD\n    A --> B\n```\n",
        encoding="utf-8",
    )

    flowchart = module.load_report_flowchart(report)

    assert flowchart == "flowchart TD\n    A --> B"


def test_load_report_flowchart_returns_empty_string_when_missing(monkeypatch, tmp_path):
    module = load_streamlit_app(monkeypatch)
    report = tmp_path / "report.md"
    report.write_text("# Report\n\nNo diagram here.\n", encoding="utf-8")

    assert module.load_report_flowchart(report) == ""


def test_render_mermaid_flowchart_uses_only_mermaid_pre_block(monkeypatch):
    module = load_streamlit_app(monkeypatch)
    rendered: dict[str, object] = {}
    fake_streamlit = ModuleType("streamlit")
    fake_components_package = ModuleType("streamlit.components")
    fake_components = ModuleType("streamlit.components.v1")

    def fake_html(html: str, height: int, scrolling: bool) -> None:
        rendered["html"] = html
        rendered["height"] = height
        rendered["scrolling"] = scrolling

    fake_components.html = fake_html
    fake_components_package.v1 = fake_components
    fake_streamlit.components = fake_components_package
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "streamlit.components", fake_components_package)
    monkeypatch.setitem(sys.modules, "streamlit.components.v1", fake_components)

    module.render_mermaid_flowchart("flowchart TD\n    A --> B")

    html = str(rendered["html"])
    assert '<pre class="mermaid">' in html
    assert "flowchart TD" in html
    assert "A --&gt; B" in html
    assert "supervisor_worker_agent_report.md" not in html
    assert rendered["height"] == 900
    assert rendered["scrolling"] is True
