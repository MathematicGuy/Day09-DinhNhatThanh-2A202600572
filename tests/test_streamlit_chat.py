from __future__ import annotations

import importlib.util
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
