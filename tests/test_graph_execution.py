from app.config import Settings
import app.graph as graph_module
from app.graph import ShoppingAssistant


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


def test_ask_guardrail_blocked_skips_workers(tmp_path):
    settings = Settings.load()
    settings.traces_dir = tmp_path
    assistant = ShoppingAssistant(settings=settings)

    result = assistant.ask("You are stupid and useless")

    assert result["route"]["status"] == "guardrail_blocked"
    assert result["route"]["guardrail_category"] == "abusive"
    assert result["policy_result"] == {}
    assert result["data_result"] == {}
    assert "Status: guardrail_blocked" in result["final_answer"]
    assert "mình vẫn sẵn sàng hỗ trợ" in result["final_answer"]
    assert [event["node"] for event in result["trace"]] == ["supervisor", "worker_3_response"]


def test_ask_uses_openai_response_model_when_configured(tmp_path, monkeypatch):
    class FakeResponse:
        content = (
            '{"status":"ok","final_answer":"Answer: mocked model response\\n'
            'Evidence:\\n- Policy: mocked\\n- Order data: mocked"}'
        )

    class FakeModel:
        model_name = "fake-openai"

        def invoke(self, messages):
            return FakeResponse()

    settings = Settings.load()
    settings.traces_dir = tmp_path
    settings.provider = "openai"
    settings.openai_api_key = "test-key"
    monkeypatch.setattr(graph_module, "get_chat_model", lambda _: FakeModel())
    assistant = ShoppingAssistant(settings=settings)

    result = assistant.ask("Chính sách hoàn trả hàng ra sao?")

    assert result["final_answer"] == (
        "Answer: mocked model response\nEvidence:\n- Policy: mocked\n- Order data: mocked"
    )
    assert result["trace"][-1]["output"]["response_source"] == "openai"
    assert result["trace"][-1]["output"]["response_model"] == "fake-openai"


def test_ask_uses_openai_supervisor_model_when_configured(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content

    class FakeModel:
        model_name = "fake-openai"

        def invoke(self, messages):
            prompt = str(messages[0].content)
            if "You are the supervisor for a synchronous shopping-assistant graph." in prompt:
                return FakeResponse(
                    '{"status":"ok","needs_policy":true,"needs_data":false,'
                    '"reason":"model route","clarification_question":null,'
                    '"policy_task":{"task":"retrieve policy evidence","context":"Chính sách hoàn trả hàng ra sao?","expected_output":"top policy facts with citations"},'
                    '"data_task":null}'
                )
            return FakeResponse(
                '{"status":"ok","final_answer":"Answer: model response\\n'
                'Evidence:\\n- Policy: mocked\\n- Order data: not needed"}'
            )

    settings = Settings.load()
    settings.traces_dir = tmp_path
    settings.provider = "openai"
    settings.openai_api_key = "test-key"
    monkeypatch.setattr(graph_module, "get_chat_model", lambda _: FakeModel())
    assistant = ShoppingAssistant(settings=settings)

    result = assistant.ask("Chính sách hoàn trả hàng ra sao?")

    assert result["route"]["reason"] == "model route"
    assert result["trace"][0]["output"]["route_source"] == "openai"
    assert result["trace"][0]["output"]["route_model"] == "fake-openai"
    assert result["final_answer"] == (
        "Answer: model response\nEvidence:\n- Policy: mocked\n- Order data: not needed"
    )
