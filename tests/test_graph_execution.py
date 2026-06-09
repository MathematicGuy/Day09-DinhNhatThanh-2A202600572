from app.config import Settings
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
