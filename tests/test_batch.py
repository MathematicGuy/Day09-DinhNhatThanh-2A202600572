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
