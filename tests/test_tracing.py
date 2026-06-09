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
