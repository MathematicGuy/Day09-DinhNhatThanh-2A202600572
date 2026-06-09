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
