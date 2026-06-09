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
