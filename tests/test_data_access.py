from pathlib import Path

from app.data_access import ShoppingDataStore


DATA_PATH = Path("data/order_customer_mock_data.json")


def test_get_customer_by_id_ok():
    store = ShoppingDataStore(DATA_PATH)
    result = store.get_customer_by_id("C001")
    assert result["status"] == "ok"
    assert result["customer"]["customer_id"] == "C001"


def test_get_order_detail_by_order_id_not_found():
    store = ShoppingDataStore(DATA_PATH)
    result = store.get_order_detail_by_order_id("9999")
    assert result["status"] == "not_found"
    assert result["order_id"] == "9999"


def test_get_vouchers_by_customer_id_only_active():
    store = ShoppingDataStore(DATA_PATH)
    result = store.get_vouchers_by_customer_id("C001", only_active=True)
    assert result["status"] == "ok"
    assert all(voucher["status"] == "active" for voucher in result["vouchers"])
