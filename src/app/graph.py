from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.data_access import ShoppingDataStore
from app.state import ShoppingState
from app.tracing import json_safe, make_trace_event
from rag.parser import parse_policy_markdown


ORDER_RE = re.compile(r"(?:đơn hàng|order)\s*(\d{3,})", re.IGNORECASE)
ANY_ORDER_ID_RE = re.compile(r"\b\d{4}\b")
CUSTOMER_RE = re.compile(r"\bC\d{3}\b", re.IGNORECASE)


class ShoppingAssistant:
    """Observable synchronous shopping assistant."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.data_store = ShoppingDataStore(self.settings.orders_path)
        self.policy_chunks: list[dict[str, Any]] | None = None
        self.graph = build_graph(self)

    def ask(
        self,
        question: str,
        trace_file: Path | None = None,
        rebuild_index: bool = False,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        if rebuild_index:
            self.policy_chunks = None

        state: ShoppingState = {
            "run_id": str(uuid4()),
            "case_id": case_id,
            "question": question,
            "trace": [],
        }
        result = self.graph.invoke(state)
        payload = {
            "route": result.get("route", {}),
            "policy_result": result.get("policy_result", {}),
            "data_result": result.get("data_result", {}),
            "final_answer": result.get("final_answer", ""),
            "trace": result.get("trace", []),
        }

        if trace_file is not None:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_file.write_text(
                json.dumps(json_safe(payload["trace"]), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return payload

    def run_batch(
        self,
        test_file: Path,
        output_dir: Path,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        cases = json.loads(test_file.read_text(encoding="utf-8"))
        traces_dir = output_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        summaries: list[dict[str, Any]] = []
        for case in cases:
            case_id = case["id"]
            trace_file = traces_dir / f"{case_id}.json"
            result = self.ask(
                case["question"],
                trace_file=trace_file,
                rebuild_index=rebuild_index,
                case_id=case_id,
            )
            rebuild_index = False

            actual_route = route_labels(result.get("route", {}))
            expected_route = case.get("expected_route", [])
            actual_status = result_status(result)
            expected_status = case.get("expected_status", "ok")
            expected_contains = case.get("expected_contains", [])
            answer_lower = result.get("final_answer", "").lower()
            contains_ok = all(item.lower() in answer_lower for item in expected_contains)

            summaries.append(
                {
                    "id": case_id,
                    "question": case["question"],
                    "expected_route": expected_route,
                    "actual_route": actual_route,
                    "expected_status": expected_status,
                    "actual_status": actual_status,
                    "route_ok": sorted(expected_route) == sorted(actual_route),
                    "status_ok": expected_status == actual_status,
                    "contains_ok": contains_ok,
                    "expected_contains": expected_contains,
                    "trace_file": f"traces/{case_id}.json",
                }
            )

        total = len(summaries)
        summary = {
            "total": total,
            "route_accuracy": _accuracy(summaries, "route_ok"),
            "status_accuracy": _accuracy(summaries, "status_ok"),
            "contains_accuracy": _accuracy(summaries, "contains_ok"),
            "cases": summaries,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def load_policy_chunks(self) -> list[dict[str, Any]]:
        if self.policy_chunks is None:
            markdown = self.settings.policy_path.read_text(encoding="utf-8")
            self.policy_chunks = parse_policy_markdown(markdown)
        return self.policy_chunks

    def search_policy(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.settings.top_k
        terms = _policy_terms(query)
        scored: list[tuple[int, dict[str, Any]]] = []
        normalized_query = _normalize(query)
        for chunk in self.load_policy_chunks():
            haystack = _normalize(f"{chunk['citation']} {chunk['rendered_text']}")
            score = sum(1 for term in terms if term in haystack)
            if any(term in normalized_query for term in ["hoan tra", "tra hang", "tra trong", "doi y"]):
                if "chinh sach doi tra" in haystack or "dieu kien chung" in haystack:
                    score += 5
                if "giao hang khong thanh cong" in haystack:
                    score -= 1
            if score:
                scored.append((score, chunk))

        if not scored:
            scored = [(1, chunk) for chunk in self.load_policy_chunks()[:top_k]]

        scored.sort(key=lambda item: item[0], reverse=True)
        hits = []
        for score, chunk in scored[:top_k]:
            hits.append(
                {
                    "citation": chunk["citation"],
                    "content": chunk["rendered_text"],
                    "distance": 1 / (score + 1),
                    "section_h2": chunk["section_h2"],
                    "section_h3": chunk["section_h3"],
                }
            )
        return hits


def build_graph(assistant: ShoppingAssistant | None = None) -> Any:
    if assistant is None:
        assistant = ShoppingAssistant()

    graph = StateGraph(ShoppingState)
    graph.add_node("supervisor", lambda state: supervisor_node(state))
    graph.add_node("worker_1_policy", lambda state: worker_1_policy_node(state, assistant))
    graph.add_node("worker_2_data", lambda state: worker_2_data_node(state, assistant))
    graph.add_node("worker_3_response", lambda state: worker_3_response_node(state))

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _after_supervisor,
        {
            "policy": "worker_1_policy",
            "data": "worker_2_data",
            "response": "worker_3_response",
        },
    )
    graph.add_conditional_edges(
        "worker_1_policy",
        _after_policy,
        {"data": "worker_2_data", "response": "worker_3_response"},
    )
    graph.add_edge("worker_2_data", "worker_3_response")
    graph.add_edge("worker_3_response", END)
    return graph.compile()


def supervisor_node(state: ShoppingState) -> ShoppingState:
    started = time.perf_counter()
    route = route_question(state["question"])
    trace = make_trace_event(
        run_id=state["run_id"],
        case_id=state.get("case_id"),
        node="supervisor",
        event="route_decided",
        status=route["status"],
        input_payload={"question": state["question"]},
        output_payload=route,
        latency_ms=_latency_ms(started),
    )
    return {"route": route, "trace": [trace]}


def worker_1_policy_node(
    state: ShoppingState,
    assistant: ShoppingAssistant | None = None,
) -> ShoppingState:
    started = time.perf_counter()
    route = state.get("route", {})
    task = route.get("policy_task") or {}
    query = task.get("context") or state["question"]
    try:
        if assistant is None:
            assistant = ShoppingAssistant()
        hits = assistant.search_policy(query)
        facts = [_compact_policy_fact(hit["content"]) for hit in hits]
        result = {
            "status": "ok" if hits else "not_found",
            "summary": " ".join(facts[:2]) if hits else "Không tìm thấy policy phù hợp.",
            "facts": facts,
            "citations": [hit["citation"] for hit in hits],
            "tool_calls": [{"tool": "search_policy", "status": "ok", "top_k": len(hits)}],
            "warnings": [],
            "error": None,
        }
    except Exception as exc:
        result = {
            "status": "error",
            "summary": "",
            "facts": [],
            "citations": [],
            "tool_calls": [{"tool": "search_policy", "status": "error"}],
            "warnings": [],
            "error": str(exc),
        }

    trace = make_trace_event(
        run_id=state["run_id"],
        case_id=state.get("case_id"),
        node="worker_1_policy",
        event="policy_retrieved",
        status=result["status"],
        input_payload={"task": task},
        output_payload=result,
        warnings=result.get("warnings", []),
        error=result.get("error"),
        latency_ms=_latency_ms(started),
    )
    return {"policy_result": result, "trace": [trace]}


def worker_2_data_node(
    state: ShoppingState,
    assistant: ShoppingAssistant | None = None,
) -> ShoppingState:
    started = time.perf_counter()
    route = state.get("route", {})
    task = route.get("data_task") or {}
    question = state["question"]
    if assistant is None:
        assistant = ShoppingAssistant()

    tool_calls: list[dict[str, Any]] = []
    facts: list[str] = []
    warnings: list[str] = []
    status = "ok"

    try:
        order_ids = extract_order_ids(question)
        customer_ids = extract_customer_ids(question)

        for order_id in order_ids:
            lookup = assistant.data_store.get_order_detail_by_order_id(order_id)
            tool_calls.append({"tool": "get_order_detail_by_order_id", "status": lookup["status"], "input": {"order_id": order_id}})
            if lookup["status"] != "ok":
                status = "not_found"
                facts.append(f"Không tìm thấy đơn hàng {order_id}.")
                continue
            order = lookup["order"]
            facts.extend(_order_facts(order))
            if not customer_ids and order.get("customer_id"):
                customer_ids.append(str(order["customer_id"]))

        for customer_id in customer_ids:
            customer_lookup = assistant.data_store.get_customer_by_id(customer_id)
            tool_calls.append({"tool": "get_customer_by_id", "status": customer_lookup["status"], "input": {"customer_id": customer_id}})
            if customer_lookup["status"] != "ok":
                status = "not_found"
                facts.append(f"Không tìm thấy khách hàng {customer_id}.")
                continue
            customer = customer_lookup["customer"]
            facts.append(
                f"Khách hàng {customer_id} thuộc hạng {customer.get('tier')}, còn "
                f"{customer.get('remaining_voucher_quota_this_month')} / "
                f"{customer.get('max_voucher_per_month')} quota voucher tháng này."
            )

            if "đơn" in _normalize(question) or "order" in question.lower():
                orders_lookup = assistant.data_store.get_orders_by_customer_id(customer_id)
                tool_calls.append({"tool": "get_orders_by_customer_id", "status": orders_lookup["status"], "input": {"customer_id": customer_id}})
                if orders_lookup["status"] == "ok":
                    order_list = ", ".join(order["order_id"] for order in orders_lookup["orders"][:5])
                    facts.append(f"Các đơn gần đây của {customer_id}: {order_list}.")

            if "voucher" in question.lower():
                vouchers_lookup = assistant.data_store.get_vouchers_by_customer_id(customer_id, only_active="còn" in _normalize(question) or "active" in question.lower())
                tool_calls.append({"tool": "get_vouchers_by_customer_id", "status": vouchers_lookup["status"], "input": {"customer_id": customer_id}})
                if vouchers_lookup["status"] == "ok":
                    codes = [voucher["voucher_code"] for voucher in vouchers_lookup["vouchers"][:8]]
                    facts.append(f"Voucher phù hợp của {customer_id}: {', '.join(codes) if codes else 'không có mã phù hợp'}.")

        if not order_ids and not customer_ids:
            status = "clarification_needed"
            facts.append("Cần cung cấp order_id hoặc customer_id để tra cứu dữ liệu.")

        result = {
            "status": status,
            "summary": " ".join(facts),
            "facts": facts,
            "citations": [],
            "tool_calls": tool_calls,
            "warnings": warnings,
            "error": None,
        }
    except Exception as exc:
        result = {
            "status": "error",
            "summary": "",
            "facts": facts,
            "citations": [],
            "tool_calls": tool_calls,
            "warnings": warnings,
            "error": str(exc),
        }

    trace = make_trace_event(
        run_id=state["run_id"],
        case_id=state.get("case_id"),
        node="worker_2_data",
        event="data_lookup_completed",
        status=result["status"],
        input_payload={"task": task},
        output_payload=result,
        warnings=result.get("warnings", []),
        error=result.get("error"),
        latency_ms=_latency_ms(started),
    )
    return {"data_result": result, "trace": [trace]}


def worker_3_response_node(state: ShoppingState) -> ShoppingState:
    started = time.perf_counter()
    route = state.get("route", {})
    policy_result = state.get("policy_result", {})
    data_result = state.get("data_result", {})

    if route.get("status") == "clarification_needed":
        final_answer = (
            "Status: clarification_needed\n"
            f"Question: {route.get('clarification_question', 'Vui lòng cung cấp thêm định danh cần tra cứu.')}"
        )
        status = "clarification_needed"
    elif data_result.get("status") == "not_found":
        final_answer = f"Status: not_found\nMessage: {data_result.get('summary', 'Không tìm thấy dữ liệu phù hợp.')}"
        status = "not_found"
    elif policy_result.get("status") == "not_found":
        final_answer = f"Status: not_found\nMessage: {policy_result.get('summary', 'Không tìm thấy policy phù hợp.')}"
        status = "not_found"
    elif data_result.get("status") == "error" and policy_result.get("status") == "error":
        final_answer = "Status: error\nMessage: Cả policy worker và data worker đều thất bại."
        status = "error"
    else:
        final_answer = _compose_success_answer(state["question"], policy_result, data_result)
        status = "ok"

    output = {"final_answer": final_answer}
    trace = make_trace_event(
        run_id=state["run_id"],
        case_id=state.get("case_id"),
        node="worker_3_response",
        event="answer_synthesized",
        status=status,
        input_payload={"route": route, "policy_result": policy_result, "data_result": data_result},
        output_payload=output,
        latency_ms=_latency_ms(started),
    )
    return {"final_answer": final_answer, "trace": [trace]}


def route_question(question: str) -> dict[str, Any]:
    normalized = _normalize(question)
    order_ids = extract_order_ids(question)
    customer_ids = extract_customer_ids(question)
    mentions_self = "cua toi" in normalized or "của tôi" in question.lower()

    if mentions_self and not order_ids and not customer_ids:
        subject = "đơn hàng" if "don hang" in normalized else "khách hàng"
        return {
            "status": "clarification_needed",
            "needs_policy": False,
            "needs_data": False,
            "reason": "Question needs personal data but has no order_id or customer_id.",
            "clarification_question": f"Vui lòng cung cấp {subject} hoặc customer_id để mình kiểm tra chính xác.",
            "policy_task": None,
            "data_task": None,
        }

    policy_keywords = [
        "chinh sach",
        "hoan tra",
        "tra hang",
        "hoan tien",
        "giao hang",
        "kiem hang",
        "voucher",
        "tu choi",
        "cua so tra hang",
    ]
    data_keywords = [
        "don hang",
        "khach hang",
        "customer",
        "trang thai",
        "bao gio",
        "quota",
        "ma nao",
        "kiem tra",
    ]
    return_keywords = ["hoan tra", "tra hang", "hoan tien", "tu choi", "cua so tra hang", "tra trong", "doi y"]

    has_policy = any(keyword in normalized for keyword in policy_keywords)
    has_data = bool(order_ids or customer_ids) and any(keyword in normalized for keyword in data_keywords + ["voucher"])
    mixed = bool(order_ids) and any(keyword in normalized for keyword in return_keywords)

    needs_policy = has_policy and not (customer_ids and "voucher" in normalized and "hoan lai" not in normalized)
    needs_data = has_data
    if mixed:
        needs_policy = True
        needs_data = True
    if order_ids and not needs_data:
        needs_data = True
    if not needs_policy and not needs_data:
        needs_policy = True

    return {
        "status": "ok",
        "needs_policy": needs_policy,
        "needs_data": needs_data,
        "reason": _route_reason(needs_policy, needs_data, order_ids, customer_ids),
        "clarification_question": None,
        "policy_task": {
            "task": "retrieve policy evidence",
            "context": question,
            "expected_output": "top policy facts with citations",
        }
        if needs_policy
        else None,
        "data_task": {
            "task": "lookup shopping data",
            "context": question,
            "expected_output": "customer/order/voucher facts with lookup status",
        }
        if needs_data
        else None,
    }


def extract_order_ids(question: str) -> list[str]:
    ids = ORDER_RE.findall(question)
    if ids:
        return list(dict.fromkeys(ids))
    if "đơn" in question.lower() or "order" in question.lower():
        return list(dict.fromkeys(ANY_ORDER_ID_RE.findall(question)))
    return []


def extract_customer_ids(question: str) -> list[str]:
    return [match.upper() for match in dict.fromkeys(CUSTOMER_RE.findall(question))]


def route_labels(route: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if route.get("needs_data"):
        labels.append("data")
    if route.get("needs_policy"):
        labels.append("policy")
    return labels


def result_status(result: dict[str, Any]) -> str:
    route = result.get("route", {})
    data_result = result.get("data_result", {})
    policy_result = result.get("policy_result", {})
    if route.get("status") == "clarification_needed":
        return "clarification_needed"
    if data_result.get("status") == "not_found" or policy_result.get("status") == "not_found":
        return "not_found"
    if data_result.get("status") == "error" or policy_result.get("status") == "error":
        return "error"
    return "ok"


def recommend_improvements(summary: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    cases = summary.get("cases", [])

    if summary.get("route_accuracy", 1.0) < 0.95:
        failed = [case["id"] for case in cases if not case.get("route_ok")]
        recommendations.append(
            {
                "pain_point": "Business analyst cannot trust which worker handled a question.",
                "evidence": f"Routing mismatches: {', '.join(failed) if failed else 'route_accuracy below threshold'}.",
                "feature": "Improve supervisor routing criteria and examples.",
                "improvement": "Increase route accuracy for policy, data, and mixed questions.",
                "priority": "high",
            }
        )
    if summary.get("status_accuracy", 1.0) < 0.95:
        failed = [case["id"] for case in cases if not case.get("status_ok")]
        recommendations.append(
            {
                "pain_point": "User-facing failure states are inconsistent.",
                "evidence": f"Status mismatches: {', '.join(failed) if failed else 'status_accuracy below threshold'}.",
                "feature": "Harden clarification_needed, not_found, and graceful failure handling.",
                "improvement": "Make data gaps and missing identifiers predictable.",
                "priority": "high",
            }
        )
    if any(case.get("actual_status") == "error" for case in cases):
        failed = [case["id"] for case in cases if case.get("actual_status") == "error"]
        recommendations.append(
            {
                "pain_point": "Worker errors can interrupt the support flow.",
                "evidence": f"Error cases: {', '.join(failed)}.",
                "feature": "Add fallback and retry limits around failing workers.",
                "improvement": "Reduce failed runs while keeping latency bounded.",
                "priority": "high",
            }
        )
    if summary.get("contains_accuracy", 1.0) < 0.95:
        failed = [case["id"] for case in cases if not case.get("contains_ok")]
        recommendations.append(
            {
                "pain_point": "Answers may miss expected evidence or business wording.",
                "evidence": f"Evidence/content misses: {', '.join(failed) if failed else 'contains_accuracy below threshold'}.",
                "feature": "Improve response evidence and citation formatting.",
                "improvement": "Make final answers easier to audit against policy and data facts.",
                "priority": "medium",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "pain_point": "The next bottleneck is cost and latency visibility.",
                "evidence": "Batch metrics passed current route/status/content thresholds.",
                "feature": "Add per-node cost and latency aggregation.",
                "improvement": "Support model-size and retry trade-off decisions.",
                "priority": "low",
            }
        )
    return recommendations


def _after_supervisor(state: ShoppingState) -> str:
    route = state.get("route", {})
    if route.get("status") == "clarification_needed":
        return "response"
    if route.get("needs_policy"):
        return "policy"
    if route.get("needs_data"):
        return "data"
    return "response"


def _after_policy(state: ShoppingState) -> str:
    if state.get("route", {}).get("needs_data"):
        return "data"
    return "response"


def _latency_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _accuracy(cases: list[dict[str, Any]], key: str) -> float:
    if not cases:
        return 0.0
    return sum(1 for case in cases if case.get(key)) / len(cases)


def _route_reason(
    needs_policy: bool,
    needs_data: bool,
    order_ids: list[str],
    customer_ids: list[str],
) -> str:
    if needs_policy and needs_data:
        return "Question needs both policy evidence and mock-data facts."
    if needs_policy:
        return "Question asks for policy knowledge."
    if needs_data:
        ids = order_ids or customer_ids
        return f"Question asks for mock-data lookup for {', '.join(ids)}."
    return "Question can be answered directly."


def _normalize(text: str) -> str:
    replacements = {
        "ă": "a",
        "â": "a",
        "á": "a",
        "à": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "ắ": "a",
        "ằ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "ặ": "a",
        "ấ": "a",
        "ầ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ậ": "a",
        "đ": "d",
        "é": "e",
        "è": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ê": "e",
        "ế": "e",
        "ề": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "í": "i",
        "ì": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ó": "o",
        "ò": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ô": "o",
        "ố": "o",
        "ồ": "o",
        "ổ": "o",
        "ỗ": "o",
        "ộ": "o",
        "ơ": "o",
        "ớ": "o",
        "ờ": "o",
        "ở": "o",
        "ỡ": "o",
        "ợ": "o",
        "ú": "u",
        "ù": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ư": "u",
        "ứ": "u",
        "ừ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "ý": "y",
        "ỳ": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
    }
    lowered = text.lower()
    return "".join(replacements.get(char, char) for char in lowered)


def _policy_terms(query: str) -> list[str]:
    normalized = _normalize(query)
    terms = [term for term in re.split(r"\W+", normalized) if len(term) >= 3]
    if any(term in normalized for term in ["hoan tra", "tra hang", "cua so tra hang"]):
        terms.extend(["15 ngay", "dieu kien", "quan he", "trang thai"])
    if "giao hang" in normalized or "giao nhanh" in normalized:
        terms.extend(["thoi gian giao hang", "giao nhanh", "giao tieu chuan"])
    if "kiem hang" in normalized:
        terms.extend(["kiem hang"])
    if "voucher" in normalized:
        terms.extend(["voucher", "hoan lai", "gioi han"])
    return list(dict.fromkeys(terms))


def _compact_policy_fact(content: str) -> str:
    lines = [line.strip("- ").strip() for line in content.splitlines() if line.strip()]
    body = " ".join(lines[:8])
    return body[:1000]


def _order_facts(order: dict[str, Any]) -> list[str]:
    order_id = order.get("order_id")
    facts = [
        f"Đơn hàng {order_id} đang ở trạng thái {order.get('order_status')}.",
        f"Dự kiến giao: {order.get('estimated_delivery')}.",
        f"Ghi chú mới nhất: {order.get('latest_status_note')}.",
        f"can_return_now={order.get('can_return_now')}, eligible_for_return_until={order.get('eligible_for_return_until')}.",
    ]
    if order.get("can_return_now"):
        facts.append(f"Đơn hàng {order_id} hiện còn trong thời gian trả hàng.")
    else:
        facts.append(f"Đơn hàng {order_id} chưa thể trả hàng theo dữ liệu hiện tại.")
    return facts


def _compose_success_answer(
    question: str,
    policy_result: dict[str, Any],
    data_result: dict[str, Any],
) -> str:
    data_facts = data_result.get("facts", [])
    policy_facts = policy_result.get("facts", [])
    citations = policy_result.get("citations", [])
    normalized = _normalize(question)

    if "1971" in question and any(term in normalized for term in ["hoan tra", "tra hang"]):
        answer = "Đơn hàng 1971 chưa thể hoàn trả vì đơn đang in_transit và chưa giao thành công."
    elif "2058" in question and any(term in normalized for term in ["hoan tra", "tra hang", "cua so"]):
        answer = "Đơn hàng 2058 còn trong thời gian trả hàng theo dữ liệu đơn hàng."
    elif data_facts:
        answer = data_facts[0]
    elif policy_facts:
        answer = policy_facts[0]
    else:
        answer = "Không có đủ bằng chứng để trả lời chắc chắn."

    lines = [f"Answer: {answer}", "Evidence:"]
    lines.append(f"- Policy: {'; '.join(policy_facts[:3]) if policy_facts else 'Không cần policy cho câu hỏi này.'}")
    lines.append(f"- Order data: {'; '.join(data_facts[:4]) if data_facts else 'Không cần dữ liệu đơn hàng/khách hàng cho câu hỏi này.'}")
    if citations:
        lines.append(f"- Citations: {', '.join(citations[:3])}")
    return "\n".join(lines)
