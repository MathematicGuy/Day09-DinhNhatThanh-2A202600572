"""Optional external-provider smoke tests.

These tests never contain hardcoded credentials. They are opt-in because the
default suite must stay offline and deterministic.
"""

from __future__ import annotations

import os
import pytest
import requests
from dotenv import load_dotenv
import json

load_dotenv()


def _require_external_smoke(provider_name: str):
    if os.getenv("RUN_EXTERNAL_PROVIDER_SMOKE") != "1":
        pytest.skip(f"Set RUN_EXTERNAL_PROVIDER_SMOKE=1 to run {provider_name} live smoke test.")


@pytest.mark.external
def test_jina_reranker_api_smoke():
    _require_external_smoke("Jina")
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        pytest.skip("JINA_API_KEY is not configured")

    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "jina-reranker-v3",
        "query": "ReAct reasoning actions",
        "top_n": 3,
        "documents": [
            "ReAct combines reasoning traces with task-specific actions.",
            "A document unrelated to scientific paper retrieval.",
        ],
        "return_documents": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
    response.raise_for_status()
    payload = response.json()
    assert "results" in payload


@pytest.mark.external
def test_openrouter_model_smoke():
    _require_external_smoke("OpenRouter")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY is not configured")

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        messages=[{"role": "user", "content": "Reply with OK only."}],
        temperature=0,
    )
    assert response.choices[0].message.content


@pytest.mark.external
def test_openai_model_smoke():
    _require_external_smoke("OpenAI")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not configured")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.4-nano"),
        messages=[{"role": "user", "content": "Reply with OK only."}],
        temperature=0,
    )
    assert response.choices[0].message.content


@pytest.mark.external
def test_mistral_ocr_client_available():
    _require_external_smoke("Mistral")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY is not configured")

    try:
        from mistralai import Mistral
    except Exception:
        from mistralai.client import Mistral

    client = Mistral(api_key=api_key)
    assert client is not None


if __name__ == '__main__':
    test_openai_model_smoke()
