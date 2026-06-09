from pathlib import Path

from rag.parser import parse_policy_markdown
from rag.vector_store import ChromaPolicyStore


def test_parse_policy_markdown_returns_h2_h3_chunks():
    markdown = Path("data/policy_mock_vi.md").read_text(encoding="utf-8")
    chunks = parse_policy_markdown(markdown)

    assert chunks
    assert all("section_h2" in chunk for chunk in chunks)
    assert all("section_h3" in chunk for chunk in chunks)
    assert all("citation" in chunk for chunk in chunks)
    assert all("rendered_text" in chunk for chunk in chunks)
    assert any("trả hàng" in chunk["rendered_text"].lower() for chunk in chunks)


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def test_chroma_policy_store_search_returns_citations(tmp_path):
    store = ChromaPolicyStore(tmp_path, FakeEmbeddings(), collection_name="test_policy")
    store.rebuild(Path("data/policy_mock_vi.md"))

    hits = store.search("hoan tra", top_k=2)

    assert len(hits) == 2
    assert "citation" in hits[0]
    assert "content" in hits[0]
    assert "distance" in hits[0]
