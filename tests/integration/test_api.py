"""
Integration tests — full HTTP round-trips against TestClient.

All external I/O (Anthropic, ChromaDB) is replaced with MagicMock fakes
defined in conftest.py, so these tests run without any API keys.
"""

from fastapi.testclient import TestClient


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body


class TestDocuments:
    def test_create_document(self, client: TestClient) -> None:
        payload = {"title": "Test Doc", "content": "Hello world " * 20}
        r = client.post("/api/v1/documents", json=payload)
        assert r.status_code == 201
        body = r.json()
        assert body["title"] == "Test Doc"
        assert "id" in body
        assert body["chunk_count"] >= 1

    def test_create_document_empty_content_rejected(self, client: TestClient) -> None:
        r = client.post("/api/v1/documents", json={"title": "T", "content": "   "})
        # FastAPI Pydantic validation rejects min_length=1 violated
        assert r.status_code in (422, 400)

    def test_list_documents_empty(self, client: TestClient) -> None:
        r = client.get("/api/v1/documents")
        assert r.status_code == 200
        body = r.json()
        assert "items" in body
        assert "total" in body

    def test_get_document_not_found(self, client: TestClient) -> None:
        r = client.get("/api/v1/documents/nonexistent-id")
        assert r.status_code == 404

    def test_create_then_get(self, client: TestClient) -> None:
        create_r = client.post(
            "/api/v1/documents",
            json={"title": "Article", "content": "Content " * 30, "source": "http://example.com"},
        )
        assert create_r.status_code == 201
        doc_id = create_r.json()["id"]

        get_r = client.get(f"/api/v1/documents/{doc_id}")
        assert get_r.status_code == 200
        assert get_r.json()["id"] == doc_id

    def test_delete_document(self, client: TestClient) -> None:
        create_r = client.post(
            "/api/v1/documents", json={"title": "To Delete", "content": "text " * 10}
        )
        doc_id = create_r.json()["id"]

        del_r = client.delete(f"/api/v1/documents/{doc_id}")
        assert del_r.status_code == 204

        get_r = client.get(f"/api/v1/documents/{doc_id}")
        assert get_r.status_code == 404

    def test_delete_nonexistent_document(self, client: TestClient) -> None:
        r = client.delete("/api/v1/documents/does-not-exist")
        assert r.status_code == 404


class TestQueries:
    def test_ask_returns_answer(self, client: TestClient) -> None:
        r = client.post("/api/v1/queries/ask", json={"question": "What is this about?"})
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert body["question"] == "What is this about?"
        assert isinstance(body["sources"], list)

    def test_ask_question_too_short_rejected(self, client: TestClient) -> None:
        r = client.post("/api/v1/queries/ask", json={"question": "Hi"})
        assert r.status_code == 422

    def test_summarize_not_found(self, client: TestClient) -> None:
        r = client.post("/api/v1/queries/summarize", json={"document_id": "missing"})
        assert r.status_code == 404
