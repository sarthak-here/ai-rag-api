from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from rag_api.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    content: str
    distance: float


class VectorStore:
    """ChromaDB-backed vector store for semantic chunk retrieval."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vector_store_ready", collection=collection_name, persist_dir=persist_dir)

    def add_chunks(
        self,
        chunk_ids: list[str],
        contents: list[str],
        document_id: str,
    ) -> None:
        if not chunk_ids:
            return
        self._collection.add(
            ids=chunk_ids,
            documents=contents,
            metadatas=[{"document_id": document_id, "chunk_id": cid} for cid in chunk_ids],
        )
        logger.debug("chunks_indexed", count=len(chunk_ids), document_id=document_id)

    def delete_document(self, document_id: str) -> None:
        results = self._collection.get(where={"document_id": document_id})
        ids: list[str] = results.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
            logger.debug("chunks_deleted", count=len(ids), document_id=document_id)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count() or 1),
        )
        hits: list[SearchResult] = []
        ids_list = results.get("ids") or [[]]
        docs_list = results.get("documents") or [[]]
        metas_list = results.get("metadatas") or [[]]
        distances_list = results.get("distances") or [[]]

        for cid, content, meta, dist in zip(
            ids_list[0], docs_list[0], metas_list[0], distances_list[0], strict=False
        ):
            hits.append(
                SearchResult(
                    chunk_id=cid,
                    document_id=str(meta.get("document_id", "")),
                    content=content,
                    distance=float(dist),
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()
