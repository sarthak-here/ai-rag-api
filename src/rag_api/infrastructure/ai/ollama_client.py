import httpx

from rag_api.core.config import Settings
from rag_api.core.exceptions import AIServiceError
from rag_api.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a precise document assistant. Answer questions using ONLY the context provided.
If the context does not contain enough information to answer, say so clearly.
Be concise, factual, and cite which context passages support your answer.\
"""


class OllamaClient:
    """Ollama-backed AI client for offline Llama models."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model

    def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return "No relevant context found to answer this question."

        context_block = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        user_message = f"<context>\n{context_block}\n</context>\n\nQuestion: {question}"

        return self._chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

    def summarize(self, text: str) -> str:
        return self._chat(
            [
                {
                    "role": "user",
                    "content": f"Summarize the following text in 3-5 sentences:\n\n{text}",
                }
            ]
        )

    def _chat(self, messages: list[dict[str, str]]) -> str:
        try:
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": messages, "stream": False},
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("ollama_http_error", error=str(exc))
            raise AIServiceError(f"Ollama request failed: {exc}") from exc

        data: dict[str, object] = response.json()
        message = data.get("message", {})
        if not isinstance(message, dict):
            raise AIServiceError("Unexpected response format from Ollama.")

        content = message.get("content", "")
        logger.info("ollama_answer_generated", model=self._model)
        return str(content)
