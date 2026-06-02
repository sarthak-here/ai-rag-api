import anthropic

from rag_api.core.config import Settings
from rag_api.core.exceptions import AIServiceError
from rag_api.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a precise document assistant. Answer questions using ONLY the context provided.
If the context does not contain enough information to answer, say so clearly.
Be concise, factual, and cite which context passages support your answer.\
"""


class AIClient:
    """Thin wrapper around the Anthropic SDK with structured RAG generation."""

    def __init__(self, settings: Settings) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    def generate_answer(
        self,
        question: str,
        context_chunks: list[str],
    ) -> str:
        """Generate a grounded answer using retrieved context chunks."""
        if not context_chunks:
            return "No relevant context found to answer this question."

        context_block = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        user_message = f"<context>\n{context_block}\n</context>\n\nQuestion: {question}"

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIError as exc:
            logger.error("anthropic_api_error", error=str(exc))
            raise AIServiceError(f"Anthropic API error: {exc}") from exc

        content = response.content[0]
        if content.type != "text":
            raise AIServiceError("Unexpected response type from Anthropic API.")

        logger.info(
            "answer_generated",
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return content.text

    def summarize(self, text: str) -> str:
        """Return a concise summary of the provided text."""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize the following text in 3-5 sentences:\n\n{text}",
                    }
                ],
            )
        except anthropic.APIError as exc:
            logger.error("anthropic_summarize_error", error=str(exc))
            raise AIServiceError(f"Anthropic API error: {exc}") from exc

        content = response.content[0]
        if content.type != "text":
            raise AIServiceError("Unexpected response type from Anthropic API.")
        return content.text
