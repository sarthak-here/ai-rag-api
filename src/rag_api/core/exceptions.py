from http import HTTPStatus


class AppError(Exception):
    """Base application error — carries an HTTP status and a human-readable detail."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class NotFoundError(AppError):
    status_code = HTTPStatus.NOT_FOUND
    detail = "The requested resource was not found."


class ConflictError(AppError):
    status_code = HTTPStatus.CONFLICT
    detail = "Resource already exists."


class ValidationError(AppError):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    detail = "Validation failed."


class AIServiceError(AppError):
    status_code = HTTPStatus.BAD_GATEWAY
    detail = "AI service returned an unexpected response."


class DocumentTooLargeError(ValidationError):
    detail = "Document exceeds the maximum allowed size."


class EmptyDocumentError(ValidationError):
    detail = "Document content must not be empty."
