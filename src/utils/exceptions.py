"""Custom exception classes for the API."""


class AgentZeroException(Exception):
    """Base exception for AgentZero API."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ModelLoadError(AgentZeroException):
    """Raised when model fails to load."""
    def __init__(self, message: str = "Failed to load model"):
        super().__init__(message, status_code=503)


class PredictionError(AgentZeroException):
    """Raised when prediction fails."""
    def __init__(self, message: str = "Prediction failed"):
        super().__init__(message, status_code=500)


class AuthenticationError(AgentZeroException):
    """Raised when authentication fails."""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class RateLimitError(AgentZeroException):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


class ValidationError(AgentZeroException):
    """Raised when input validation fails."""
    def __init__(self, message: str = "Validation failed"):
        super().__init__(message, status_code=400)