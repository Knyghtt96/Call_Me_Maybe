"""Custom exceptions raised by the Call Me Maybe pipeline."""


class CallMeMaybeError(Exception):
    """Base class for every error raised by this project."""


class InputError(CallMeMaybeError):
    """Raised when an input file is missing, unreadable or invalid."""


class VocabError(CallMeMaybeError):
    """Raised when the model vocabulary cannot be loaded or decoded."""


class GenerationError(CallMeMaybeError):
    """Raised when constrained generation reaches a dead end."""
