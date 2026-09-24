"""Exceptions raised by the program.

Every error that the program knows how to explain derives from
:class:`CallMeMaybeError`, so the entry point can catch one type, print a
clear message and exit without a traceback.
"""


class CallMeMaybeError(Exception):
    """Base class for every error reported to the user."""


class InputError(CallMeMaybeError):
    """A file cannot be read, parsed or does not match the expected shape."""


class GenerationError(CallMeMaybeError):
    """The model or the decoder could not produce a function call."""
