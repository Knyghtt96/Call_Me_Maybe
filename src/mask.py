"""Constrained token selection.

This is where constrained decoding happens: the logits of every token that
would break the structure are replaced by minus infinity, and the best
remaining token is selected.
"""

from collections.abc import Sequence

import numpy as np

from .errors import GenerationError


def masked_argmax(logits: Sequence[float], allowed: set[int]) -> int:
    """Select the best token among the allowed ones.

    Args:
        logits: Raw scores returned by the model, one per token.
        allowed: Identifiers of the tokens that keep the output valid.

    Returns:
        The identifier of the allowed token with the highest score.

    Raises:
        GenerationError: If no allowed token exists in the logit vector.
    """
    scores = np.asarray(logits, dtype=np.float32)
    index = np.fromiter(allowed, dtype=np.int64, count=len(allowed))
    index = index[(index >= 0) & (index < scores.size)]
    if index.size == 0:
        raise GenerationError("No token is allowed at this step.")
    masked = np.full(scores.shape, -np.inf, dtype=np.float32)
    masked[index] = scores[index]
    return int(np.argmax(masked))
