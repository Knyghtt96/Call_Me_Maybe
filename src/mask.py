"""Logit masking: the step where constrained decoding actually happens."""

from collections.abc import Iterable, Sequence

import numpy as np

from .errors import GenerationError
from .state import AllowedIds


def _as_index_array(allowed: AllowedIds, limit: int) -> np.ndarray:
    """Normalise an allowed-token collection into a bounded index array.

    Args:
        allowed: Token identifiers, as a set or a numpy array.
        limit: Number of available logits.

    Returns:
        A one-dimensional array of valid indices.
    """
    if isinstance(allowed, np.ndarray):
        index = allowed.astype(np.int64, copy=False)
    else:
        index = np.fromiter(allowed, dtype=np.int64, count=len(allowed))
    if index.size == 0:
        return index
    return index[(index >= 0) & (index < limit)]


def masked_argmax(
    logits: Sequence[float],
    allowed: AllowedIds,
    extra: Iterable[int] | None = None,
    banned: set[int] | None = None,
) -> int:
    """Select the best token among the allowed ones.

    Every logit outside ``allowed`` is replaced by negative infinity, which
    removes those tokens from the distribution before selection. The model
    still expresses its preference, but only inside the legal subset.

    Args:
        logits: Raw scores returned by the model for the next token.
        allowed: Identifiers of the tokens that keep the output valid.
        extra: Optional additional tokens that end the current slot.
        banned: Optional tokens removed from the allowed set.

    Returns:
        The identifier of the selected token.

    Raises:
        GenerationError: If no allowed token exists.
    """
    scores = np.asarray(logits, dtype=np.float32)
    index = _as_index_array(allowed, scores.size)
    if banned:
        index = index[~np.isin(index, np.fromiter(banned, dtype=np.int64))]
    if extra is not None:
        index = np.append(index, _as_index_array(set(extra), scores.size))
    if index.size == 0:
        raise GenerationError("No valid token available at this step.")
    masked = np.full(scores.shape, -np.inf, dtype=np.float32)
    masked[index] = scores[index]
    return int(np.argmax(masked))
