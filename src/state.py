"""Generation states used by the constrained decoder.

Each state (a "slot") answers a single question: given what has already been
produced, which tokens may legally come next? The decoder never parses the
output afterwards -- validity is guaranteed while generating.
"""

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, ConfigDict

from .vocab import Vocabulary

#: Either an explicit set of token ids or a pre-computed numpy array.
AllowedIds = set[int] | np.ndarray

_DIGITS = "0123456789"

#: Size of the n-gram that may not be repeated inside a string value.
_NO_REPEAT_NGRAM = 3


class Slot(BaseModel, ABC):
    """Base class for a constrained generation state.

    Attributes:
        produced: Text generated so far for this slot.
        tokens: Identifiers of the tokens accepted by this slot.
        max_tokens: Hard limit protecting against runaway generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    produced: str = ""
    tokens: list[int] = []
    max_tokens: int = 32

    @abstractmethod
    def allowed_ids(self, vocab: Vocabulary) -> AllowedIds:
        """Return the tokens that may extend this slot.

        Args:
            vocab: Vocabulary of the model.

        Returns:
            The identifiers of every legal continuation token.
        """

    @abstractmethod
    def can_terminate(self) -> bool:
        """Tell whether the slot currently holds a complete value.

        Returns:
            ``True`` when generation may stop here.
        """

    @abstractmethod
    def fallback(self) -> str:
        """Return a safe value used when generation reaches a dead end.

        Returns:
            A value that always satisfies the schema.
        """

    def banned_ids(self) -> set[int]:
        """Return tokens forbidden on top of the allowed set.

        Returns:
            Identifiers that must be masked out at this step.
        """
        return set()

    def accept(self, token_id: int, text: str) -> None:
        """Record a chosen token.

        Args:
            token_id: Identifier of the accepted token.
            text: Real text of the accepted token.
        """
        self.produced += text
        self.tokens.append(token_id)

    def exhausted(self) -> bool:
        """Tell whether the token budget of this slot is spent.

        Returns:
            ``True`` when no further token may be generated.
        """
        return len(self.tokens) >= self.max_tokens


class ChoiceSlot(Slot):
    """Slot restricted to one value out of a closed list.

    Used for function names and for booleans: the model decides, but it can
    only ever spell out one of the candidates.

    Attributes:
        candidates: Every acceptable complete value.
    """

    candidates: list[str]

    def allowed_ids(self, vocab: Vocabulary) -> AllowedIds:
        """Return tokens that extend at least one remaining candidate.

        Args:
            vocab: Vocabulary of the model.

        Returns:
            Identifiers of every legal continuation token.
        """
        allowed: set[int] = set()
        for candidate in self.candidates:
            if not candidate.startswith(self.produced):
                continue
            remainder = candidate[len(self.produced):]
            for size in range(1, len(remainder) + 1):
                token_id = vocab.text_to_id.get(remainder[:size])
                if token_id is not None:
                    allowed.add(token_id)
        return allowed

    def can_terminate(self) -> bool:
        """Tell whether the produced text is already a full candidate.

        Returns:
            ``True`` when the value is complete.
        """
        return self.produced in self.candidates

    def fallback(self) -> str:
        """Return the shortest candidate still compatible.

        Returns:
            A complete candidate value.
        """
        matching = [
            candidate
            for candidate in self.candidates
            if candidate.startswith(self.produced)
        ]
        pool = matching or self.candidates
        return min(pool, key=len)


class NumberSlot(Slot):
    """Slot producing a JSON number.

    Qwen tokenises digits one by one, so the legal set is tiny: the ten
    digits, an optional leading minus sign and at most one decimal point.
    Leading zeros are rejected to stay strictly JSON compliant.
    """

    max_tokens: int = 16

    def allowed_ids(self, vocab: Vocabulary) -> AllowedIds:
        """Return the digits and punctuation legal at this position.

        Args:
            vocab: Vocabulary of the model.

        Returns:
            Identifiers of every legal continuation token.
        """
        allowed: set[int] = set()
        digits = self._digits_allowed()
        for digit in _DIGITS if digits else "":
            token_id = vocab.text_to_id.get(digit)
            if token_id is not None:
                allowed.add(token_id)
        if self.produced == "":
            minus = vocab.text_to_id.get("-")
            if minus is not None:
                allowed.add(minus)
        if self._point_allowed():
            point = vocab.text_to_id.get(".")
            if point is not None:
                allowed.add(point)
        return allowed

    def _has_digit(self) -> bool:
        """Tell whether at least one digit was produced.

        Returns:
            ``True`` when the value already holds a digit.
        """
        return any(char in _DIGITS for char in self.produced)

    def _digits_allowed(self) -> bool:
        """Tell whether another digit may be appended.

        Returns:
            ``False`` when it would create a leading zero.
        """
        return self.produced not in ("0", "-0")

    def _point_allowed(self) -> bool:
        """Tell whether a decimal point may be appended.

        Returns:
            ``True`` when the point keeps the number valid.
        """
        return self._has_digit() and "." not in self.produced

    def can_terminate(self) -> bool:
        """Tell whether the number is complete.

        Returns:
            ``True`` when at least one digit was produced and the value does
            not end with a decimal point.
        """
        return self._has_digit() and not self.produced.endswith(".")

    def fallback(self) -> str:
        """Return a valid number when generation fails.

        Returns:
            The produced value if usable, ``"0"`` otherwise.
        """
        return self.produced if self.can_terminate() else "0"


class StringSlot(Slot):
    """Slot producing the body of a JSON string.

    Every token containing a quote, a backslash or a control character is
    masked out, so the produced text never needs escaping and the closing
    quote can only appear where the decoder allows it.
    """

    max_tokens: int = 32

    def banned_ids(self) -> set[int]:
        """Forbid tokens that would repeat an already produced n-gram.

        Small models fall into loops such as ``$1 $2 $3 $1 $2 $3``. Blocking
        any repeated n-gram inside the same value stops that drift without
        constraining the content itself.

        Returns:
            Identifiers that would close a repeated n-gram.
        """
        window = _NO_REPEAT_NGRAM - 1
        if len(self.tokens) < window:
            return set()
        prefix = tuple(self.tokens[-window:])
        banned: set[int] = set()
        for start in range(len(self.tokens) - window):
            if tuple(self.tokens[start:start + window]) == prefix:
                banned.add(self.tokens[start + window])
        return banned

    def allowed_ids(self, vocab: Vocabulary) -> AllowedIds:
        """Return the tokens that are safe inside a JSON string.

        Args:
            vocab: Vocabulary of the model.

        Returns:
            Identifiers of every legal continuation token.
        """
        if self.produced == "":
            return vocab.string_ids_no_space
        return vocab.string_ids

    def can_terminate(self) -> bool:
        """Tell whether the string may be closed.

        Returns:
            ``True`` once at least one token was produced.
        """
        return self.produced != ""

    def fallback(self) -> str:
        """Return the text produced so far.

        Returns:
            The current value, possibly empty.
        """
        return self.produced


def slot_for_type(type_name: str) -> Slot:
    """Build the slot matching a parameter type.

    Args:
        type_name: Type declared in ``functions_definition.json``.

    Returns:
        A slot enforcing that type.

    Raises:
        ValueError: If the type is not supported.
    """
    if type_name == "number":
        return NumberSlot()
    if type_name == "string":
        return StringSlot()
    if type_name == "boolean":
        return ChoiceSlot(candidates=["true", "false"])
    raise ValueError(f"Unsupported parameter type: {type_name}")
