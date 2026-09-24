"""Generation slots: one per kind of value the model has to fill.

The program writes the JSON structure itself and only leaves *slots* to
the model: the function name and each argument value. A slot knows which
tokens keep it valid at the current step, when it may be closed, and how
to turn the produced text into a typed Python value.
"""

import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .vocab import Vocabulary

DIGITS = "0123456789"


class Slot(BaseModel, ABC):
    """Common behaviour of every slot.

    Attributes:
        produced: Text generated so far.
        tokens: Identifiers accepted so far.
        max_tokens: Budget after which the slot is closed by force.
    """

    model_config = ConfigDict(extra="forbid")

    produced: str = ""
    tokens: list[int] = Field(default_factory=list)
    max_tokens: int = 32

    @abstractmethod
    def allowed_ids(self, vocab: Vocabulary) -> set[int]:
        """Return the tokens that may extend the value at this step.

        Args:
            vocab: Token tables.

        Returns:
            Identifiers of the legal tokens.
        """

    @abstractmethod
    def stop_ids(self, vocab: Vocabulary) -> set[int]:
        """Return the tokens that close the value.

        Args:
            vocab: Token tables.

        Returns:
            Identifiers of the closing tokens.
        """

    @abstractmethod
    def value(self) -> Any:
        """Return the typed Python value built from the produced text."""

    def banned_ids(self) -> set[int]:
        """Return tokens forbidden at this step even though they are legal.

        Returns:
            Identifiers to remove from the allowed set. Empty by default.
        """
        return set()

    def can_terminate(self) -> bool:
        """Tell whether the value produced so far is complete.

        Returns:
            ``True`` when a closing token may be accepted.
        """
        return True

    def exhausted(self) -> bool:
        """Tell whether the token budget is spent.

        Returns:
            ``True`` when no more token may be generated.
        """
        return len(self.tokens) >= self.max_tokens

    def accept(self, token_id: int, text: str) -> None:
        """Record a generated token.

        Args:
            token_id: Identifier of the token.
            text: Text of the token.
        """
        self.tokens.append(token_id)
        self.produced += text


class ChoiceSlot(Slot):
    """A value that must be exactly one of several candidates.

    Used for the function name: the model may only produce tokens that
    continue at least one declared name.

    Attributes:
        candidates: Acceptable values.
    """

    candidates: list[str]

    def allowed_ids(self, vocab: Vocabulary) -> set[int]:
        """Return the tokens that keep the text a prefix of a candidate.

        Args:
            vocab: Token tables.

        Returns:
            Identifiers of every token whose text, appended to the text
            produced so far, is still the beginning of a candidate.
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

    def stop_ids(self, vocab: Vocabulary) -> set[int]:
        """Return the tokens closing a quoted value.

        Args:
            vocab: Token tables.

        Returns:
            Identifiers of the tokens starting with a quote.
        """
        return vocab.quote_ids

    def can_terminate(self) -> bool:
        """Tell whether the produced text is a complete candidate.

        Returns:
            ``True`` when the text matches one of the candidates.
        """
        return self.produced in self.candidates

    def chosen(self) -> str:
        """Return the chosen candidate.

        When the budget ran out before a candidate was complete, the first
        candidate that starts with the produced text is used.

        Returns:
            One of the candidates.
        """
        if self.produced in self.candidates:
            return self.produced
        for candidate in self.candidates:
            if candidate.startswith(self.produced):
                return candidate
        return self.candidates[0]

    def value(self) -> Any:
        """Return the chosen candidate as the value of the slot.

        Returns:
            One of the candidates.
        """
        return self.chosen()


class BooleanSlot(ChoiceSlot):
    """A ``true`` or ``false`` literal, written without quotes.

    Attributes:
        candidates: Always the two JSON boolean literals.
    """

    candidates: list[str] = ["true", "false"]

    def stop_ids(self, vocab: Vocabulary) -> set[int]:
        """Return the tokens closing an unquoted value.

        Args:
            vocab: Token tables.

        Returns:
            Identifiers of the tokens starting with a comma or a brace.
        """
        return vocab.comma_ids | vocab.brace_ids

    def value(self) -> Any:
        """Return the boolean matching the produced literal.

        Returns:
            ``True`` for ``true``, ``False`` otherwise.
        """
        return self.chosen() == "true"


class NumberSlot(Slot):
    """A JSON number: optional minus sign, digits, optional decimal point.

    Qwen tokenises digits one by one, so the legal set is tiny. Leading
    zeros are refused because ``01`` is not valid JSON.

    Attributes:
        integer: When ``True`` the decimal point is forbidden.
    """

    integer: bool = False
    max_tokens: int = 24

    def allowed_ids(self, vocab: Vocabulary) -> set[int]:
        """Return the tokens that keep the text a valid number prefix.

        Args:
            vocab: Token tables.

        Returns:
            Identifiers of the legal digit and sign tokens.
        """
        legal = ""
        if self.produced == "":
            legal += "-"
        if self.produced not in ("0", "-0"):
            legal += DIGITS
        if (
            not self.integer
            and self._has_digit()
            and "." not in self.produced
        ):
            legal += "."
        return {
            vocab.text_to_id[char]
            for char in legal
            if char in vocab.text_to_id
        }

    def stop_ids(self, vocab: Vocabulary) -> set[int]:
        """Return the tokens closing an unquoted value.

        Args:
            vocab: Token tables.

        Returns:
            Identifiers of the tokens starting with a comma or a brace.
        """
        return vocab.comma_ids | vocab.brace_ids

    def can_terminate(self) -> bool:
        """Tell whether the produced text is a complete number.

        Returns:
            ``True`` once a digit exists and the text does not end on a
            decimal point.
        """
        return self._has_digit() and not self.produced.endswith(".")

    def value(self) -> int | float:
        """Return the produced number with the right Python type.

        Returns:
            An ``int`` when no decimal point was produced, a ``float``
            otherwise, or ``0`` if no digit was produced at all.
        """
        text = self.produced.rstrip(".")
        if not any(char in DIGITS for char in text):
            return 0
        if "." in text:
            return float(text)
        return int(text)

    def _has_digit(self) -> bool:
        """Tell whether at least one digit was produced.

        Returns:
            ``True`` when the text contains a digit.
        """
        return any(char in DIGITS for char in self.produced)


class StringSlot(Slot):
    """The content of a JSON string, without its quotes.

    Every accepted token keeps the string valid JSON: no control character,
    no unescaped quote and no backslash except in an escaped quote, so the
    text can always be parsed back into a Python string.

    Attributes:
        ngram: Length of the repeated sequences that are blocked.
    """

    ngram: int = 3

    def allowed_ids(self, vocab: Vocabulary) -> set[int]:
        """Return the tokens legal inside a string.

        The first token may not start with a space, so that a value comes
        out as ``shrek`` rather than `` shrek``.

        Args:
            vocab: Token tables.

        Returns:
            Identifiers of the legal tokens.
        """
        if self.produced == "":
            return vocab.string_start_ids
        return vocab.string_ids

    def stop_ids(self, vocab: Vocabulary) -> set[int]:
        """Return the tokens closing a quoted value.

        Args:
            vocab: Token tables.

        Returns:
            Identifiers of the tokens starting with a quote.
        """
        return vocab.quote_ids

    def banned_ids(self) -> set[int]:
        """Forbid the token that would repeat an n-gram already produced.

        Small models fall into loops such as ``$1 $2 $1 $2``. Any sequence
        of ``ngram`` tokens already present in the value cannot be produced
        again.

        Returns:
            Identifiers that would complete a repeated n-gram.
        """
        if len(self.tokens) < self.ngram:
            return set()
        prefix = tuple(self.tokens[-(self.ngram - 1):])
        banned: set[int] = set()
        for start in range(len(self.tokens) - self.ngram + 1):
            window = self.tokens[start:start + self.ngram]
            if tuple(window[:-1]) == prefix:
                banned.add(window[-1])
        return banned

    def value(self) -> str:
        """Return the produced text as a Python string.

        The text is JSON source, so an escaped quote is turned back into a
        plain quote by the JSON parser.

        Returns:
            The decoded string.
        """
        try:
            decoded = json.loads(f'"{self.produced}"')
        except json.JSONDecodeError:
            return self.produced
        return str(decoded)
