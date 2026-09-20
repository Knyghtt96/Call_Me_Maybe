"""Model vocabulary loading and byte-level token decoding.

The vocabulary file maps a byte-level representation of each token to its
identifier. Constrained decoding needs the opposite direction as well as the
*real* text of every token, so both tables are built once at start-up.
"""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from .errors import VocabError

#: Characters that may never appear inside a raw JSON string value.
_FORBIDDEN_IN_STRING = {'"', "\\"}


def _bytes_to_unicode() -> dict[int, str]:
    """Build the byte-level encoder used by GPT-2 style tokenizers.

    Returns:
        A mapping from byte value to the printable character used to
        represent it inside the vocabulary file.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapping = {byte: chr(byte) for byte in printable}
    spare = 0
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(256 + spare)
            spare += 1
    return mapping


def _unicode_to_bytes() -> dict[str, int]:
    """Invert :func:`_bytes_to_unicode`.

    Returns:
        A mapping from printable character to the byte value it encodes.
    """
    return {char: byte for byte, char in _bytes_to_unicode().items()}


class Vocabulary(BaseModel):
    """Token tables required by the constrained decoder.

    Attributes:
        id_to_text: Real text produced by each decodable token.
        text_to_id: Reverse lookup, used to force literal segments.
        logits_size: Number of logits returned by the model.
        string_ids: Tokens allowed inside a JSON string value.
        string_ids_no_space: Same set, without tokens starting with a space.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    id_to_text: dict[int, str]
    text_to_id: dict[str, int]
    logits_size: int
    string_ids: np.ndarray
    string_ids_no_space: np.ndarray

    @classmethod
    def load(cls, path_str: str, logits_size: int) -> "Vocabulary":
        """Load a vocabulary file and build every lookup table.

        Args:
            path_str: Path to the ``vocab.json`` file of the model.
            logits_size: Length of the logit vector returned by the model.

        Returns:
            A fully built :class:`Vocabulary`.

        Raises:
            VocabError: If the file is missing or is not valid JSON.
        """
        path = Path(path_str)
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except OSError as exc:
            raise VocabError(f"Cannot read vocabulary file: {path}") from exc
        except json.JSONDecodeError as exc:
            raise VocabError(f"Invalid vocabulary file: {path}") from exc

        if not isinstance(raw, dict):
            raise VocabError("Vocabulary file must contain a JSON object.")

        decoder = _unicode_to_bytes()
        id_to_text: dict[int, str] = {}
        text_to_id: dict[str, int] = {}

        for symbol, token_id in raw.items():
            if not isinstance(token_id, int) or token_id >= logits_size:
                continue
            text = cls._decode_symbol(str(symbol), decoder)
            if text is None or text == "":
                continue
            id_to_text[token_id] = text
            # Keep the lowest id when several tokens share the same text.
            if text not in text_to_id or token_id < text_to_id[text]:
                text_to_id[text] = token_id

        if not id_to_text:
            raise VocabError("Vocabulary file contains no usable token.")

        string_ids = [
            token_id
            for token_id, text in id_to_text.items()
            if cls._is_string_safe(text)
        ]
        no_space = [
            token_id
            for token_id in string_ids
            if not id_to_text[token_id].startswith(" ")
        ]

        return cls(
            id_to_text=id_to_text,
            text_to_id=text_to_id,
            logits_size=logits_size,
            string_ids=np.asarray(sorted(string_ids), dtype=np.int64),
            string_ids_no_space=np.asarray(sorted(no_space), dtype=np.int64),
        )

    @staticmethod
    def _decode_symbol(symbol: str, decoder: dict[str, int]) -> str | None:
        """Turn a byte-level symbol into real text.

        Args:
            symbol: Token as stored in the vocabulary file.
            decoder: Mapping from printable character to byte value.

        Returns:
            The decoded text, or ``None`` when the token is only a fragment
            of a multi-byte character and cannot stand on its own.
        """
        try:
            payload = bytes(decoder[char] for char in symbol)
        except KeyError:
            return None
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            return None

    @staticmethod
    def _is_string_safe(text: str) -> bool:
        """Tell whether a token may appear inside a raw JSON string.

        Args:
            text: Real text of the token.

        Returns:
            ``True`` when the token needs no JSON escaping.
        """
        if any(char in _FORBIDDEN_IN_STRING for char in text):
            return False
        return all(ord(char) >= 0x20 for char in text)

    _prefix_cache: dict[str, set[int]] = PrivateAttr(default_factory=dict)

    def ids_starting_with(self, prefix: str) -> set[int]:
        """Return every token whose text begins with ``prefix``.

        A delimiter such as the closing quote seldom appears alone in the
        continuation a model favours: merged tokens like ``",`` or ``"}`` are
        far more likely. Accepting any token that *starts* with the delimiter
        is therefore required to let the model close a value at all.

        Args:
            prefix: Delimiter to look for.

        Returns:
            Identifiers of every matching token.
        """
        cached = self._prefix_cache.get(prefix)
        if cached is None:
            cached = {
                token_id
                for token_id, text in self.id_to_text.items()
                if text.startswith(prefix)
            }
            self._prefix_cache[prefix] = cached
        return cached

    def token_id(self, text: str) -> int:
        """Return the identifier of the token matching ``text`` exactly.

        Args:
            text: Text to look up.

        Returns:
            The token identifier.

        Raises:
            VocabError: If no token matches.
        """
        try:
            return self.text_to_id[text]
        except KeyError as exc:
            raise VocabError(f"No token for {text!r}") from exc

    def text_of(self, token_id: int) -> str:
        """Return the text produced by a token.

        Args:
            token_id: Identifier of the token.

        Returns:
            The decoded text of the token.

        Raises:
            VocabError: If the identifier is unknown.
        """
        try:
            return self.id_to_text[token_id]
        except KeyError as exc:
            raise VocabError(f"Unknown token id: {token_id}") from exc
