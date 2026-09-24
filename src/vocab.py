"""Token table of the model.

Constrained decoding needs to know, for every token identifier, the text
the token produces. ``vocab.json``, reached through
``get_path_to_vocab_file()``, lists the identifiers of the real tokens, and
the ``decode`` method of the SDK gives the text of each one. Every token is
classified once at start-up, so that a generation step only has to look up
ready-made sets.
"""

import json
from pathlib import Path

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict

from .errors import InputError

# JSON knows several escape sequences, but only the escaped quote is
# accepted inside a value. A request never needs a control character, and
# letting the model write an escaped backslash was measured harmful: it
# invents regex escapes that are not in the request. A quote, on the other
# hand, cannot appear in a value any other way.


def is_json_string_text(text: str) -> bool:
    """Tell whether ``text`` can be appended inside a JSON string as is.

    The text must not contain a control character, must not contain an
    unescaped quote (it would close the string) and every backslash must
    be followed by a quote (the only escape accepted), so that the string
    stays valid JSON after the token is added.

    Args:
        text: Text of one token.

    Returns:
        ``True`` when the token keeps the JSON string valid.
    """
    escaped = False
    for char in text:
        if escaped:
            if char != '"':
                return False
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"' or ord(char) < 0x20:
            return False
    return not escaped


class Vocabulary(BaseModel):
    """Token tables used by the constrained decoder.

    Attributes:
        id_to_text: Text produced by each usable token.
        text_to_id: Identifier producing a given text (lowest one wins).
        string_ids: Tokens legal inside a JSON string.
        string_start_ids: Same set, minus tokens starting with a space.
        quote_ids: Tokens starting with ``"``, which close a string.
        comma_ids: Tokens starting with ``,``, which end an argument.
        brace_ids: Tokens starting with ``}``, which end the call.
    """

    model_config = ConfigDict(extra="forbid")

    id_to_text: dict[int, str]
    text_to_id: dict[str, int]
    string_ids: set[int]
    string_start_ids: set[int]
    quote_ids: set[int]
    comma_ids: set[int]
    brace_ids: set[int]

    @classmethod
    def build(cls, llm: Small_LLM_Model) -> "Vocabulary":
        """Decode every token of the model and classify it.

        Args:
            llm: Loaded model, used for ``get_path_to_vocab_file`` and
                ``decode``.

        Returns:
            The token tables.

        Raises:
            InputError: If the vocabulary file cannot be read.
        """
        id_to_text: dict[int, str] = {}
        text_to_id: dict[str, int] = {}
        for token_id in cls._token_ids(llm):
            text: str = llm.decode([token_id])
            # Special tokens decode to nothing; tokens holding a fragment
            # of a multi-byte character decode to U+FFFD. Neither can be
            # written inside a value.
            if text == "" or "\ufffd" in text:
                continue
            id_to_text[token_id] = text
            if text not in text_to_id:
                text_to_id[text] = token_id
        if not id_to_text:
            raise InputError("The model vocabulary contains no usable token.")

        string_ids = {
            token_id
            for token_id, text in id_to_text.items()
            if is_json_string_text(text)
        }
        return cls(
            id_to_text=id_to_text,
            text_to_id=text_to_id,
            string_ids=string_ids,
            string_start_ids={
                token_id
                for token_id in string_ids
                if not id_to_text[token_id].startswith(" ")
            },
            quote_ids=cls._starting_with(id_to_text, '"'),
            comma_ids=cls._starting_with(id_to_text, ","),
            brace_ids=cls._starting_with(id_to_text, "}"),
        )

    @staticmethod
    def _token_ids(llm: Small_LLM_Model) -> list[int]:
        """Read the token identifiers listed in ``vocab.json``.

        Args:
            llm: Loaded model.

        Returns:
            Every identifier found in the file, in increasing order.

        Raises:
            InputError: If the file is missing or malformed.
        """
        try:
            path = Path(llm.get_path_to_vocab_file())
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError) as exc:
            raise InputError(
                f"Cannot read the vocabulary file: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise InputError("The vocabulary file must be a JSON object.")
        return sorted(
            token_id for token_id in raw.values()
            if isinstance(token_id, int) and token_id >= 0
        )

    @staticmethod
    def _starting_with(id_to_text: dict[int, str], prefix: str) -> set[int]:
        """Collect the tokens whose text starts with ``prefix``.

        A delimiter such as the closing quote seldom appears alone in the
        continuation a model prefers: merged tokens like ``",`` or ``"}``
        are far more likely. Accepting every token that *starts* with the
        delimiter is therefore what lets the model close a value.

        Args:
            id_to_text: Token table.
            prefix: Delimiter to look for.

        Returns:
            Identifiers of every matching token.
        """
        return {
            token_id
            for token_id, text in id_to_text.items()
            if text.startswith(prefix)
        }

    def text_of(self, token_id: int) -> str:
        """Return the text produced by a token.

        Args:
            token_id: Token identifier.

        Returns:
            The decoded text, or an empty string for an unusable token.
        """
        return self.id_to_text.get(token_id, "")
