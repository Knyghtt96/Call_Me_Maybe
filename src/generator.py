"""Constrained generation of function calls.

The JSON skeleton is written by this module, never by the model. The model is
only asked to fill two kinds of holes: which function to call, and what each
argument is worth. Both are produced token by token under a mask, so the
result is valid JSON by construction.
"""

import json
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, PrivateAttr

from .errors import GenerationError
from .mask import masked_argmax
from .models import FunctionCallResult, FunctionDefinition
from .prompt import build_prompt
from .state import ChoiceSlot, Slot, slot_for_type
from .vocab import Vocabulary


@runtime_checkable
class LanguageModel(Protocol):
    """Minimal interface required from the LLM SDK."""

    def encode(self, text: str) -> Any:
        """Encode text into token identifiers."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return the next-token logits for a token sequence."""

    def get_path_to_vocab_file(self) -> str:
        """Return the path to the vocabulary file."""


def _flatten(raw: Any) -> list[int]:
    """Turn an encoder output into a flat list of token identifiers.

    Args:
        raw: Value returned by the SDK encoder.

    Returns:
        The token identifiers as a flat list.

    Raises:
        GenerationError: If the value cannot be interpreted.
    """
    data = raw.tolist() if hasattr(raw, "tolist") else raw
    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, list) or not all(
        isinstance(item, int) for item in data
    ):
        raise GenerationError("Unexpected token encoding from the SDK.")
    return [int(item) for item in data]


class Generator(BaseModel):
    """Translate natural language prompts into structured function calls.

    Attributes:
        llm: Language model wrapper provided by the SDK.
        vocab: Token tables used to build the masks.
        functions: Every callable function.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    llm: LanguageModel
    vocab: Vocabulary
    functions: list[FunctionDefinition]

    _end_of_turn: int | None = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        """Resolve the end-of-turn token of the chat format.

        An instruction-tuned model signals the end of its answer with a
        dedicated token. Accepting it as a stop signal lets the model decide
        that a value is complete, instead of forcing it to spell out the
        closing quote.

        Args:
            context: Validation context, unused.
        """
        try:
            ids = self._encode("<|im_end|>")
        except GenerationError:
            return
        self._end_of_turn = ids[0] if len(ids) == 1 else None

    def generate(self, user_prompt: str) -> FunctionCallResult:
        """Produce the function call matching a natural language prompt.

        Args:
            user_prompt: Natural language request.

        Returns:
            The validated function call.

        Raises:
            GenerationError: If no function is defined.
        """
        if not self.functions:
            raise GenerationError("No function definition available.")

        text = build_prompt(self.functions, user_prompt)
        ids = self._encode(text)

        answer = '{"name": "'
        ids += self._encode(answer)

        names = [function.name for function in self.functions]
        name_slot = ChoiceSlot(candidates=names)
        name = self._fill(ids, name_slot, '"')
        answer += name

        function = self._function_named(name)
        segment = '", "parameters": {'
        answer += segment
        ids += self._encode(segment)

        answer += self._fill_parameters(ids, function)
        return self._validate(user_prompt, answer)

    def _fill_parameters(
        self,
        ids: list[int],
        function: FunctionDefinition,
    ) -> str:
        """Generate every argument of a function.

        Args:
            ids: Token sequence built so far, extended in place.
            function: Definition of the selected function.

        Returns:
            The JSON text of the arguments, closing braces included.
        """
        items = list(function.parameters.items())
        if not items:
            closing = "}}"
            ids += self._encode(closing)
            return closing

        answer = ""
        last = len(items) - 1
        for position, (key, param) in enumerate(items):
            opening = f'"{key}": '
            if param.type == "string":
                opening += '"'
            answer += opening
            ids += self._encode(opening)

            terminator = '"' if param.type == "string" else (
                "}" if position == last else ","
            )
            answer += self._fill(ids, slot_for_type(param.type), terminator)

            closing = '"' if param.type == "string" else ""
            closing += "}}" if position == last else ", "
            answer += closing
            ids += self._encode(closing)
        return answer

    def _fill(self, ids: list[int], slot: Slot, terminator: str) -> str:
        """Generate one slot under a mask.

        Args:
            ids: Token sequence built so far, extended in place.
            slot: State describing what is legal at each step.
            terminator: Text whose first token ends the slot.

        Returns:
            The text produced for this slot.
        """
        stops = set(self.vocab.ids_starting_with(terminator))
        if self._end_of_turn is not None:
            stops.add(self._end_of_turn)
        while not slot.exhausted():
            extra = stops if slot.can_terminate() else None
            try:
                token = masked_argmax(
                    self.llm.get_logits_from_input_ids(ids),
                    slot.allowed_ids(self.vocab),
                    extra,
                    slot.banned_ids(),
                )
            except GenerationError:
                break
            if token in stops and slot.can_terminate():
                return slot.produced
            slot.accept(token, self.vocab.text_of(token))
            ids.append(token)
        return slot.fallback()

    def _function_named(self, name: str) -> FunctionDefinition:
        """Look up a function by name.

        Args:
            name: Name produced by the decoder.

        Returns:
            The matching definition.

        Raises:
            GenerationError: If the name is unknown.
        """
        for function in self.functions:
            if function.name == name:
                return function
        raise GenerationError(f"Generated unknown function name: {name}")

    def _validate(self, user_prompt: str, answer: str) -> FunctionCallResult:
        """Parse and validate the generated JSON.

        Args:
            user_prompt: Original request.
            answer: Generated JSON object.

        Returns:
            The validated result.

        Raises:
            GenerationError: If the JSON is unusable.
        """
        try:
            payload = json.loads(answer)
        except json.JSONDecodeError as exc:
            raise GenerationError(f"Generated invalid JSON: {answer}") from exc

        name = payload.get("name", "")
        function = self._function_named(str(name))
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise GenerationError("Generated parameters are not an object.")

        typed: dict[str, Any] = {}
        for key, param in function.parameters.items():
            value = parameters.get(key)
            typed[key] = (
                float(value)
                if param.type == "number" and isinstance(value, (int, float))
                else value
            )
        return FunctionCallResult(
            prompt=user_prompt,
            name=function.name,
            parameters=typed,
        )

    def _encode(self, text: str) -> list[int]:
        """Encode a literal segment of the answer.

        Args:
            text: Text to encode.

        Returns:
            The token identifiers of that text.
        """
        return _flatten(self.llm.encode(text))
