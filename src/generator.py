"""Constrained generation of one function call.

The generator writes the JSON skeleton itself, token by token, and asks
the model to fill only the slots: the function name, then each argument
value. At every step the logits are masked so that only tokens keeping the
call valid can be selected.
"""

from typing import Any

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict

from .errors import GenerationError
from .mask import masked_argmax
from .models import FunctionCallResult, FunctionDefinition
from .prompt import build_prompt
from .state import BooleanSlot, ChoiceSlot, NumberSlot, Slot, StringSlot
from .vocab import Vocabulary


def slot_for_type(type_name: str) -> Slot:
    """Create the slot matching a declared parameter type.

    Args:
        type_name: One of the supported type names.

    Returns:
        A fresh slot.

    Raises:
        GenerationError: If the type is not supported.
    """
    if type_name == "string":
        return StringSlot()
    if type_name == "number":
        return NumberSlot()
    if type_name == "integer":
        return NumberSlot(integer=True)
    if type_name == "boolean":
        return BooleanSlot()
    raise GenerationError(f"Unsupported parameter type: {type_name!r}")


class Generator(BaseModel):
    """Turn a natural language request into a validated function call.

    Attributes:
        llm: Loaded model.
        vocab: Token tables built from the model.
        functions: Callable functions, in file order.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    llm: Small_LLM_Model
    vocab: Vocabulary
    functions: list[FunctionDefinition]

    def generate(self, user_prompt: str) -> FunctionCallResult:
        """Generate the function call answering ``user_prompt``.

        Args:
            user_prompt: Natural language request.

        Returns:
            The validated function call.

        Raises:
            GenerationError: If the model returns unusable logits.
        """
        ids = self._encode(build_prompt(self.functions, user_prompt))

        ids += self._encode('{"name": "')
        names = [function.name for function in self.functions]
        name = str(self._fill(ids, ChoiceSlot(candidates=names)))
        function = self._function_named(name)
        ids += self._encode('", "parameters": {')

        parameters: dict[str, Any] = {}
        items = list(function.parameters.items())
        for position, (key, param) in enumerate(items):
            quote = '"' if param.type == "string" else ""
            ids += self._encode(f'"{key}": {quote}')
            parameters[key] = self._fill(ids, slot_for_type(param.type))
            closing = "}}" if position == len(items) - 1 else ", "
            ids += self._encode(quote + closing)
        if not items:
            ids += self._encode("}}")

        return FunctionCallResult(
            prompt=user_prompt,
            name=name,
            parameters=parameters,
        )

    def _fill(self, ids: list[int], slot: Slot) -> Any:
        """Let the model fill one slot under constraint.

        Args:
            ids: Token identifiers of the text generated so far. Accepted
                tokens are appended in place.
            slot: Slot to fill.

        Returns:
            The typed value produced by the slot.

        Raises:
            GenerationError: If the model returns unusable logits.
        """
        while not slot.exhausted():
            allowed = slot.allowed_ids(self.vocab) - slot.banned_ids()
            stops: set[int] = set()
            if slot.can_terminate():
                stops = slot.stop_ids(self.vocab)
            if not allowed and not stops:
                break
            logits = self.llm.get_logits_from_input_ids(ids)
            token_id = masked_argmax(logits, allowed | stops)
            if token_id in stops:
                break
            slot.accept(token_id, self.vocab.text_of(token_id))
            ids.append(token_id)
        return slot.value()

    def _function_named(self, name: str) -> FunctionDefinition:
        """Return the declared function called ``name``.

        Args:
            name: Function name produced by the model.

        Returns:
            The matching definition.

        Raises:
            GenerationError: If no function has that name.
        """
        for function in self.functions:
            if function.name == name:
                return function
        raise GenerationError(f"Unknown function name: {name!r}")

    def _encode(self, text: str) -> list[int]:
        """Tokenise a piece of text with the SDK.

        Args:
            text: Text to tokenise.

        Returns:
            The token identifiers as plain integers.
        """
        tensor = self.llm.encode(text)
        return [int(token_id) for token_id in tensor.tolist()[0]]
