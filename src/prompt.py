"""Prompt construction.

The prompt carries semantics only: which functions exist, and what the user
asked. The JSON structure is never requested from the model, it is imposed by
the decoder. Two choices matter here:

* the chat format of the model is reproduced explicitly, because an
  instruction-tuned model follows a request far better inside the turn layout
  it was trained on;
* a single example is shown, built on functions that do not exist in the
  input file. It teaches the answer *shape*, never the answers themselves.
"""

from .models import FunctionDefinition

_SYSTEM = (
    "<|im_start|>system\n"
    "You are a function calling engine. You select exactly one function and "
    "extract each argument from the request. Arguments are literal and as "
    "short as possible: a replacement is plain text, never a pattern. "
    "Never repeat yourself.<|im_end|>\n"
)

_EXAMPLE = (
    "<|im_start|>user\n"
    "Available functions:\n"
    "- fn_count_chars(text: string) -> number: Count the characters of a "
    "string.\n"
    "- fn_replace_pattern(haystack: string, pattern: string, value: string)"
    " -> string: Replace every regular expression match in a string.\n"
    "\nRequest: Replace every digit in 'abc123' with #<|im_end|>\n"
    "<|im_start|>assistant\n"
    '{"name": "fn_replace_pattern", "parameters": {"haystack": "abc123", '
    '"pattern": "[0-9]", "value": "#"}}<|im_end|>\n'
)

_REQUEST = (
    "<|im_start|>user\n"
    "Available functions:\n"
    "{functions}"
    "\nRequest: {prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def describe_function(function: FunctionDefinition) -> str:
    """Render one function as a single readable line.

    Args:
        function: Definition loaded from the input file.

    Returns:
        A one-line description of the function.
    """
    params = ", ".join(
        f"{name}: {param.type}"
        for name, param in function.parameters.items()
    )
    return (
        f"- {function.name}({params}) -> {function.returns.type}: "
        f"{function.description}\n"
    )


def build_prompt(
    functions: list[FunctionDefinition],
    user_prompt: str,
) -> str:
    """Build the full prompt sent to the model.

    Args:
        functions: Every callable function.
        user_prompt: Natural language request.

    Returns:
        The prompt text, ending right before the generated answer.
    """
    body = "".join(describe_function(function) for function in functions)
    return _SYSTEM + _EXAMPLE + _REQUEST.format(
        functions=body,
        prompt=user_prompt,
    )
