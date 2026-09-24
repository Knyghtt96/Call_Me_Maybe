"""Reading and validating the two input files."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .errors import InputError
from .models import FunctionDefinition, PromptItem


def load_json_array(path: Path) -> list[Any]:
    """Read a file that must contain a JSON array.

    Args:
        path: File to read.

    Returns:
        The parsed array.

    Raises:
        InputError: If the file is missing, unreadable, not valid JSON or
            not a JSON array.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise InputError(f"File not found: {path}") from exc
    except OSError as exc:
        raise InputError(f"Cannot read {path}: {exc.strerror}") from exc
    except UnicodeDecodeError as exc:
        raise InputError(f"{path} is not valid UTF-8 text.") from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            f"{path} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from exc
    if not isinstance(data, list):
        raise InputError(f"{path} must contain a JSON array.")
    return data


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load and validate the function definition file.

    Args:
        path: Path of ``functions_definition.json``.

    Returns:
        The declared functions, in file order.

    Raises:
        InputError: If the file is invalid or declares no function.
    """
    functions: list[FunctionDefinition] = []
    for position, entry in enumerate(load_json_array(path), start=1):
        try:
            functions.append(FunctionDefinition.model_validate(entry))
        except ValidationError as exc:
            raise InputError(
                f"{path}: function #{position} is invalid: "
                f"{_first_error(exc)}"
            ) from exc
    if not functions:
        raise InputError(f"{path} declares no function.")
    names = [function.name for function in functions]
    if len(set(names)) != len(names):
        raise InputError(f"{path} declares the same function name twice.")
    return functions


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate the prompt file.

    Args:
        path: Path of ``function_calling_tests.json``.

    Returns:
        The prompts, in file order.

    Raises:
        InputError: If the file is invalid.
    """
    prompts: list[PromptItem] = []
    for position, entry in enumerate(load_json_array(path), start=1):
        try:
            prompts.append(PromptItem.model_validate(entry))
        except ValidationError as exc:
            raise InputError(
                f"{path}: prompt #{position} is invalid: "
                f"{_first_error(exc)}"
            ) from exc
    return prompts


def _first_error(exc: ValidationError) -> str:
    """Summarise a pydantic error in one readable line.

    Args:
        exc: The validation error.

    Returns:
        The location and message of the first reported problem.
    """
    errors = exc.errors()
    if not errors:
        return "invalid entry"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "invalid value"))
    return f"{location}: {message}" if location else message
