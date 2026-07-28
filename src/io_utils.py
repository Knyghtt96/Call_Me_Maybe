"""Utilities for readind and writing JSON files."""

import json
from pathlib import Path

from pydantic import ValidationError

from .models import FunctionDefinition, FunctionDefinitionList
from .models import PromptItem, PromptItemList


def load_json_file(path_str: str) -> object:
    """Load raw JSON content from a file path."""
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {path}") from e


def load_function_definitions(path_str: str) -> list[FunctionDefinition]:
    """Load and validate function definitions."""
    raw_data = load_json_file(path_str)

    if not isinstance(raw_data, list):
        raise ValueError("function_definitions.json must"
                         "contain a JSON array.")

    try:
        validated = FunctionDefinitionList.model_validate({"items": raw_data})
    except ValidationError as e:
        raise ValueError(f"Invalid function definitions: {e}") from e

    return validated.items


def load_prompt_items(path_str: str) -> list[PromptItem]:
    """Load and validate prompt items."""
    raw_data = load_json_file(path_str)

    if not isinstance(raw_data, list):
        raise ValueError("function_calling_tests.json"
                         "must contain a JSON array.")

    try:
        validated = PromptItemList.model_validate({"items": raw_data})
    except ValidationError as e:
        raise ValueError(f"Invalid prompt items: {e}") from e

    return validated.items
