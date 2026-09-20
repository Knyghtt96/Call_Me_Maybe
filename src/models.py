"""Pydantic models describing every input and output structure."""

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

ALLOWED_TYPES = {"string", "number", "boolean"}


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FunctionParameter(StrictBaseModel):
    """A single parameter of a callable function."""

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Reject any type the subject does not allow.

        Args:
            value: Declared type name.

        Returns:
            The validated type name.

        Raises:
            ValueError: If the type is not supported.
        """
        if value not in ALLOWED_TYPES:
            raise ValueError(f"Unsupported parameter type: {value}")
        return value


class ReturnDefinition(StrictBaseModel):
    """Return type of a callable function."""

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Reject any return type the subject does not allow.

        Args:
            value: Declared type name.

        Returns:
            The validated type name.

        Raises:
            ValueError: If the type is not supported.
        """
        if value not in ALLOWED_TYPES:
            raise ValueError(f"Unsupported return type: {value}")
        return value


class FunctionDefinition(StrictBaseModel):
    """A function the model may decide to call."""

    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: ReturnDefinition


class PromptItem(StrictBaseModel):
    """A natural language request read from the input file."""

    prompt: str


class FunctionCallResult(StrictBaseModel):
    """One entry of the results file."""

    prompt: str
    name: str
    parameters: dict[str, Any]


class FunctionDefinitionList(StrictBaseModel):
    """Wrapper validating a JSON array of function definitions."""

    items: list[FunctionDefinition]


class PromptItemList(StrictBaseModel):
    """Wrapper validating a JSON array of prompts."""

    items: list[PromptItem]
