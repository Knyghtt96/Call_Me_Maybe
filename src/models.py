from pydantic import BaseModel, ConfigDict, field_validator
from typing import Any

ALLOWED_TYPES = {"string", "number", "boolean"}


class FunctionParameter(BaseModel):

    model_config = ConfigDict(extra="forbid")

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:

        if value not in ALLOWED_TYPES:
            raise ValueError(f"Unsupported parameter type: {value}")
        return value


class ReturnDefinition(BaseModel):

    model_config = ConfigDict(extra="forbid")

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:

        if value not in ALLOWED_TYPES:
            raise ValueError(f"Unsupported return type: {value}")
        return value


class FunctionDefinition(BaseModel):

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: ReturnDefinition


class PromptItem(BaseModel):

    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCallResult(BaseModel):

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]


class FunctionDefinitionList(BaseModel):

    model_config = ConfigDict(extra="forbid")

    items: list[FunctionDefinition]


class PromptItemList(BaseModel):

    model_config = ConfigDict(extra="forbid")

    items: list[PromptItem]
