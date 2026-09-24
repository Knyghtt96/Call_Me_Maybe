"""Command line arguments."""

import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


class Arguments(BaseModel):
    """Validated command line arguments.

    Attributes:
        functions_definition: Path of the function definition file.
        input: Path of the prompt file.
        output: Path of the result file.
    """

    model_config = ConfigDict(extra="forbid")

    functions_definition: Path = DEFAULT_FUNCTIONS
    input: Path = DEFAULT_INPUT
    output: Path = DEFAULT_OUTPUT


def parse_args(argv: list[str] | None = None) -> Arguments:
    """Parse the command line.

    Args:
        argv: Arguments to parse, or ``None`` to use ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural language prompts into function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        help="JSON file listing the callable functions",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="JSON file listing the prompts to process",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON file to write the function calls to",
    )
    namespace = parser.parse_args(argv)
    return Arguments(
        functions_definition=namespace.functions_definition,
        input=namespace.input,
        output=namespace.output,
    )
