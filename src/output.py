"""Writing of the results file."""

import json
from pathlib import Path

from .errors import InputError
from .models import FunctionCallResult


def write_results(path_str: str, results: list[FunctionCallResult]) -> None:
    """Write the results as a JSON array.

    Args:
        path_str: Destination path, created with its parent directories.
        results: One entry per processed prompt.

    Raises:
        InputError: If the file cannot be written.
    """
    path = Path(path_str)
    payload = [
        {
            "prompt": result.prompt,
            "name": result.name,
            "parameters": result.parameters,
        }
        for result in results
    ]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise InputError(f"Cannot write results to {path}") from exc
