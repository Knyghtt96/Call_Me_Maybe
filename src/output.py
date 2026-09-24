"""Writing the result file."""

import json
from pathlib import Path

from .errors import InputError
from .models import FunctionCallResult


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write the function calls as a JSON array.

    The parent directory is created when needed. Each object holds exactly
    the keys ``prompt``, ``name`` and ``parameters``.

    Args:
        path: Destination file.
        results: Function calls to write, in prompt order.

    Raises:
        InputError: If the file cannot be written.
    """
    payload = [result.model_dump() for result in results]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise InputError(f"Cannot write {path}: {exc.strerror}") from exc
