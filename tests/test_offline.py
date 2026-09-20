"""Offline structural test of the constrained decoder.

No model weights are needed: a mock model returns pseudo-random logits.
Whatever the model says, the decoder must still produce a valid call, which
is exactly the property the subject asks for.

Run with: uv run python tests/test_offline.py
"""

import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generator import Generator  # noqa: E402
from src.io_utils import (  # noqa: E402
    load_function_definitions,
    load_prompt_items,
)
from src.vocab import Vocabulary  # noqa: E402

LOGITS_SIZE = 151936


def find_vocab() -> str:
    """Locate a cached vocab.json without loading the model.

    Returns:
        Path to the vocabulary file.

    Raises:
        SystemExit: If no cached vocabulary is found.
    """
    root = Path.home() / ".cache" / "huggingface" / "hub"
    for candidate in root.glob("models--Qwen--*/snapshots/*/vocab.json"):
        return str(candidate)
    raise SystemExit("No cached vocab.json found: run make run once.")


class MockModel:
    """Longest-match encoder plus pseudo-random logits."""

    def __init__(self, vocab: Vocabulary, seed: int) -> None:
        """Store the vocabulary and seed the generator."""
        self._vocab = vocab
        self._rng = random.Random(seed)

    def encode(self, text: str) -> list[list[int]]:
        """Encode text with a greedy longest-match strategy."""
        ids: list[int] = []
        position = 0
        while position < len(text):
            for size in range(min(16, len(text) - position), 0, -1):
                chunk = text[position:position + size]
                token = self._vocab.text_to_id.get(chunk)
                if token is not None:
                    ids.append(token)
                    position += size
                    break
            else:
                position += 1
        return [ids]

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return random logits, ignoring the context."""
        return [self._rng.uniform(-5.0, 5.0) for _ in range(LOGITS_SIZE)]

    def get_path_to_vocab_file(self) -> str:
        """Return the vocabulary path."""
        return find_vocab()


def main() -> int:
    """Check that every generated call is structurally valid.

    Returns:
        ``0`` when every check passes.
    """
    vocab = Vocabulary.load(find_vocab(), LOGITS_SIZE)
    functions = load_function_definitions(
        "data/input/functions_definition.json"
    )
    prompts = load_prompt_items("data/input/function_calling_tests.json")
    expected: dict[str, type] = {
        "number": float,
        "string": str,
        "boolean": bool,
    }

    for seed in range(3):
        generator = Generator(
            llm=MockModel(vocab, seed),
            vocab=vocab,
            functions=functions,
        )
        for item in prompts:
            result = generator.generate(item.prompt)
            payload: dict[str, Any] = json.loads(result.model_dump_json())
            function = next(
                f for f in functions if f.name == payload["name"]
            )
            assert set(payload["parameters"]) == set(function.parameters)
            for key, param in function.parameters.items():
                value = payload["parameters"][key]
                assert isinstance(value, expected[param.type]), (key, value)
        print(f"seed {seed}: {len(prompts)}/{len(prompts)} valid calls")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
