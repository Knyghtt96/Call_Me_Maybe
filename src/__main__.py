"""Entry point: load the inputs, generate every call, write the results."""

import sys
import time

from llm_sdk import Small_LLM_Model

from .cli import parse_args
from .errors import CallMeMaybeError
from .generator import Generator
from .io_utils import load_function_definitions, load_prompt_items
from .models import FunctionCallResult
from .output import write_results
from .vocab import Vocabulary

#: Number of logits returned by Qwen3 models, including padding slots.
_PROBE_TEXT = "probe"


def main() -> int:
    """Run the whole pipeline.

    Returns:
        ``0`` on success, ``1`` when an error was reported.
    """
    args = parse_args()

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompt_items(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(functions)} function definitions.")
    print(f"Loaded {len(prompts)} prompts.")

    try:
        llm = Small_LLM_Model(device="cpu")
        probe = llm.get_logits_from_input_ids(
            [int(token) for token in llm.encode(_PROBE_TEXT).tolist()[0]]
        )
        vocab = Vocabulary.load(llm.get_path_to_vocab_file(), len(probe))
    except (CallMeMaybeError, OSError, ValueError) as exc:
        print(f"Error: cannot initialise the model: {exc}", file=sys.stderr)
        return 1

    print(f"Vocabulary ready: {len(vocab.id_to_text)} decodable tokens.")

    generator = Generator(llm=llm, vocab=vocab, functions=functions)
    results: list[FunctionCallResult] = []
    failures = 0

    for position, item in enumerate(prompts, start=1):
        started = time.monotonic()
        try:
            result = generator.generate(item.prompt)
        except CallMeMaybeError as exc:
            failures += 1
            print(f"  [{position}] failed: {exc}", file=sys.stderr)
            continue
        elapsed = time.monotonic() - started
        results.append(result)
        print(
            f"  [{position}] {result.name} "
            f"{result.parameters} ({elapsed:.1f}s)"
        )

    try:
        write_results(args.output, results)
    except CallMeMaybeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)} results to {args.output}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
