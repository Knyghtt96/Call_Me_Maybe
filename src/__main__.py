"""Entry point: load the inputs, generate every call, write the results."""

import sys
import time

from llm_sdk import Small_LLM_Model

from .cli import Arguments, parse_args
from .errors import CallMeMaybeError
from .generator import Generator
from .io_utils import load_function_definitions, load_prompts
from .models import FunctionCallResult
from .output import write_results
from .vocab import Vocabulary


def run(args: Arguments) -> int:
    """Run the whole pipeline.

    Args:
        args: Parsed command line arguments.

    Returns:
        ``0`` on success, ``1`` when an error was reported.
    """
    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompts(args.input)
    except CallMeMaybeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(functions)} function definitions.")
    print(f"Loaded {len(prompts)} prompts.", flush=True)

    try:
        # "cpu" keeps the SDK from requesting device_map="auto", which
        # would need the optional accelerate package on CUDA machines.
        llm = Small_LLM_Model(device="cpu")
        vocab = Vocabulary.build(llm)
    except CallMeMaybeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # the model loader can raise anything
        print(f"Error: cannot load the model: {exc}", file=sys.stderr)
        return 1
    print(
        f"Vocabulary ready: {len(vocab.id_to_text)} usable tokens.",
        flush=True,
    )

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
        results.append(result)
        elapsed = time.monotonic() - started
        print(
            f"  [{position}] {result.name} "
            f"{result.parameters} ({elapsed:.1f}s)",
            flush=True,
        )

    try:
        write_results(args.output, results)
    except CallMeMaybeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(results)} results to {args.output}.")
    return 1 if failures else 0


def main() -> int:
    """Parse the command line and run, never letting an exception escape.

    Returns:
        The exit code of the program.
    """
    try:
        return run(parse_args())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # last line of defence: report, never crash
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
