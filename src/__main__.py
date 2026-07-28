
import sys

from .cli import parse_args
from .io_utils import load_function_definitions, load_prompt_items


def main() -> int:
    """run the program."""
    args = parse_args()

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompt_items(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Call Me Maybe is running")
    print(f"Loaded {len(functions)} function definitions.")
    print(f"Loaded {len(prompts)} prompts")
    print(f"Output path: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
