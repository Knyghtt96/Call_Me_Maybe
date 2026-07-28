import argparse
from argparse import Namespace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="call-me-maybe",
        description="Translate prompts into structured function calls.",
        # usage="uv run python -m src"
        add_help=True  # enable/disable the -h command, default = True
    )
    parser.add_argument(
                        "--functions_definition",
                        default="data/input/functions_definition.json",
                        help="Path to the functions definition JSON file."
                    )
    parser.add_argument(
                        "--input",
                        default="data/input/function_calling_tests.json",
                        help="Path to the input prompts file."
                    )
    parser.add_argument(
                        "--output",
                        default="data/output/function_calling_results.json",
                        help="Path to the output prompts file."
                    )
    return parser


def parse_args() -> Namespace:
    parser = build_parser()
    return parser.parse_args()
