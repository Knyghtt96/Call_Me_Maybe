from src.cli import parse_args


def main() -> None:
    args = parse_args()
    print("Call Me Maybe is running.")
    print(args)


if __name__ == "__main__":
    main()
