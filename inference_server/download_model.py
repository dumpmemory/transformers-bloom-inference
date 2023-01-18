import argparse

from inference_server.models import get_hf_model_class


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model to use",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        required=True,
        help="model class to use",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = get_args()
    get_hf_model_class(args.model_class).from_pretrained(args.model_name)


if __name__ == "__main__":
    main()
