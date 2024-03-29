import argparse
from pathlib import Path

from utils.utils import plot_metrics


def main():
    parser = argparse.ArgumentParser(description="Generate images of the model metrics.")

    parser.add_argument(
        "-m",
        "--metrics-file",
        help="Path to the model metrics file generated by the `model_train.py` script.",
        required=True,
        type=str)

    parser.add_argument(
        "-o",
        "--output",
        help="Where to save the resulting predictions.",
        type=str)

    parser.add_argument(
        "--figsize",
        help="Dimensions of the figures to be generated. Expects input shape in the following format: `HEIGHTxWIDTH`.",
        default="15x15",
        type=str)

    args = parser.parse_args()

    if not args.output:
        output = Path(args.metrics_file).parent
    else:
        output = args.output

    figsize = args.figsize.lower().split("x")
    figsize = (int(figsize[0]), int(figsize[1]))

    plot_metrics(
        args.metrics_file,
        output=output,
        figsize=figsize)


if __name__ == "__main__":
    main()
