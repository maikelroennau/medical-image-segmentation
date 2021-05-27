import argparse
import os
from pathlib import Path

from utils import predict


def main():
    parser = argparse.ArgumentParser(description="Predicts using the indicated model on the indicated data.")

    parser.add_argument(
        "-m",
        "--model",
        help="Path to the model to use for prediction.",
        required=True,
        type=str)

    parser.add_argument(
        "-i",
        "--images",
        help="Path to the directory containing the images to predict or to a single image file.",
        default="dataset/test/images/",
        type=str)

    parser.add_argument(
        "-o",
        "--output",
        help="Where to save the resulting predictions.",
        type=str)

    parser.add_argument(
        "--input-shape",
        help="Whether to replace the model's input shape with the given input shape. Expects input shape in the following format: `HEIGHTxWIDTH`.",
        type=str)

    parser.add_argument(
        "-b",
        "--batch-size",
        help="Batch size during evaluation.",
        default=1,
        type=int)

    parser.add_argument(
        "-c",
        "--copy-images",
        help="Copy predicted images to output directory. Only effective when `--output` is provided.",
        default=True,
        action="store_true")

    parser.add_argument(
        "-gpu",
        "--gpu",
        help="What GPU to use. Pass `-1` to use CPU.",
        default=0)

    parser.add_argument(
        "--memory-growth",
        help="Whether or not to allow GPU memory growth.",
        default=False,
        action="store_true")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = str(args.memory_growth).lower()

    if args.input_shape:
        input_shape = args.input_shape.lower().split("x")
        input_shape = (int(input_shape[0]), int(input_shape[1]), (3))
    else:
        input_shape = None

    if not args.output and Path(args.images).is_file():
        output = Path(args.images).parent
    elif not args.output and not Path(args.images).is_file():
        output = Path(args.images)
    else:
        output = args.output

    if args.output != output:
        args.copy_images = False

    predict(args.model, args.images, args.batch_size, output, args.copy_images, input_shape)


if __name__ == "__main__":
    main()
