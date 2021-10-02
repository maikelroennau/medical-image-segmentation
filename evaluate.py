import argparse
import os

import tensorflow as tf

from utils.utils import evaluate


def main():
    parser = argparse.ArgumentParser(description="Predicts using the indicated model on the indicated data.")

    parser.add_argument(
        "-m",
        "--model",
        help="Path to the model to evaluate. If path is a directory, will evaluate all models within it.",
        required=True,
        type=str)

    parser.add_argument(
        "-i",
        "--images",
        help="Path to the directory containing the images to predict or to a single image file.",
        default="dataset/test/",
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
        "--classes",
        help="Number of classes.",
        default=1,
        type=int)

    parser.add_argument(
        "--ohe",
        help="Wheter or not to convert masks to one-hot-encoded.",
        default=False,
        action="store_true")

    parser.add_argument(
        "-gpu",
        "--gpu",
        help="What GPU to use. Pass `-1` to use CPU.",
        default=0)

    parser.add_argument(
        "--seed",
        help="Seed for reproducibility.",
        default=1145,
        type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tf.random.set_seed(args.seed)

    if args.input_shape:
        input_shape = args.input_shape.lower().split("x")
        input_shape = (int(input_shape[0]), int(input_shape[1]), (3))
    else:
        input_shape = None

    evaluate(
        model=args.model,
        images_path=args.images,
        batch_size=args.batch_size,
        input_shape=input_shape,
        classes=args.classes,
        one_hot_encoded=args.ohe)


if __name__ == "__main__":
    main()
