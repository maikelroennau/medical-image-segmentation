import argparse
import os

from utils.evaluate import evaluate, evaluate_from_files


def main():
    parser = argparse.ArgumentParser(description="Predicts using the indicated model on the indicated data.")

    parser.add_argument(
        "-m",
        "--model",
        help="Path to the model to evaluate. If path is a directory, will evaluate all models within it.",
        required=False,
        type=str)

    parser.add_argument(
        "-i",
        "--images",
        help="Path to the directory containing `images` and `masks` directories.",
        required=False,
        type=str)

    parser.add_argument(
        "-g",
        "--ground-truth",
        help="Path to the directory containing the ground truth.",
        required=False,
        type=str)

    parser.add_argument(
        "-p",
        "--predictions",
        help="Path to the directory containing the predictions.",
        required=False,
        type=str)

    parser.add_argument(
        "--input-shape",
        help="Whether to replace the model's input shape with the given input shape. Expects input shape in the following format: `HEIGHTxWIDTHxCHANNELS`.",
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
        help="Number of classes. Affects the one hot encoding of the masks.",
        default=3,
        type=int)

    parser.add_argument(
        "--ohe",
        help="Whether or not to convert masks to one hot encoded.",
        default=True,
        action="store_true")

    parser.add_argument(
        "-gpu",
        "--gpu",
        help="What GPU to use. Pass `-1` to use CPU.",
        default="0")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.input_shape:
        input_shape = args.input_shape.lower().split("x")
        input_shape = (int(input_shape[0]), int(input_shape[1]), int(input_shape[2]))
    else:
        input_shape = None

    if args.images is not None:
        evaluate(
            models_paths=args.model,
            images_path=args.images,
            batch_size=args.batch_size,
            classes=args.classes,
            one_hot_encoded=args.ohe,
            input_shape=input_shape
        )
    elif args.predictions is not None and args.ground_truth is not None:
        evaluate_from_files(
            ground_truth_path=args.ground_truth,
            predictions_path=args.predictions,
            classes=args.classes
        )


if __name__ == "__main__":
    main()
