import argparse
import os

from utils.qualify_segmentation import qualify_segmentation


def main():
    parser = argparse.ArgumentParser(description="Evaluates the segmentation against the ground truth.")

    parser.add_argument(
        "-g",
        "--ground-truth",
        help="Path to the directory containing the ground truth.",
        required=True,
        type=str)

    parser.add_argument(
        "-p",
        "--predictions",
        help="Path to the directory containing the predictions.",
        required=True,
        type=str)

    parser.add_argument(
        "-o",
        "--output",
        help="The path where to save the qualification information.",
        default="qualification.csv",
        type=str)

    parser.add_argument(
        "-c",
        "--classes",
        help="Number of classes. Affects the one hot encoding of the masks.",
        default=3,
        type=int)

    parser.add_argument(
        "-v",
        "--visualization",
        help="Whether or not to create a visualization showing the differences in respect to the ground truth.",
        default=False,
        action="store_true")

    parser.add_argument(
        "-gpu",
        "--gpu",
        help="What GPU to use. Pass `-1` to use CPU.",
        default="0")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    qualify_segmentation(
        ground_truth_path=args.ground_truth,
        predictions_path=args.predictions,
        output_qualification=args.output,
        classes=args.classes,
        visualization=args.visualization
    )


if __name__ == "__main__":
    main()
