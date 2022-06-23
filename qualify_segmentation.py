import argparse
import os
from pathlib import Path

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
        "-c",
        "--classes",
        help="Number of classes. Affects the one hot encoding of the masks.",
        default=3,
        type=int)

    parser.add_argument(
        "-o",
        "--output",
        help="The path where to save the qualification information.",
        default="qualification.csv",
        type=str)

    parser.add_argument(
        "-v",
        "--visualization",
        help="Path where to save the visualization showing the differences in respect to the ground truth. Does not generate visualization if `None`.",
        default=None,
        type=str)

    parser.add_argument(
        "-bbox",
        "--bbox-annotations",
        help="Path to the `labelme` annotations containing bounding boxes for the nuclei to be considered.",
        type=str,
        required=False,
        default=None)

    parser.add_argument(
        "-gpu",
        "--gpu",
        help="What GPU to use. Pass the GPU id to use GPU.",
        default="-1")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.output == "qualification.csv":
        args.output = Path(args.predictions).joinpath(args.output)

    qualify_segmentation(
        ground_truth_path=args.ground_truth,
        predictions_path=args.predictions,
        classes=args.classes,
        output_qualification=args.output,
        output_visualization=args.visualization,
        bbox_annotations_path=args.bbox_annotations)


if __name__ == "__main__":
    main()
