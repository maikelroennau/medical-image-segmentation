import argparse
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from pathlib import Path

from utils.data import SUPPORTED_IMAGE_TYPES
from utils.predict import predict


def main():
    parser = argparse.ArgumentParser(description="Predicts using the indicated model on the indicated data.")

    parser.add_argument(
        "-m",
        "--model",
        help="The path to the model to be used to perform the prediction(s).",
        required=True,
        type=str)

    parser.add_argument(
        "-i",
        "--images",
        help="A path to an image file, or a path to a directory containing images, or a path to a directory containing subdirectories of classes.",
        required=True,
        type=str)

    parser.add_argument(
        "-n",
        "--normalize",
        help="Whether or not to put the image values between zero and one ([0,1]).",
        default=True,
        action="store_true")

    parser.add_argument(
        "--input-shape",
        help="The input shape the loaded model and images should have, in format `(HEIGHT, WIDTH, CHANNELS)`. If `model` is a `tf.keras.model` with an input shape different from `input_shape`, then its input shape will be changed to `input_shape`.",
        required=False,
        type=str)

    parser.add_argument(
        "-c",
        "--copy-images",
        help="Whether or not to copy the input images to the predictions output directory.",
        default=False,
        action="store_true")

    parser.add_argument(
        "-g",
        "--grayscale",
        help="Whether or not to save predictions as grayscale masks.",
        default=False,
        action="store_true")

    parser.add_argument(
        "-a",
        "--analyze-contours",
        help="Whether or not to apply the contour analysis algorithm. If `True`, it will also write the contour measurements to a `.csv` file.",
        default=False,
        action="store_true")

    parser.add_argument(
        "-o",
        "--output",
        help="The path where to save the predicted segmentation masks.",
        default="predictions",
        type=str)

    parser.add_argument(
        "--analysis-output",
        help="The path where to save the `.csv` file containing the contour measurements. Only effective if `analyze_contour` is `True`.",
        required=False,
        type=str)

    parser.add_argument(
        "--record-id",
        help="An ID that will identify the contour measurements.",
        required=False,
        type=str)

    parser.add_argument(
        "--record-class",
        help="The class the contour measurements belong to.",
        required=False,
        type=str)

    parser.add_argument(
        "-bboxes",
        "--bboxes",
        help="The path to a directory containing `labelme` annotations with bounding boxes indicating objects to be considered.",
        required=False,
        type=str)

    parser.add_argument(
        "--classify-agnor",
        help="Whether or not to classify AgNORs in `cluster` or `satellite`.",
        default=False,
        action="store_true")

    parser.add_argument(
        "--decision-tree-model",
        help="The path to the decision tree model to be used to classify AgNORs. Only effective if `--classify-agnor` is `True`.",
        required=False,
        type=str)

    parser.add_argument(
        "--measures-only",
        help="Do not save the predicted images or copy the input images to the output path. If `True`, it will override the effect of `output`.",
        default=False,
        action="store_true")

    parser.add_argument(
        "--multi-measurements",
        help="Performs the measurement of multiple records in a directory containing subdirectories of classes. The classes subdirectories should contain subdirectories with images.",
        default=False,
        action="store_true")

    parser.add_argument(
        "-gpu",
        "--gpu",
        help="What GPU to use. Pass `-1` to use CPU.",
        default=0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.input_shape:
        input_shape = args.input_shape.lower().split("x")
        input_shape = (int(input_shape[0]), int(input_shape[1]), int(input_shape[2]))
    else:
        input_shape = None

    if args.output:
        output = args.output
    elif not args.output and Path(args.model).is_file():
        output = Path(args.images)
    elif not args.output and Path(args.model).is_dir():
        output = Path(args.model).joinpath("predictions")

    if args.multi_measurements:
        directories = set([
            file.parent
            for file in Path(args.images).rglob("*")
            if file.is_file() and file.suffix in SUPPORTED_IMAGE_TYPES
        ])

        for directory in directories:
            predict(
                model=args.model,
                images=str(directory),
                normalize=args.normalize,
                input_shape=input_shape,
                copy_images=args.copy_images,
                grayscale=args.grayscale,
                analyze_contours=args.analyze_contours,
                output_predictions=str(Path(output).joinpath(directory.parent.name).joinpath(directory.name)),
                output_contour_analysis=output,
                record_id=directory.name,
                record_class=directory.parent.name,
                bboxes=args.bboxes,
                classify_agnor=args.classify_agnor,
                decision_tree_model_path=args.decision_tree_model_path,
                measures_only=args.measures_only,
            )
    else:
        predict(
            model=args.model,
            images=args.images,
            normalize=args.normalize,
            input_shape=input_shape,
            copy_images=args.copy_images,
            grayscale=args.grayscale,
            analyze_contours=args.analyze_contours,
            output_predictions=output,
            output_contour_analysis=args.analysis_output,
            record_id=args.record_id,
            record_class=args.record_class,
            bboxes=args.bboxes,
            classify_agnor=args.classify_agnor,
            decision_tree_model_path=args.decision_tree_model,
            measures_only=args.measures_only,
        )


if __name__ == "__main__":
    main()
