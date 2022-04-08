import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from utils import contour_analysis
from utils.data import (list_files, load_image, one_hot_encode,
                        reset_class_values)
from utils.predict import color_classes


def run(
    images: str,
    output_contour_analysis: Optional[str] = "contour_analysis",
    classes: Optional[int] = None,
    record_id: Optional[str] = None,
    record_class: Optional[str] = None,
    current_time: Optional[str] = time.strftime('%Y%m%d%H%M%S'),
    copy_images: Optional[bool] = False,
    grayscale: Optional[bool] = False,
    measures_only: Optional[bool] = True) -> None:

    if isinstance(images, str):
        if Path(images).is_dir():
            files = list_files(images, as_numpy=True)
        elif Path(images).is_file():
            files = [images]
        else:
            raise FileNotFoundError(f"The directory or file was not found at `{images}`.")
    else:
        raise ValueError(f"`images` must be a `str`. Given `{type(images)}`.")

    classes_undefined = True if classes is None else False

    output_contour_analysis = Path(output_contour_analysis)
    output_contour_analysis.mkdir(exist_ok=True, parents=True)

    for file in tqdm(files, desc=record_id):
        mask = load_image(file, as_gray=True)

        if classes_undefined:
            classes = np.unique(mask).size

        mask = one_hot_encode(mask, classes=classes, as_numpy=True) * 127

        prediction, detail = contour_analysis.analyze_contours(mask=mask)
        prediction, parent_contours, child_contours = prediction
        detail, discarded_parent_contours, discarded_child_contours = detail

        parent_measurements, child_measurements = contour_analysis.get_contour_measurements(
            parent_contours=parent_contours,
            child_contours=child_contours,
            shape=mask.shape[:2],
            mask_name=Path(file).name,
            record_id=record_id,
            record_class=record_class)

        contour_analysis.write_contour_measurements(
            parent_measurements=parent_measurements,
            child_measurements=child_measurements,
            output_path=output_contour_analysis,
            datetime=current_time)

        if len(discarded_parent_contours) > 0 or len(discarded_child_contours) > 0:
            discarded_parent_measurements, discarded_child_measurements = contour_analysis.get_contour_measurements(
                parent_contours=discarded_parent_contours,
                child_contours=discarded_child_contours,
                shape=mask.shape[:2],
                mask_name=Path(file).name,
                record_id=record_id,
                record_class=record_class,
                start_index=len(parent_measurements),
                contours_flag="invalid")

            contour_analysis.write_contour_measurements(
                parent_measurements=discarded_parent_measurements,
                child_measurements=discarded_child_measurements,
                output_path=output_contour_analysis,
                datetime=current_time)

        if detail is not None and not measures_only:
            filtered_objects = output_contour_analysis.joinpath("filtered_objects")
            filtered_objects.mkdir(exist_ok=True, parents=True)

            cv2.imwrite(str(filtered_objects.joinpath(f"{Path(file).stem}_image_detail.png")), cv2.cvtColor(detail, cv2.COLOR_BGR2RGB))

            if copy_images:
                image = load_image(image_path=str(file), normalize=False, as_numpy=True, as_gray=True)
                image = one_hot_encode(image, classes=classes, as_numpy=True)
                if not grayscale:
                    image = color_classes(image)
                image = contour_analysis.draw_contour_lines(image, discarded_parent_contours)
                cv2.imwrite(str(filtered_objects.joinpath(f"{Path(file).stem}_image.png")), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not measures_only:
            if grayscale:
                prediction = reset_class_values(prediction)
            else:
                prediction = color_classes(prediction)
                prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)

            cv2.imwrite(str(output_contour_analysis.joinpath(f"{Path(file).stem}_prediction.png")), prediction)
            if copy_images:
                if grayscale:
                    shutil.copyfile(str(file), output_contour_analysis.joinpath(Path(file).name))
                else:
                    image = load_image(image_path=str(file), normalize=False, as_numpy=True, as_gray=True)
                    image = one_hot_encode(image, classes=classes, as_numpy=True)
                    image = color_classes(image)
                    cv2.imwrite(str(output_contour_analysis.joinpath(Path(file).name)), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def main():
    parser = argparse.ArgumentParser(description="Analyze the contours one or more segmentation masks.")

    parser.add_argument(
        "-i",
        "--images",
        help="A path to an image file, or a path to a directory containing images to be evaluated.",
        required=True,
        type=str)

    parser.add_argument(
        "-o",
        "--output",
        help="The path where to save the analysis result.",
        default="contour_analysis",
        type=str)

    parser.add_argument(
        "--classes",
        help="Number of classes. Affects the one hot encoding of the masks.",
        default=3,
        type=int)

    parser.add_argument(
        "--record-id",
        help="An ID that will identify the contour measurements.",
        default=None,
        type=str)

    parser.add_argument(
        "--record-class",
        help="The class the contour measurements belong to.",
        default=None,
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
        help="Whether or not to save predictions as gayscale masks.",
        default=False,
        action="store_true")

    parser.add_argument(
        "--measures-only",
        help="Do not save the predicted images or copy the input images to the output path. If `True`, it will override the effect of `output`.",
        default=False,
        action="store_true")

    parser.add_argument(
        "--multi-measurements",
        help="Performs the measurement of multiple records in a directory containing subdirectories of classes. The classes subdirectores should contain subdirectories with images.",
        default=False,
        action="store_true")

    parser.add_argument(
        "-gpu",
        "--gpu",
        help="What GPU to use. Pass `-1` to use CPU.",
        default=0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.multi_measurements:
        directories = [str(directory) for directory in Path(args.images).glob("*") if directory.is_dir()]

        for directory in directories:
            run(
                directory,
                output_contour_analysis=args.output,
                classes=args.classes,
                record_id=args.record_id,
                record_class=args.record_class,
                copy_images=args.copy_images,
                grayscale=args.grayscale,
                measures_only=args.measures_only
            )
    else:
        run(
            images=args.images,
            output_contour_analysis=args.output,
            classes=args.classes,
            record_id=args.record_id,
            record_class=args.record_class,
            copy_images=args.copy_images,
            grayscale=args.grayscale,
            measures_only=args.measures_only
        )


if __name__ == "__main__":
    main()
