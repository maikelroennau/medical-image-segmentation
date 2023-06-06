import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import imgviz
import labelme
import tifffile
from skimage.io import imsave
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.utils import get_color_map, one_hot_encoded_to_rgb


def parse_labels_file(labels_file: str, filter_labels: bool) -> Tuple[List, dict]:
    """Parse the labels file into the required format to be used with the `Labelme` API.

    Args:
        labels_file (str): Path to the labels file.
        filter_labels (bool): A list of labels to be ignored.

    Raises:
        FileNotFoundError: If the labels file path does not point to a existing file.

    Returns:
        Tuple[Tuple, dict]: A tuple where the first element is another tuple containing the classes names. The second element is a `dict` mapping the class name to an ID.
    """
    if Path(labels_file).is_file():
        if filter_labels:
            lines = [line.strip() for line in open(labels_file).readlines() if line.strip() not in filter_labels]
        else:
            lines = open(labels_file).readlines()

        class_names = []
        class_name_to_id = {}
        for i, line in enumerate(lines):
            class_id = i - 1  # starts with -1
            class_name = line.strip()
            class_name_to_id[class_name] = class_id

            if class_id == -1:
                assert class_name == "__ignore__"
                continue
            elif class_id == 0:
                assert class_name == "_background_"

            class_names.append(class_name)
        class_names = tuple(class_names)

        return class_names, class_name_to_id
    else:
        raise FileNotFoundError(f"No file was found at `{labels_file}`.")


def filter_shapes(label_file: labelme.LabelFile) -> labelme.LabelFile:
    """Filter shapes that do not meet the minimum points criteria.

    Args:
        label_file (labelme.LabelFile): The label file containing the shapes to be evaluated.

    Returns:
        labelme.LabelFile: The label file with the filtered shapes.
    """
    shapes = []
    for shape in label_file.shapes:
        if shape["shape_type"] == "polygon" and len(shape["points"]) > 2:
            shapes.append(shape)
        elif shape["shape_type"] == "circle" and len(shape["points"]) == 2:
            shapes.append(shape)
    label_file.shapes = shapes
    return label_file


def merge_classes(label_file: labelme.LabelFile, stand_class: str, mergin_class: str) -> labelme.LabelFile:
    """Merges the shapes of a class into another shape class.

    Args:
        label_file (labelme.LabelFile): The label file containing the shape classes to be evaluated.
        stand_class (str): The class that will receive the shapes from `mergin_class`.
        mergin_class (str): The class that will be merged into `stand_class`.

    Returns:
        labelme.LabelFile: The label file with the updated shape classes.
    """
    shapes = []
    for shape in label_file.shapes:
        if shape["label"] == mergin_class:
            shape["label"] = stand_class
        shapes.append(shape)
    label_file.shapes = shapes
    return label_file


def sort_shapes(label_file: labelme.LabelFile, classes_order: list) -> labelme.LabelFile:
    """Sort shapes accordingly to the given order.

    Args:
        label_file (labelme.LabelFile): The label file containing the shapes to be sorted.
        classes_order (list): The order the shapes must be sorted into.

    Returns:
        labelme.LabelFile: The label file with the sorted shapes
    """
    shapes = {}
    for shape in label_file.shapes:
        if shape["label"] not in shapes.keys():
            shapes[shape["label"]] = []
        shapes[shape["label"]].append(shape)

    sorted_shapes = []
    for class_name in classes_order:
        if class_name in shapes.keys():
            sorted_shapes.extend(shapes[class_name])

    label_file.shapes = sorted_shapes
    return label_file


def convert_annotations_to_masks(
    input_dir: str,
    output_dir: str,
    labels: str,
    filter_labels: Optional[list] = None,
    filter_invalid: Optional[bool] = False,
    save_as_tif: Optional[bool] = False,
    overlay: Optional[bool] = False,
    color: Optional[bool] = False) -> None:
    """Converts `Labelme` annotations (Pascal VOC) into images and labels.

    Args:
        input_dir (str): The path to the images and annotation files.
        output_dir (str): The path where to save the converted images and labels.
        labels (str): The path to a `.txt`. file containing annotated classes. It must contain one class per row and start with `__ignore__`, followed by `_background_`.
        filter_labels (Optional[list], optional): A `list` of labels to ignore. Defaults to None.
        filter_invalid (Optional[bool], optional): Whether or not to ignore images annotated as invalid. Defaults to False.
        save_as_tif (Optional[bool], optional): Whether or not to save images in the `.tif` format. If `False`, images will be saved as `.png`. Defaults to False.
        overlay (Optional[bool], optional): Whether or not to generate a annotation overlay. Defaults to False.
        color (Optional[bool], optional): Whether or not to save masks as colored images. Defaults to False.

    Raises:
        FileNotFoundError: If the input directory is not found.
    """
    input_dir = Path(input_dir)

    if input_dir.is_dir():
        annotations = [annotation for annotation in input_dir.glob("*.json")]
        class_names, class_name_to_id = parse_labels_file(labels, filter_labels)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images_dir = output_dir.joinpath("images")
        images_dir.mkdir(parents=True, exist_ok=True)

        masks_dir = output_dir.joinpath("masks")
        masks_dir.mkdir(parents=True, exist_ok=True)

        if overlay:
            overlay_dir = output_dir.joinpath("overlay")
            overlay_dir.mkdir(parents=True, exist_ok=True)

        for annotation in tqdm(annotations):
            if filter_invalid:
                with open(str(annotation), "r", encoding="utf-8") as annotation_file:
                    annotation_file = json.load(annotation_file)
                    if "invalidated" in annotation_file.keys():
                        if annotation_file["invalidated"]:
                            continue

            label_file = labelme.LabelFile(filename=annotation)

            if filter_labels:
                label_file.shapes = [shape for shape in label_file.shapes if shape["label"] in class_names]

            image_type = ".tif" if save_as_tif else ".png"
            image_file_path = str(images_dir.joinpath(annotation.stem + image_type))
            mask_file_path = str(masks_dir.joinpath(annotation.stem + "_mask.png"))

            image = labelme.utils.img_data_to_arr(label_file.imageData)
            if save_as_tif:
                tifffile.imwrite(image_file_path, image, photometric="rgb")
            else:
                cv2.imwrite(image_file_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            label_file = filter_shapes(label_file)
            # label_file = merge_classes(label_file, "nucleus", "discarded_nucleus")
            # label_file = merge_classes(label_file, "nor", "discarded_nor")
            label_file = sort_shapes(label_file, classes_order=class_names)
            # class_names = tuple(class_name for class_name in class_names if class_name not in ["discarded_nucleus", "discarded_nor"])

            mask, _ = labelme.utils.shapes_to_label(
                img_shape=image.shape,
                shapes=label_file.shapes,
                label_name_to_value=class_name_to_id)

            if color:
                colored_mask = one_hot_encoded_to_rgb(mask.copy())
                imsave(mask_file_path, colored_mask, check_contrast=False)
            else:
                cv2.imwrite(mask_file_path, mask)

            if len(class_names) > 4:
                color_map = get_color_map(colormap="papanicolaou")
            else:
                color_map = get_color_map(colormap="agnor")

            if overlay:
                overlay_file_path = str(overlay_dir.joinpath(annotation.stem + "_overlay.png"))
                overlay_mask = imgviz.label2rgb(
                    label=mask,
                    image=image,
                    font_size=20,
                    label_names=class_names,
                    loc="rb",
                    colormap=color_map)
                cv2.imwrite(overlay_file_path, cv2.cvtColor(overlay_mask, cv2.COLOR_BGR2RGB))
    else:
        raise FileNotFoundError(f"No directory was found at `{input_dir}`.")


def main():
    parser = argparse.ArgumentParser(description="Converts labelme files into multiple resolution segmentation masks in the VOC format.")

    parser.add_argument(
        "-i",
        "--input-dir",
        help="Path to images and labels file.",
        required=True,
        type=str)

    parser.add_argument(
        "-o",
        "--output",
        help="Output directory.",
        default="voc",
        type=str)

    parser.add_argument(
        "-l",
        "--labels",
        help="Path to labels file containing the classes names. If not specified, will search for a .txt file named `labels.txt` in the input dir.",
        default=None,
        type=str)

    parser.add_argument(
        "-f",
        "--filter-labels",
        help="List of labels to remove, separated by comma. Example: `-f nucleus,nor`.",
        default=None,
        type=str)

    parser.add_argument(
        "--filter-invalid",
        help="Keep empty masks.",
        default=False,
        action="store_true")

    parser.add_argument(
        "-t",
        "--tif",
        help="Save images using `.tif` format.",
        default=False,
        action="store_true")

    parser.add_argument(
        "--overlay",
        help="Generate overlay visualization.",
        default=False,
        action="store_true")

    parser.add_argument(
        "-c",
        "--color",
        help="Generate RGB segmentation masks.",
        default=False,
        action="store_true")

    args = parser.parse_args()

    if not args.labels:
        args.labels = str(Path(args.input_dir).joinpath("labels.txt"))
    if args.filter_labels:
        args.filter_labels = args.filter_labels.split(",")

    convert_annotations_to_masks(
        input_dir=args.input_dir,
        output_dir=args.output,
        labels=args.labels,
        filter_labels=args.filter_labels,
        filter_invalid=args.filter_invalid,
        save_as_tif=args.tif,
        overlay=args.overlay,
        color=args.color)


if __name__ == "__main__":
    main()
