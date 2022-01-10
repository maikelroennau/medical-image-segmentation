import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.data import list_files, load_image
from utils.utils import color_classes


def get_masks_properties(masks_paths: List) -> Tuple[dict, dict]:
    """Obtains the parent contours of all masks.

    Args:
        masks_paths (List): A list of segmentation masks.

    Returns:
        Tuple[dict, dict]: A tuple containing the contour objects, and the bounding rectangle of the contours.
    """
    polygons = {}
    rects = {}

    for mask_path in tqdm(masks_paths, desc="Get masks properties"):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 255

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for j, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            polygons[f"{Path(mask_path).stem}_{j}"] = contour
            rects[f"{Path(mask_path).stem}_{j}"] = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }

    return polygons, rects


def contains_rectangle(cropping_rect: np.ndarray, bounding_rect: np.ndarray) -> bool:
    """Verifies if a rectangle that will be used to crop pixels from an image contains the entirety of the contour bounding rectangle.

    Args:
        cropping_rect (np.ndarray): The greater contour that will be used to crop pixels from the image.
        bounding_rect (np.ndarray): The rectangle bounding the contour.

    Returns:
        bool: Whether or not `cropping_rect` contains `bounding_rect`.
    """
    return (cropping_rect["x"] <= bounding_rect["x"]) and (cropping_rect["y"] <= bounding_rect["y"]) \
        and (cropping_rect["xx"] >= bounding_rect["xx"]) and (cropping_rect["yy"] >= bounding_rect["yy"])


def random_crop(
    input_dir: str,
    output_dir: str,
    side: int,
    gamma: Optional[float] = 1.5,
    color: Optional[bool] = False) -> None:
    """Crops a rectangle randomly positoned over the bounding rectangle of a segmentation contour.

    Args:
        input_dir (str): Path to the directory containing the images and masks to crop.
        output_dir (str): Path to the output directory.
        side (int): Side of the crop in pixels. (e.g., `100` will crop a 100x100 box). If not specified will used the biggest found in the data + gamma.
        gamma (int): Percentage factor used to rescale the size of the crop retangle. Defaults to 50 percent (1.5).
        color (Optional[bool], optional): Whether or not to color the segmentation masks.
    """
    input_path = Path(input_dir)
    images_paths = list_files(str(input_path.joinpath("images")), as_numpy=True)
    masks_paths = list_files(str(input_path.joinpath("masks")), as_numpy=True)

    output = Path(output_dir)
    images_output = output.joinpath("images")
    masks_output = output.joinpath("masks")
    images_output.mkdir(exist_ok=True, parents=True)
    masks_output.mkdir(exist_ok=True, parents=True)

    _, rects = get_masks_properties(masks_paths)

    if not side:
        biggest_area = 0
        for value in rects.values():
            height = value["h"]
            width = value["w"]
            area = height * width
            if area > biggest_area:
                biggest_area = area
                side = height if height > width else width

        # Adjust side accordingly to gama to allow some space for randomizing the cut
        side *= gamma

    # Round side to the next even number
    side = int(np.ceil(side / 2.) * 2)
    half_side = side // 2

    for image_path, mask_path in tqdm(zip(images_paths, masks_paths), total=len(images_paths), desc="Random crop"):
        image = load_image(image_path, as_numpy=True)
        mask = load_image(mask_path, as_gray=True, as_numpy=True)

        rectangles = [key for key in rects.keys() if key.startswith(Path(image_path).stem)]
        for rectangle in rectangles:
            while True:
                new_x = np.random.randint(rects[rectangle]["x"], rects[rectangle]["x"] + rects[rectangle]["w"]) - half_side
                new_y = np.random.randint(rects[rectangle]["y"], rects[rectangle]["y"] + rects[rectangle]["h"]) - half_side

                if new_x < 0:
                    new_x = 0
                elif new_x + side > image.shape[1]:
                    new_x = image.shape[1] - side

                if new_y < 0:
                    new_y = 0
                elif new_y + side > image.shape[0]:
                    new_y = image.shape[0] - side

                contains = contains_rectangle(
                    cropping_rect={
                        "x": new_x,
                        "y": new_y,
                        "xx": new_x + side,
                        "yy": new_y + side
                    },
                    bounding_rect={
                        "x": rects[rectangle]["x"],
                        "y": rects[rectangle]["y"],
                        "xx": rects[rectangle]["x"] + rects[rectangle]["w"],
                        "yy": rects[rectangle]["y"] + rects[rectangle]["h"]
                    }
                )

                if contains:
                    break

            cropped_image = image[new_y:new_y+side, new_x:new_x+side, :]
            cropped_mask = mask[new_y:new_y+side, new_x:new_x+side, :]

            cv2.imwrite(
                str(Path(images_output).joinpath(rectangle + ".png")), cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            if color:
                cropped_mask = color_classes(cropped_mask)
            else:
                cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_BGR2RGB)

            cv2.imwrite(str(Path(masks_output).joinpath(rectangle + ".png")), cropped_mask)


def main():
    parser = argparse.ArgumentParser(description="Randomly crops a fix sized bounding box around the bounding rect of segmented objects.")

    parser.add_argument(
        "-i",
        "--input-dir",
        help="Path to the directory containing the images and masks to crop.",
        required=True,
        type=str)

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Path to the output dir.",
        required=False,
        type=str)

    parser.add_argument(
        "--side",
        help="Side of the crop in pixels. (e.g., '100' will crop a 100x100 box). If not specified will used the biggest found in the data + gamma.",
        type=int)

    parser.add_argument(
        "--gamma",
        help="Percentage factor used to rescale the size of the crop retangle. Defaults to 50 percent (1.5).",
        default=1.5,
        type=float)

    parser.add_argument(
        "-s",
        "--seed",
        help="Seed for reproducibility.",
        type=int)

    parser.add_argument(
        "--color",
        help="Whether or not to color the segmentation masks.",
        default=False,
        action="store_true")

    args = parser.parse_args()

    if args.seed:
        seed = args.seed
    else:
        seed = 7613

    random.seed(seed)
    np.random.seed(seed)

    if not args.output_dir:
        args.output_dir = "cropped"

    random_crop(args.input_dir, args.output_dir, args.side, args.gamma, args.color)


if __name__ == "__main__":
    main()
