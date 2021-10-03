import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.data_io import list_files, load_image, load_mask


def get_masks_properties(masks_paths):
    # Obtain all contours from all masks
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


def contains_rectangle(cropping_rect, bounding_rect):
    return (cropping_rect["x"] <= bounding_rect["x"]) and (cropping_rect["y"] <= bounding_rect["y"]) \
        and (cropping_rect["xx"] >= bounding_rect["xx"]) and (cropping_rect["yy"] >= bounding_rect["yy"])


def random_crop(input_dir, output_dir, side, gamma, suffix):
    images_paths, masks_paths = list_files(input_dir, validate_masks=True)

    output = Path(output_dir)
    images_output = output.joinpath("images")
    masks_output = output.joinpath("masks")
    images_output.mkdir(exist_ok=True, parents=True)
    masks_output.mkdir(exist_ok=True, parents=True)

    polygons, rects = get_masks_properties(masks_paths)

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
        image = load_image(image_path)
        mask = load_mask(mask_path)

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
            cv2.imwrite(str(Path(images_output).joinpath(rectangle + ".png")), cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            cv2.imwrite(str(Path(masks_output).joinpath(rectangle + ".png")), cv2.cvtColor(cropped_mask, cv2.COLOR_BGR2RGB))


def main():
    parser = argparse.ArgumentParser(description="Randomly crops a fix sized bounding box around the bounding rect of segmetned objects.")

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
        "--suffix",
        help="Suffix for the generated images.",
        default="",
        type=str)

    parser.add_argument(
        "-s",
        "--seed",
        help="Seed for reproducibility.",
        type=int)

    args = parser.parse_args()

    if args.seed:
        seed = args.seed
    else:
        seed = 7613

    random.seed(seed)
    np.random.seed(seed)

    if not args.output_dir:
        args.output_dir = "cropped"

    if len(args.suffix) > 0:
        args.suffix = f"_{args.suffix}"

    random_crop(args.input_dir, args.output_dir, args.side, args.gamma, args.suffix)


if __name__ == "__main__":
    main()
