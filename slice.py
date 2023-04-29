import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.io import imsave
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.contour_analysis import discard_contours_by_size, get_contours
from utils.data import list_files, load_image
from utils.utils import color_classes


def slice_images(
    input_dir: str,
    output: str,
    height: int,
    width: int,
    filter: Optional[bool] = False,
    overlap: Optional[bool] = False,
    color: Optional[bool] = False) -> None:
    """Slice images and masks in the specified height and width.

    Args:
        input_dir (str): The path to the images to augment. Should be a path to a directory containing an `images` subdirectory and `masks` subdirectory.
        output (str): THe path where to save the sliced images.
        height (int): The hight the sliced images should have.
        width (int): The width the sliced images should have.
        filter (Optional[bool], optional): Whether or not to filter sliced images and masks contains only background pixels. Defaults to False.
        overlap (Optional[bool], optional): Whether or not to stride the slicing so that half of the image is in overlap with the previous one. Defaults to False.
        color (Optional[bool], optional): Whether or not to color the segmentation masks.
    """
    input_path = Path(input_dir)
    images_paths = list_files(str(input_path.joinpath("images")), as_numpy=True)
    masks_paths = list_files(str(input_path.joinpath("masks")), as_numpy=True)

    if output:
        output = Path(output)
        output.mkdir(exist_ok=True, parents=True)
    else:
        output = input_path.joinpath("sliced")

    for image_path, mask_path in tqdm(zip(images_paths, masks_paths), total=len(images_paths)):
        image = load_image(image_path, as_numpy=True)
        mask = load_image(mask_path, as_gray=True, as_numpy=True)

        if overlap:
            half_height = height // 2
            half_width = width // 2
            x_range = range(0, image.shape[0]-half_height, half_height)
            y_range = range(0, mask.shape[1]-half_width, half_width)
        else:
            half_height = height // 1
            half_width = width // 1
            x_range = range(0, image.shape[0], half_height)
            y_range = range(0, mask.shape[1], half_width)

        for x in x_range:
            for y in y_range:
                sliced_image = image[x:x+height, y:y+width]
                sliced_mask = mask[x:x+height, y:y+width]

                if filter:
                    if np.unique(sliced_mask).size == 1:
                        continue
                    else:
                        contours = get_contours(sliced_mask.copy())
                        contours, _ = discard_contours_by_size(contours, shape=image.shape[:2])
                        if len(contours) == 0:
                            continue

                image_path = Path(image_path)
                mask_path = Path(mask_path)
                output_image = output.joinpath(image_path.parent.name).joinpath(f"{image_path.stem}_x{x}-{x+height}_y{y}-{y+width}.png")
                output_mask = output.joinpath(mask_path.parent.name).joinpath(f"{mask_path.stem}_x{x}-{x+height}_y{y}-{y+width}.png")
                output_image.parent.mkdir(parents=True, exist_ok=True)
                output_mask.parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(output_image), cv2.cvtColor(sliced_image, cv2.COLOR_BGR2RGB))
                if color:
                    sliced_mask = color_classes(sliced_mask)
                imsave(str(output_mask), sliced_mask, check_contrast=False)

    print(f"Created {len([image_path for image_path in output.joinpath('images').glob('*.*')])} images.")
    print(f"Created {len([mask_path for mask_path in output.joinpath('masks').glob('*.*')])} masks.")


def main():
    parser = argparse.ArgumentParser(description="Slice")

    parser.add_argument(
        "-i",
        "--input-dir",
        help="Path to the images and masks directory.",
        type=str)

    parser.add_argument(
        "-o",
        "--output",
        help="Path where to save the sliced images. If not specified, will save in `input-dir`.",
        default=None,
        type=str)

    parser.add_argument(
        "--height",
        help="Height of the sliced image.",
        type=int)

    parser.add_argument(
        "--width",
        help="Width of the sliced image.",
        type=int)

    parser.add_argument(
        "-f",
        "--filter-empty",
        help="Filter out empty masks.",
        default=False,
        action="store_true")

    parser.add_argument(
        "--overlap",
        help="Allow overlaping crops.",
        default=False,
        action="store_true")

    parser.add_argument(
        "--color",
        help="Whether or not to color the segmentation masks.",
        default=False,
        action="store_true")

    args = parser.parse_args()
    slice_images(
        input_dir=args.input_dir,
        output=args.output,
        height=args.height,
        width=args.width,
        filter=args.filter_empty,
        overlap=args.overlap,
        color=args.color)


if __name__ == "__main__":
    main()
