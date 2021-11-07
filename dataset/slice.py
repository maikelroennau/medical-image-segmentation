import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.post_process import filter_contours_by_size, get_contours


def slice_images(input_dir, height, width, filter, overlap, output):
    input_path = Path(input_dir)
    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    images_paths = [image_path for image_path in input_path.joinpath("images").glob("*.*") if image_path.suffix.lower() in supported_types and not image_path.stem.endswith("_prediction")]
    masks_paths = [mask_path for mask_path in input_path.joinpath("masks").glob("*.*") if mask_path.suffix.lower() in supported_types and not mask_path.stem.endswith("_prediction")]

    assert len(images_paths) == len(masks_paths), f"Different quantity of images ({len(images_paths)}) and masks ({len(masks_paths)})"

    images_paths.sort()
    masks_paths.sort()

    for image, mask in zip(images_paths, masks_paths):
        assert image.stem.lower() == mask.stem.lower(), f"Image and mask do not correspond: {image.name} <==> {image.name}"

    if output:
        output = Path(output)
        output.mkdir(exist_ok=True, parents=True)
    else:
        output = input_path.joinpath("sliced")

    for image_path, mask_path in tqdm(zip(images_paths, masks_paths), total=len(images_paths)):
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path))

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
                        contours = get_contours(sliced_mask[:, :, 1].copy())
                        contours, _ = filter_contours_by_size(contours)
                        if len(contours) == 0:
                            continue

                output_image = output.joinpath(image_path.parent.name).joinpath(f"{image_path.stem}_x{x}-{x+height}_y{y}-{y+width}.png")
                output_mask = output.joinpath(mask_path.parent.name).joinpath(f"{mask_path.stem}_x{x}-{x+height}_y{y}-{y+width}.png")
                output_image.parent.mkdir(parents=True, exist_ok=True)
                output_mask.parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(output_image), sliced_image)
                cv2.imwrite(str(output_mask), sliced_mask)

    print(f"Created {len([image_path for image_path in output.joinpath('images').glob('*.*')])} images.")
    print(f"Created {len([mask_path for mask_path in output.joinpath('masks').glob('*.*')])} masks.")


def main():
    parser = argparse.ArgumentParser(description="Slice")

    parser.add_argument(
        "-i",
        "--input-dir",
        help="Path to the images directory.",
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
        "-o",
        "--output",
        help="Path where to save the sliced images. If not specified, will save in `input-dir`.",
        default=None,
        type=str)

    args = parser.parse_args()
    slice_images(args.input_dir, args.height, args.width, args.filter_empty, args.overlap, args.output)


if __name__ == "__main__":
    main()
