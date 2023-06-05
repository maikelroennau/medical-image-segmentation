import argparse
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

from utils.data import SUPPORTED_IMAGE_TYPES
from utils.utils import pad_along_axis


def process(images: str, axis: int, pixels: int, mode: Optional[str] = "constant", grayscale: Optional[bool] = False) -> None:
    """Process the images and pad them or remove pixels from them according to the arguments.

    Args:
        images (str): The path to the images to process.
        axis (int): The axis of the images to change. Must be either `0` (width) or `1` (height).
        pixels (int): The number of pixels the output images must have.
        mode (Optional[str], optional): How to fill new pixels. Only effective if adding new pixels to the image. Defaults to "constant".
        grayscale (Optional[bool], optional): Load images in grayscale. Defaults to `False`.
    """
    files = [str(file) for file in Path(images).glob("*") if file.suffix in SUPPORTED_IMAGE_TYPES]

    for file in tqdm(files):
        if grayscale:
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(file)

        if image.shape[axis] > pixels:
            if axis == 0:
                if len(image.shape) == 3:
                    image = image[:pixels, :, :]
                else:
                    image = image[:pixels, :]
            elif axis == 1:
                if len(image.shape) == 3:
                    image = image[:, :pixels, :]
                else:
                    image = image[:, :pixels]
        else:
            image = pad_along_axis(image, pixels, axis, mode=mode)

        cv2.imwrite(file, image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pad or remove padding from images.")

    parser.add_argument(
        "-i",
        "--images",
        help="The image to pad or remove padding.",
        type=str)

    parser.add_argument(
        "-a",
        "--axis",
        help="The axis of the image to pad or remove padding from. Must be either `0` (height) or `1` (width).",
        type=int)

    parser.add_argument(
        "-p",
        "--pixels",
        help="The number of pixels the resulting image must have in the given axis.",
        type=int)

    parser.add_argument(
        "-m",
        "--mode",
        help="The interpolation method. Must be one of [`constant`, `edge`, `linear_ramp`, `maximum`, `mean`, `median`, `minimum`, `reflect`, `symmetric`, `wrap`, `empty`]. Only effective if adding pixels to the image. Defaults to `constant`.",
        default="constant",
        type=str)
    
    parser.add_argument(
        "-g",
        "--grayscale",
        help="Load images in grayscale. Defaults to `False`.",
        choices=["true", "false"],
        default="false",
        type=str)

    args = parser.parse_args()

    process(
        images=args.images,
        axis=args.axis,
        pixels=args.pixels,
        mode=args.mode,
        grayscale=True if args.augmentation.lower() == "true" else False)


if __name__ == "__main__":
    main()
