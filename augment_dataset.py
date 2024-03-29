import argparse
import itertools
import json
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import albumentations as A
import cv2
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.data import list_files, load_image
from utils.utils import color_classes


def get_transformations() -> List:
    """Creates and returns a list of image augmentation transformations.

    Returns:
        List: A list containing the transformation pipeline.
    """
    transformations = [
        A.RandomBrightness(p=0.5),
        A.RandomContrast(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(p=0.5),
        A.ElasticTransform(p=0.5)
    ]

    combinations = [
        list(transformation_step)
        for transformation_step
        in itertools.combinations(transformations, 5)
    ]

    for combination in combinations:
        random.shuffle(combination)

    combinations = [
        A.Compose(transformation_step)
        for transformation_step
        in combinations
    ]

    return combinations


class ImageAugmentation:
    """A class that performs image augmentation in parallel."""
    def __init__(
        self,
        transformation_pipeline: List,
        output_dir: str,
        suffix: str,
        specs: Optional[dict] = None,
        color: Optional[bool] = False) -> None:
        """Initialize the a `ImageAugmentation` instance.

        Args:
            transformation_pipeline (List): The list of transformations to be applied.
            output_dir (str): Where to save the augmented images.
            suffix (str): A suffix to be added to the augmented images.
            specs (Optional[dict], optional): A dictionary containing the specification of the transformations applied.. Defaults to None.
            color (Optional[bool], optional): Whether or not to color the segmentation masks.
        """
        self.transformations = transformation_pipeline
        self.output_dir = output_dir
        self.suffix = suffix

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        self.images_output = output_dir.joinpath("images")
        self.images_output.mkdir(exist_ok=True, parents=True)

        self.masks_output = output_dir.joinpath("masks")
        self.masks_output.mkdir(exist_ok=True, parents=True)

        self.color = color

        if specs:
            with open(str(output_dir.joinpath("augmented_dataset_specs.json")), "w") as augmented_dataset_specs:
                json.dump(specs, augmented_dataset_specs, indent=4)


    def __call__(self, image_path, mask_path):
        """The callable function that will perform the image transformation.

        Args:
            image_path (str): The path to the image to augment.
            mask_path (str): The path to the respective segmentation mask to augment.
        """
        image_path = Path(image_path)
        mask_path = Path(mask_path)
        image = load_image(str(image_path), as_numpy=True)
        mask = load_image(str(mask_path), as_gray=True, as_numpy=True)

        cv2.imwrite(str(self.images_output.joinpath(f"{image_path.name}")), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.color:
            mask = color_classes(mask)
        cv2.imwrite(str(self.masks_output.joinpath(f"{mask_path.name}")), mask)

        for j, tranformation in enumerate(self.transformations):
            transformed = tranformation(image=image, mask=mask)
            transformed_image = transformed["image"]
            transformed_mask = transformed["mask"]

            cv2.imwrite(
                str(self.images_output.joinpath(f"{image_path.stem}_t{j}{self.suffix}{image_path.suffix}")),
                cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))

            if self.color:
                cv2.imwrite(
                    str(self.masks_output.joinpath(f"{mask_path.stem}_t{j}{self.suffix}{mask_path.suffix}")),
                    cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2RGB))

            cv2.imwrite(
                str(self.masks_output.joinpath(f"{mask_path.stem}_t{j}{self.suffix}{mask_path.suffix}")),
                 cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2RGB))


def augment_dataset(
    input_dir: str,
    output_dir: str,
    suffix: Optional[str] = "",
    name: Optional[str] = None,
    color: Optional[bool] = False) -> None:
    """A function that will instantiate a `ImageAugmentation` class object and use it to apply image augmentation in pararel to a dataset.

    Args:
        input_dir (str): The path to the images to augment. Should be a path to a directory containing an `images` subdirectory and `masks` subdirectory.
        output_dir (str): Where to save the augmented images.
        suffix (str): A suffix to be added to the augmented images.
        name (Optional[str], optional): A name to be added to the augmented dataset specs. Defaults to None.
        color (Optional[bool], optional): Whether or not to color the segmentation masks.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    images_paths = list_files(str(Path(input_dir).joinpath("images")), as_numpy=True)
    masks_paths = list_files(str(Path(input_dir).joinpath("masks")), as_numpy=True)
    transformations = get_transformations()

    print("\nAugmented dataset specs:")
    print(f"  - Number of images: {len(images_paths)}")
    print(f"  - Number of transformations: {len(transformations)}")
    print(f"  - Final number of images: {len(images_paths) + len(images_paths) * len(transformations)}\n")

    specs = {
        "name": name,
        "images_suffix": suffix,
        "datetime": time.strftime('%Y%m%d%H%M%S'),
        "n_images": len(images_paths),
        "n_transformations": len(transformations),
        "n_final_images": len(images_paths) + len(images_paths) * len(transformations),
        "transformations": [", ".join([str(t) for t in transformation]) for transformation in transformations]
    }

    process = ImageAugmentation(transformations, output_dir, suffix, specs, color)
    pool = multiprocessing.Pool()
    pool.starmap(process, tqdm(zip(images_paths, masks_paths), total=len(images_paths)), chunksize=1)


def main():
    parser = argparse.ArgumentParser(description="Generates augmented datasets.")

    parser.add_argument(
        "-i",
        "--input-dir",
        help="Path to the directory containing the images and masks to augment.",
        required=True,
        type=str)

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Path to the output dir.",
        required=False,
        default="augmented",
        type=str)

    parser.add_argument(
        "--suffix",
        help="Suffix for the generated images.",
        default="",
        type=str)

    parser.add_argument(
        "-n",
        "--name",
        help="A name for the dataset. If not specified, will be equal to the outputdir.",
        default="",
        type=str)

    parser.add_argument(
        "--color",
        help="Whether or not to color the segmentation masks.",
        default=False,
        action="store_true")

    args = parser.parse_args()

    if len(args.suffix) > 0:
        args.suffix = f"_{args.suffix}"

    if not args.name:
        args.name = Path(args.output_dir).name

    augment_dataset(args.input_dir, args.output_dir, args.suffix, args.name, args.color)


if __name__ == "__main__":
    main()
