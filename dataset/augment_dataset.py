import argparse
import itertools
import json
import multiprocessing
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


def list_files(path, validate_masks=False):
    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    images_path = Path(path).joinpath("images")
    masks_path = Path(path).joinpath("masks")

    images_paths = [image_path for image_path in images_path.glob("*.*") if image_path.suffix.lower() in supported_types and not image_path.stem.endswith("_prediction")]
    masks_paths = [mask_path for mask_path in masks_path.glob("*.*") if mask_path.suffix.lower() in supported_types and not mask_path.stem.endswith("_prediction")]

    assert len(images_paths) > 0, f"No images found at '{images_path}'."
    assert len(masks_paths) > 0, f"No masks found at '{masks_paths}'."

    images_paths.sort()
    masks_paths.sort()

    if validate_masks:
        assert len(images_paths) == len(masks_paths), f"Different quantity of images ({len(images_paths)}) and masks ({len(masks_paths)})"

        for image_path, mask_path in zip(images_paths, masks_paths):
            assert image_path.stem.lower().replace("image", "") == mask_path.stem.lower().replace("mask", ""), f"Image and mask do not correspond: {image_path.name} <==> {mask_path.name}"

    print(f"Dataset '{str(images_path.parent)}' contains {len(images_paths)} images and masks.")

    images_paths = [str(image_path) for image_path in images_paths]
    masks_paths = [str(masks_path) for masks_path in masks_paths]
    return images_paths, masks_paths


def get_transformations():
    transformations = [
        A.RandomBrightness(always_apply=True),
        A.RandomContrast(always_apply=True),
        A.VerticalFlip(always_apply=True),
        A.HorizontalFlip(always_apply=True),
        A.ElasticTransform(always_apply=True),
        A.ElasticTransform(always_apply=True)
    ]

    combinations = [
        list(transformation_step)
        for transformation_step
        in itertools.combinations(transformations, 3)
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
    def __init__(self, transformation_pipeline, output_dir, suffix, specs=None) -> None:
        self.transformations = transformation_pipeline
        self.output_dir = output_dir
        self.suffix = suffix

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        self.images_output = output_dir.joinpath("images")
        self.images_output.mkdir(exist_ok=True, parents=True)

        self.masks_output = output_dir.joinpath("masks")
        self.masks_output.mkdir(exist_ok=True, parents=True)

        if specs:
            with open(str(output_dir.joinpath("augmented_dataset_specs.json")), "w") as augmented_dataset_specs:
                json.dump(specs, augmented_dataset_specs, indent=4)


    def __call__(self, image_path, mask_path):
        image_path = Path(image_path)
        mask_path = Path(mask_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        cv2.imwrite(str(self.images_output.joinpath(f"{image_path.name}")), image)
        cv2.imwrite(str(self.masks_output.joinpath(f"{mask_path.name}")), mask)

        for j, tranformation in enumerate(self.transformations):
            transormed = tranformation(image=image, mask=mask)
            transformed_image = transormed["image"]
            transformed_mask = transormed["mask"]

            cv2.imwrite(str(self.images_output.joinpath(f"{image_path.stem}_t{j}{self.suffix}{image_path.suffix}")), transformed_image)
            cv2.imwrite(str(self.masks_output.joinpath(f"{mask_path.stem}_t{j}{self.suffix}{mask_path.suffix}")), transformed_mask)


def augment_dataset(input_dir, output_dir, suffix="", name=None, seed=None):
    if seed:
        np.random.seed(seed)

    images_paths, masks_paths = list_files(input_dir, validate_masks=True)
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

    process = ImageAugmentation(transformations, output_dir, suffix, specs)
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

    args = parser.parse_args()

    if len(args.suffix) > 0:
        args.suffix = f"_{args.suffix}"

    if not args.name:
        args.name = Path(args.output_dir).name

    augment_dataset(args.input_dir, args.output_dir, args.suffix, args.name)


if __name__ == "__main__":
    main()
