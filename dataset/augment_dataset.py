import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


def get_transformations():
    transformations = []

    ###########
    ## Set 1 ##
    ###########
    transformations.append(
        A.Compose([
            A.HorizontalFlip(p=1),
            A.GridDistortion(p=1)
        ])
    )

    transformations.append(
        A.Compose([
            A.VerticalFlip(p=1),
            A.GridDistortion(p=1)
        ])
    )

    transformations.append(
        A.Compose([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.GridDistortion(p=1)
        ])
    )

    transformations.append(
        A.Compose([
            A.GridDistortion(p=1),
            A.HorizontalFlip(p=1),
            A.GridDistortion(p=1),
            A.VerticalFlip(p=1),
            A.GridDistortion(p=1)
        ])
    )

    ###########
    ## Set 4 ##
    ###########
    transformations.append(
        A.Compose([
            A.GridDistortion(p=1),
            A.GridDistortion(p=1)
        ])
    )

    return transformations


def augment_dataset(input_dir, output_dir, suffix="", seed=None):
    if seed:
        np.random.seed(seed)

    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    images_path = Path(input_dir).joinpath("images")
    masks_path = Path(input_dir).joinpath("masks")

    images_paths = [image_path for image_path in images_path.glob("*.*") if image_path.suffix.lower() in supported_types and not image_path.stem.endswith("_prediction")]
    masks_paths = [mask_path for mask_path in masks_path.glob("*.*") if mask_path.suffix.lower() in supported_types and not mask_path.stem.endswith("_prediction")]

    images_paths.sort()
    masks_paths.sort()

    assert len(images_paths) == len(masks_paths), f"Different quantity of images ({len(images_paths)}) and masks ({len(masks_paths)})"

    for image_path, mask_path in zip(images_paths, masks_paths):
        assert image_path.stem.lower() == mask_path.stem.lower(), f"Image and mask do not correspond: {image_path.name} <==> {mask_path.name}"

    print(f"Dataset '{str(images_path.parent)}' contains {len(images_paths)} images and masks.")

    transformations = get_transformations()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    images_output = output_dir.joinpath("images")
    images_output.mkdir(exist_ok=True, parents=True)

    masks_output = output_dir.joinpath("masks")
    masks_output.mkdir(exist_ok=True, parents=True)

    for i, (image_path, mask_path) in tqdm(enumerate(zip(images_paths, masks_paths)), total=len(images_paths)):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        cv2.imwrite(str(images_output.joinpath(f"{image_path.name}")), image)
        cv2.imwrite(str(masks_output.joinpath(f"{mask_path.name}")), mask)

        for j, tranformation in enumerate(transformations):
            transormed = tranformation(image=image, mask=mask)
            transformed_image = transormed["image"]
            transformed_mask = transormed["mask"]

            cv2.imwrite(str(images_output.joinpath(f"{image_path.stem}_t{j}{suffix}{image_path.suffix}")), transformed_image)
            cv2.imwrite(str(masks_output.joinpath(f"{mask_path.stem}_t{j}{suffix}{mask_path.suffix}")), transformed_mask)


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

    args = parser.parse_args()

    if len(args.suffix) > 0:
        args.suffix = f"_{args.suffix}"

    augment_dataset(args.input_dir, args.output_dir, args.suffix)


if __name__ == "__main__":
    main()
