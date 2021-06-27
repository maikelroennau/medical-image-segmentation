import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def split_dataset(input_dir, output_dir, multiple_datasets, train, validation, test):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    assert input_dir.is_dir(), "Input path does not exists"

    if multiple_datasets:
        datasets = [dataset for dataset in input_dir.glob("*")]
    else:
        datasets = [input_dir]

    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    for dataset in datasets:
        images_path = [image_path for image_path in dataset.joinpath("Images").glob("*.*") if image_path.suffix.lower() in supported_types]
        masks_path = [mask_path for mask_path in dataset.joinpath("SegmentationClassPNG").glob("*.*") if mask_path.suffix.lower() in supported_types]

        if len(images_path) == 0:
            print(f"No images found in `{str(input_dir)}`")
            break

        assert len(images_path) == len(masks_path), f"Different quantity of images ({len(images_path)}) and masks ({len(masks_path)})"
        n_images = len(images_path)

        images_path.sort()
        masks_path.sort()

        for image, mask in zip(images_path, masks_path):
            assert image.stem.lower() == mask.stem.lower(), f"Image and mask do not correspond: {image.name} <==> {image.name}"

        print(f"\nDataset '{str(dataset)}' contains {n_images} images and masks.")

        randomize = np.random.RandomState(seed=1145).permutation(n_images)
        images_path = np.asarray(images_path)[randomize]
        masks_path = np.asarray(masks_path)[randomize]

        train_split = int(train * n_images)
        validation_split = train_split + int(validation * n_images)
        test_split = validation_split + int(test * n_images)

        images_train, images_validation, images_test, _ = np.split(images_path, [train_split, validation_split, test_split])
        masks_train, masks_validation, masks_test, _ = np.split(masks_path, [train_split, validation_split, test_split])

        if len(datasets) > 1:
            dataset_output_dir = output_dir.joinpath(dataset.name)
        output_dir.mkdir(exist_ok=True, parents=True)

        train_images = dataset_output_dir.joinpath("train").joinpath("images")
        train_masks = dataset_output_dir.joinpath("train").joinpath("masks")
        train_images.mkdir(exist_ok=True, parents=True)
        train_masks.mkdir(exist_ok=True, parents=True)
        for image, mask in tqdm(zip(images_train, masks_train), total=len(images_train), desc="train"):
            shutil.copyfile(image, train_images.joinpath(image.name))
            shutil.copyfile(mask, train_masks.joinpath(mask.name))

        validation_images = dataset_output_dir.joinpath("validation").joinpath("images")
        validation_masks = dataset_output_dir.joinpath("validation").joinpath("masks")
        validation_images.mkdir(exist_ok=True, parents=True)
        validation_masks.mkdir(exist_ok=True, parents=True)
        for image, mask in tqdm(zip(images_validation, masks_validation), total=len(images_validation), desc="validation"):
            shutil.copyfile(image, validation_images.joinpath(image.name))
            shutil.copyfile(mask, validation_masks.joinpath(mask.name))

        test_images = dataset_output_dir.joinpath("test").joinpath("images")
        test_masks = dataset_output_dir.joinpath("test").joinpath("masks")
        test_images.mkdir(exist_ok=True, parents=True)
        test_masks.mkdir(exist_ok=True, parents=True)
        for image, mask in tqdm(zip(images_test, masks_test), total=len(images_test), desc="test"):
            shutil.copyfile(image, test_images.joinpath(image.name))
            shutil.copyfile(mask, test_masks.joinpath(mask.name))


def main():
    parser = argparse.ArgumentParser(description="Splits a dataset of images into train, validation and test.")

    parser.add_argument(
        "-i",
        "--input-dir",
        help="Path to the dataset directory.",
        required=True,
        type=str)

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory.",
        default="split_dataset",
        type=str)

    parser.add_argument(
        "-m",
        "--multiple-datasets",
        help="Whether or not `input_dir` contains multiple datasets. If true, will split each dataset individually",
        default=False,
        action="store_true")

    parser.add_argument(
        "--train",
        help="Percentage of images for training in range [0, 1].",
        default=0.75,
        type=float)

    parser.add_argument(
        "--val",
        help="Percentage of images for validation in range [0, 1].",
        default=0.125,
        type=float)

    parser.add_argument(
        "--test",
        help="Percentage of images for test in range [0, 1].",
        default=0.125,
        type=float)

    args = parser.parse_args()

    assert args.input_dir != args.output_dir, "Input and out put dir must be different."
    assert np.sum([args.train, args.val, args.test]) == 1, "Sum of percentages should be 1."

    split_dataset(args.input_dir, args.output_dir, args.multiple_datasets, args.train, args.val, args.test)


if __name__ == "__main__":
    main()
