import os
import shutil
import sys
from glob import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main(path, output=".", x_split=0.75, y_split=0.88):
    np.random.seed(1145)

    voc_dir = Path(path)
    assert voc_dir.is_dir(), "Input path does not exists"

    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    images_path = [image_path for image_path in voc_dir.joinpath("JPEGImages").glob("*.*") if image_path.suffix.lower() in supported_types]
    masks_path = [mask_path for mask_path in voc_dir.joinpath("SegmentationClassPNG").glob("*.*") if mask_path.suffix.lower() in supported_types]

    assert len(images_path) == len(masks_path), f"Different quantity of images ({len(images_path)}) and masks ({len(masks_path)})"

    images_path.sort()
    masks_path.sort()

    for image, mask in zip(images_path, masks_path):
        assert image.stem.lower() == mask.stem.lower(), f"Image and mask do not correspond: {image.name} <==> {image.name}"

    randomize = np.random.permutation(len(images_path))
    images_path = np.asarray(images_path)[randomize]
    masks_path = np.asarray(masks_path)[randomize]

    images_train, images_validation, images_test = np.split(images_path, [int(len(images_path) * float(x_split)), int(len(images_path) * float(y_split))])
    masks_train, masks_validation, masks_test = np.split(masks_path, [int(len(masks_path) * float(x_split)), int(len(masks_path) * float(y_split))])

    split_dataset = Path(output)
    if not str(split_dataset) == ".":
        split_dataset.mkdir()

    train_images = split_dataset.joinpath("train").joinpath("images")
    train_masks = split_dataset.joinpath("train").joinpath("masks")
    train_images.mkdir(parents=True)
    train_masks.mkdir(parents=True)
    for image, mask in tqdm(zip(images_train, masks_train), total=len(images_train)):
        shutil.copyfile(image, train_images.joinpath(image.name))
        shutil.copyfile(mask, train_masks.joinpath(mask.name))

    validation_images = split_dataset.joinpath("validation").joinpath("images")
    validation_masks = split_dataset.joinpath("validation").joinpath("masks")
    validation_images.mkdir(parents=True)
    validation_masks.mkdir(parents=True)
    for image, mask in tqdm(zip(images_validation, masks_validation), total=len(images_validation)):
        shutil.copyfile(image, validation_images.joinpath(image.name))
        shutil.copyfile(mask, validation_masks.joinpath(mask.name))

    test_images = split_dataset.joinpath("test").joinpath("images")
    test_masks = split_dataset.joinpath("test").joinpath("masks")
    test_images.mkdir(parents=True)
    test_masks.mkdir(parents=True)
    for image, mask in tqdm(zip(images_test, masks_test), total=len(images_test)):
        shutil.copyfile(image, test_images.joinpath(image.name))
        shutil.copyfile(mask, test_masks.joinpath(mask.name))
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        print("Please provide the VOC images path")
