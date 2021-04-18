import glob
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def main(images_path="dataset/test/", output_path="merged_masks"):
    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    original_images = [
        original_image
        for original_image in Path(images_path).glob("*.*")
        if Path(original_image).suffix in supported_types
        and not Path(original_image).stem.endswith("_prediction")]

    nuclei_masks = [
        nuclei_mask
        for nuclei_mask in Path(images_path).glob("*.*")
        if Path(nuclei_mask).suffix in supported_types
        and Path(nuclei_mask).stem.endswith("AgNOR-Nucleus_prediction")]

    nor_masks = [
        nor_mask
        for nor_mask in Path(images_path).glob("*.*")
        if Path(nor_mask).suffix in supported_types
        and Path(nor_mask).stem.endswith("AgNOR-NOR_prediction")]

    for original_image, nucleus, nor in tqdm(zip(original_images, nuclei_masks, nor_masks), total=len(original_images)):
        nucleus_mask = cv2.imread(str(nucleus), cv2.IMREAD_UNCHANGED)
        nor_mask = cv2.imread(str(nor), cv2.IMREAD_UNCHANGED)

        nucleus_mask[nucleus_mask < 127] = 0
        nucleus_mask[nucleus_mask >= 127] = 1

        nor_mask[nor_mask < 127] = 0
        nor_mask[nor_mask >= 127] = 1

        result = np.add(nucleus_mask, nor_mask)
        result[result == 1] = 255
        result[result == 2] = 127
        result = result * nucleus_mask

        if not Path(output_path).is_dir():
            Path(output_path).mkdir(exist_ok=True)

        shutil.copyfile(str(original_image), str(Path(output_path).joinpath(original_image.name)))
        cv2.imwrite(str(Path(output_path).joinpath(f"{original_image.stem}_multiclass.png")), result)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()