import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def merge_masks(original, nucleus, nors, output, copy_images):
    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    original_images = [
        original_image
        for original_image in Path(original).glob("*.*")
        if Path(original_image).suffix in supported_types
        and not Path(original_image).stem.endswith("_prediction")]

    nuclei_masks = [
        nuclei_mask
        for nuclei_mask in Path(nucleus).glob("*.*")
        if Path(nuclei_mask).suffix in supported_types
        and Path(nuclei_mask).stem.endswith("_prediction")]

    nor_masks = [
        nor_mask
        for nor_mask in Path(nors).glob("*.*")
        if Path(nor_mask).suffix in supported_types
        and Path(nor_mask).stem.endswith("_prediction")]

    original_images.sort()
    nuclei_masks.sort()
    nor_masks.sort()

    assert len(original_images) == len(nuclei_masks) == len(nor_masks), \
        f"Number of images {len(original_images)}, nuclei masks {len(nuclei_masks)}, and NOR masks {len(nor_masks)} does not match."

    if not Path(output).is_dir():
        Path(output).mkdir(exist_ok=True)

    for original_image, nucleus, nor in tqdm(zip(original_images, nuclei_masks, nor_masks), total=len(original_images)):
        assert nucleus.stem.split("_")[0] == nor.stem.split("_")[0], \
            f"Masks do not match: \n  - Nucleus: {nucleus.stem.split('_')[0]} \n  - NOR....: {nor.stem.split('_')[0]}"

        nucleus_mask = cv2.imread(str(nucleus), cv2.IMREAD_GRAYSCALE)
        nor_mask = cv2.imread(str(nor), cv2.IMREAD_GRAYSCALE)

        if np.max(nucleus_mask) == 127:
            nucleus_mask = nucleus_mask * 2

        if np.max(nor_mask) == 127:
            nor_mask = nor_mask * 2

        nucleus_mask[nucleus_mask < 127] = 0
        nucleus_mask[nucleus_mask >= 127] = 1

        nor_mask[nor_mask < 127] = 0
        nor_mask[nor_mask >= 127] = 1

        nor_mask = nor_mask * nucleus_mask

        result = np.zeros(nucleus_mask.shape[:2] + (3,))
        result[:, :, 0] = np.where(nucleus_mask == 0, 127, 0)
        result[:, :, 1] = np.where(nucleus_mask == 1, 127, 0)
        result[:, :, 1] = np.where(nor_mask == 1, 0, result[:, :, 1])
        result[:, :, 2] = np.where(nor_mask == 1, 127, 0)

        shutil.copyfile(str(original_image), str(Path(output).joinpath(original_image.name)))
        cv2.imwrite(str(Path(output).joinpath(f"{original_image.stem}_1_multiclass.png")), cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB))

def main():
    parser = argparse.ArgumentParser(description="Predicts using the indicated model on the indicated data.")

    parser.add_argument(
        "--original",
        help="Path to the original images.",
        required=True,
        type=str)

    parser.add_argument(
        "--nucleus",
        help="Path to the nucleus predictions.",
        required=True,
        type=str)

    parser.add_argument(
        "--nors",
        help="Path to the NORs predictions.",
        required=True,
        type=str)

    parser.add_argument(
        "-o",
        "--output",
        help="Where to save the resulting predictions.",
        required=True,
        type=str)

    parser.add_argument(
        "-c",
        "--copy-images",
        help="Copy merged images to output directory.",
        default=False,
        action="store_true")

    args = parser.parse_args()

    merge_masks(args.original, args.nucleus, args.nors, args.output, args.copy_images)


if __name__ == "__main__":
    main()
