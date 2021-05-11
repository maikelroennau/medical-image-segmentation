import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def slice_images(path, height, width, filter, output):
    input_path = Path(path)
    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    images_paths = [image_path for image_path in input_path.joinpath("images").glob("*.*") if image_path.suffix.lower() in supported_types and not image_path.stem.endswith("_prediction")]
    masks_paths = [mask_path for mask_path in input_path.joinpath("masks").glob("*.*") if mask_path.suffix.lower() in supported_types and not mask_path.stem.endswith("_prediction")]

    assert len(images_paths) == len(masks_paths), f"Different quantity of images ({len(images_paths)}) and masks ({len(masks_paths)})"

    images_paths.sort()
    masks_paths.sort()

    for image, mask in zip(images_paths, masks_paths):
        assert image.stem.lower() == mask.stem.lower(), f"Image and mask do not correspond: {image.name} <==> {image.name}"

    output = input_path.joinpath(output)

    for image_path, mask_path in tqdm(zip(images_paths, masks_paths), total=len(images_paths)):
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path))

        half_height = height // 2
        half_width = width // 2

        for x in range(0, image.shape[0]-half_height, half_height):
            for y in range(0, mask.shape[1]-half_width, half_width):
                sliced_image = image[x:x+height, y:y+width]
                sliced_mask = mask[x:x+height, y:y+width]

                if filter:
                    if np.unique(sliced_mask).size == 1:
                        continue

                output_image = output.joinpath(image_path.parent.name).joinpath(f"{image_path.stem}_x{x}-{x+height}_y{y}-{y+width}.jpg")
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
        "-p",
        "--path",
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
        "-o",
        "--output",
        help="Path where to save the sliced images.",
        default="sliced",
        type=str)

    args = parser.parse_args()
    slice_images(args.path, args.height, args.width, args.filter_empty, args.output)


if __name__ == "__main__":
    main()
