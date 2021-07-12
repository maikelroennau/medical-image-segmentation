import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
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


def load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return mask


def get_masks_properties(masks_paths):
    # Obtain all contours from all masks
    polygons = {}
    rects = {}

    for mask_path in tqdm(masks_paths, desc="Get masks properties"):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 255

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for j, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            polygons[f"{Path(mask_path).stem}_{j}"] = contour
            rects[f"{Path(mask_path).stem}_{j}"] = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }

    return polygons, rects


def get_rms(array, normalized=False):
    if not normalized:
        array = array / 255.
    return np.sqrt(np.mean(array * array))


def get_contrast_groups(images_paths, masks_paths):
    groups = { f"{i}qt": { "images": [], "masks": [] } for i in range(2) }

    for image_path, mask_path in zip(images_paths, masks_paths):
        image = load_image(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rms = get_rms(image)

        if rms >= 0. and rms < 0.5:
            groups["1qt"]["images"].append(image_path)
            groups["1qt"]["masks"].append(mask_path)
        elif rms >= 0.5 and rms <= 1.:
            groups["2qt"]["images"].append(image_path)
            groups["2qt"]["masks"].append(mask_path)
    return groups


def mix_segmentation(input, output, max_mixes, group_by_contrast=False):
    if isinstance(input, str) or isinstance(input, Path):
        images_paths, masks_paths = list_files(input, validate_masks=True)

        if group_by_contrast:
            contrast_groups = get_contrast_groups(images_paths, masks_paths)
            for group in contrast_groups.keys():
                mix_segmentation(
                    (contrast_groups[group]["images"], contrast_groups[group]["masks"]),
                    output=output,
                    max_mixes=max_mixes,
                    group_by_contrast=False)
    else:
        images_paths, masks_paths = input

    output = Path(output)
    images_output = output.joinpath("images")
    masks_output = output.joinpath("masks")
    images_output.mkdir(exist_ok=True, parents=True)
    masks_output.mkdir(exist_ok=True, parents=True)

    polygons, rects = get_masks_properties(masks_paths)

    for image_path, mask_path in tqdm(zip(images_paths, masks_paths), total=len(images_paths), desc="Mix segmented areas"):
        target_image = load_image(image_path)
        target_mask = load_mask(mask_path)

        height = target_image.shape[0]
        width = target_mask.shape[1]

        polygon_keys = list(polygons.keys())
        random.shuffle(polygon_keys)
        for i, polygon_key in enumerate(polygon_keys):
            # Add new nucleis if randomly over 0.5 and nucleus not already in image
            if tf.random.uniform(()) > 0.5 and Path(image_path).stem != polygon_key.split("_")[0]:
                source_image = load_image(Path(Path(image_path).parent).joinpath(f"{polygon_key.split('_')[0]}{Path(image_path).suffix}"))
                source_mask = load_mask(Path(Path(mask_path).parent).joinpath(f"{polygon_key.split('_')[0]}{Path(mask_path).suffix}"))

                source_mask_roi = np.zeros_like(source_mask)
                cv2.drawContours(source_mask_roi, contours=[polygons[polygon_key]], contourIdx=-1, color=1, thickness=cv2.FILLED)
                source_mask_roi *= source_mask

                x, y, w, h = rects[polygon_key]["x"], rects[polygon_key]["y"], rects[polygon_key]["w"], rects[polygon_key]["h"]
                source_image_rect_roi = source_image[y:y+h, x:x+w].copy()
                source_mask_rect_roi = source_mask_roi[y:y+h, x:x+w].copy()

                background_filter = source_mask_rect_roi.copy()
                background_filter[background_filter > 0] = 1
                source_image_rect_roi *= background_filter[:, :, None]

                tranformation = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ])
                transormed = tranformation(image=source_image_rect_roi, mask=source_mask_rect_roi)
                source_image_rect_roi = transormed["image"]
                source_mask_rect_roi = transormed["mask"]

                max_colisions = 5
                colisions = 0
                while colisions < max_colisions:
                    new_x = tf.math.abs(tf.random.uniform((), minval=0, maxval=width-w, dtype="int32")).numpy()
                    new_y = tf.math.abs(tf.random.uniform((), minval=0, maxval=height-h, dtype="int32")).numpy()

                    mask_roi = np.zeros_like(target_mask)
                    mask_roi[new_y:new_y+h, new_x:new_x+w] = source_mask_rect_roi

                    colision_target = target_mask.copy()
                    colision_target[colision_target > 0] = 1
                    colision_roi = mask_roi.copy()
                    colision_roi[colision_roi > 0] = 1

                    if np.max(colision_target + colision_roi) > 1:
                        colisions += 1
                        continue
                    else:
                        break

                target_mask += mask_roi.copy()

                image_roi = np.zeros_like(target_image)
                image_roi[new_y:new_y+h, new_x:new_x+w] = source_image_rect_roi

                mask_roi[mask_roi > 0] = 1
                mask_roi[mask_roi == 0] = 255
                mask_roi[mask_roi == 1] = 0
                mask_roi[mask_roi == 255] = 1
                target_image *= mask_roi[:, :, None]
                target_image += image_roi

            if i == max_mixes - 1:
                break

        cv2.imwrite(f"{images_output.joinpath(Path(image_path).stem)}_mixed_{Path(image_path).suffix}", cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"{masks_output.joinpath(Path(mask_path).stem)}_mixed_{Path(mask_path).suffix}", target_mask)


def main():
    parser = argparse.ArgumentParser(description="Mixes segmentation masks by placing segmentation pixels from one image in another.")

    parser.add_argument(
        "-i",
        "--input-dir",
        help="Path to the directory containing the images and masks to mix.",
        required=True,
        type=str)

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Path to the output dir.",
        required=False,
        type=str)

    parser.add_argument(
        "-m",
        "--max-mixes",
        help="Max number of mixes. Defaults to 10.",
        default=15,
        type=int)

    parser.add_argument(
        "-g",
        "--group-contrasts",
        help="Mixes images considering similar contrasts.",
        default=False,
        action="store_true")

    parser.add_argument(
        "-s",
        "--seed",
        help="Seed for reproducibility.",
        type=int)

    args = parser.parse_args()

    if args.seed:
        seed = args.seed
    else:
        seed = 7613

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if not args.output_dir:
        args.output_dir = "mixed"

    mix_segmentation(args.input_dir, args.output_dir, args.max_mixes, args.group_contrasts)


if __name__ == "__main__":
    main()
