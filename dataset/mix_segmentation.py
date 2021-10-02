import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.data_io import list_files, load_image, load_mask


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
    groups = { f"{i}qt": { "images": [], "masks": [] } for i in range(1, 3) }

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


def mix_segmentation(input, output, max_mixes, group_by_contrast=False, blend=False, suffix=""):
    if isinstance(input, str) or isinstance(input, Path):
        images_paths, masks_paths = list_files(input, validate_masks=True)

        if group_by_contrast:
            contrast_groups = get_contrast_groups(images_paths, masks_paths)
            for group in contrast_groups.keys():
                mix_segmentation(
                    (contrast_groups[group]["images"], contrast_groups[group]["masks"]),
                    output=output,
                    max_mixes=max_mixes,
                    group_by_contrast=False,
                    blend=blend)
            return
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

                # Add new nucleus and NORs mask to the target mask
                target_mask += mask_roi.copy()

                # Create new image containing only the RGB pixels of the new nucleus and NORs
                image_roi = np.zeros_like(target_image)
                image_roi[new_y:new_y+h, new_x:new_x+w] = source_image_rect_roi

                # Obtain the contour that will be used for bleding the nucleus and NORs to the target image
                blend_contours, hierarchy = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                blend_roi = np.zeros_like(target_image)
                cv2.drawContours(blend_roi, contours=blend_contours, contourIdx=-1, color=[255, 255, 255], thickness=cv2.FILLED)

                # Add nucleus and NORs to the target image then blur it
                target_image = np.where(blend_roi == np.array([255, 255, 255]), image_roi, target_image)
                blured_image_roi = cv2.GaussianBlur(target_image.copy(), (15, 15), 0)

                # Copy only the blured pixels that belong to the contour and replace it in the target image
                if blend:
                    blend_roi = np.zeros_like(target_image)
                    cv2.drawContours(blend_roi, contours=blend_contours, contourIdx=-1, color=[255, 255, 255], thickness=7)
                    target_image = np.where(blend_roi == np.array([255, 255, 255]), blured_image_roi, target_image)

            if i == max_mixes - 1:
                break

        cv2.imwrite(f"{images_output.joinpath(Path(image_path).stem)}_m{i}{suffix}{Path(image_path).suffix}", cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"{masks_output.joinpath(Path(mask_path).stem)}_m{i}{suffix}{Path(mask_path).suffix}", target_mask)


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
        "-b",
        "--blend",
        help="Blend segmented areas using Gaussian blur.",
        default=False,
        action="store_true")

    parser.add_argument(
        "--suffix",
        help="Suffix for the generated images.",
        default="",
        type=str)

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

    if len(args.suffix) > 0:
        args.suffix = f"_{args.suffix}"

    mix_segmentation(args.input_dir, args.output_dir, args.max_mixes, args.group_contrasts, args.blend, args.suffix)


if __name__ == "__main__":
    main()
