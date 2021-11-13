from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage.io import imread


def write_dataset(dataset, output_path="dataset_visualization", max_batches=None, same_dir=False):
    output = Path(output_path)
    images_path = output.joinpath("images")

    if same_dir:
        masks_path = output.joinpath("images")
    else:
        masks_path = output.joinpath("masks")

    images_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)

    if max_batches:
        try:
            if max_batches > len(dataset):
                batches = len(dataset)
            else:
                batches = max_batches
        except Exception:
            batches = max_batches
    else:
        batches = len(dataset)

    for i, batch in tqdm(enumerate(dataset), total=batches):
        for j, (image, mask) in enumerate(zip(batch[0], batch[1])):
            image_name = str(images_path.joinpath(f"batch_{i}_{j}.png"))
            mask_name = str(masks_path.joinpath(f"batch_{i}_{j}.png"))
            tf.keras.preprocessing.image.save_img(image_name, image)
            if mask.shape[-1] == 2:
                mask_reshaped = np.zeros(tuple(mask.shape[:2]) + (3,))
                mask_reshaped[:, :, :2] = mask.numpy()
                mask = tf.convert_to_tensor(mask_reshaped)
            tf.keras.preprocessing.image.save_img(mask_name, mask * 127, scale=False)

        tf.keras.backend.clear_session()
        if i == batches:
            break


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
            assert image_path.stem.lower() == mask_path.stem.lower(), f"Image and mask do not correspond: {image_path.name} <==> {mask_path.name}"

    print(f"Dataset '{str(images_path.parent)}' contains {len(images_paths)} images and masks.")

    images_paths = [str(image_path) for image_path in images_paths]
    masks_paths = [str(masks_path) for masks_path in masks_paths]
    return images_paths, masks_paths


def load_files(image_path, mask_path, target_shape=(1920, 2560), classes=1, one_hot_encoded=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    if image.shape != target_shape:
        image = tf.image.resize(image, target_shape, method="nearest")
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    if mask.shape != target_shape:
        mask = tf.image.resize(mask, target_shape, method="nearest")

    if one_hot_encoded:
        mask = tf.cast(mask, dtype=tf.int32)
        mask = tf.one_hot(mask, depth=classes, axis=2, dtype=tf.int32)
        mask = tf.squeeze(mask)

    mask = tf.cast(mask, dtype=tf.float32)

    return image, mask


def load_dataset(path, batch_size=1, target_shape=(1920, 2560), repeat=False, shuffle=False, classes=1, one_hot_encoded=False, validate_masks=False, seed=None):
    if validate_masks:
        images_paths, masks_paths = list_files(path, validate_masks=validate_masks)
        dataset = tf.data.Dataset.from_tensor_slices((images_paths, masks_paths))
    else:
        images_path = Path(path).joinpath("images").joinpath("*.*")
        masks_path = Path(path).joinpath("masks").joinpath("*.*")

        images_paths = tf.data.Dataset.list_files(str(images_path), shuffle=True)
        masks_paths = tf.data.Dataset.list_files(str(masks_path), shuffle=True)

        assert len(images_paths) > 0, f"No images found at '{images_path}'."
        assert len(masks_paths) > 0, f"No masks found at '{masks_path}'."

        dataset = tf.data.Dataset.zip((images_paths, masks_paths))
        print(f"Dataset '{str(images_path.parent)}' contains {len(dataset)} images and masks.")

    dataset = dataset.map(lambda image_path, mask_path: load_files(image_path, mask_path, target_shape, classes, one_hot_encoded))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * batch_size)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def load_image(image_path):
    return imread(str(image_path))


def load_mask(mask_path):
    return load_image(mask_path)
