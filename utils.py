import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm


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
        if max_batches > len(dataset):
            batches = len(dataset)
        else:
            batches = max_batches
    else:
        batches = len(dataset)

    for i, batch in tqdm(enumerate(dataset), total=batches):
        for j, (image, mask) in enumerate(zip(batch[0], batch[1])):
            image_name = str(images_path.joinpath(f"batch_{i}_{j}.jpg"))
            mask_name = str(masks_path.joinpath(f"batch_{i}_{j}.png"))
            keras.preprocessing.image.save_img(image_name, image)
            keras.preprocessing.image.save_img(mask_name, mask)

        keras.backend.clear_session()
        if i + 1 == batches:
            break


def load_files(image_path, target_shape=(1920, 2560)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.image.resize(image, target_shape)

    mask_path = tf.strings.regex_replace(image_path, "images", "masks")
    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    for supported_type in supported_types:
        mask_path = tf.strings.regex_replace(mask_path, supported_type, ".png")

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)
    mask = tf.image.resize(mask, target_shape)
    return image, mask


def load_dataset(path, batch_size=32, target_shape=(1920, 2560), repeat=False, shuffle=False, seed=1145):
    images_path = Path(path).joinpath("images")
    masks_path = Path(path).joinpath("masks")

    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    images_paths = [image_path for image_path in images_path.glob("*.*") if image_path.suffix.lower() in supported_types and not image_path.stem.endswith("_prediction")]
    masks_paths = [mask_path for mask_path in masks_path.glob("*.*") if mask_path.suffix.lower() in supported_types and not mask_path.stem.endswith("_prediction")]

    images_paths.sort()
    masks_paths.sort()

    assert len(images_paths) == len(masks_paths), f"Different quantity of images ({len(images_paths)}) and masks ({len(masks_paths)})"

    for image_path, mask_path in zip(images_paths, masks_paths):
        assert image_path.stem.lower() == mask_path.stem.lower(), f"Image and mask do not correspond: {image_path.name} <==> {mask_path.name}"

    print(f"Dataset '{str(images_path.parent)}' contains {len(images_paths)} images and masks.")

    images_paths = [str(image_path) for image_path in images_paths]
    dataset_files = tf.data.Dataset.from_tensor_slices(images_paths)
    dataset = dataset_files.map(lambda x: load_files(x, target_shape))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=int(len(images_paths) * 0.1), seed=seed)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset

########
########

def update_model(model, input_shape):
    model_weights = model.get_weights()
    model_json = json.loads(model.to_json())

    model_json["config"]["layers"][0]["config"]["batch_input_shape"] = [None, *input_shape]
    # model_json["config"]["layers"][1]["config"]["layers"][0]["config"]["batch_input_shape"] = [None, *input_shape]
    model_json["config"]["layers"][0]["config"]["batch_input_shape"] = [None, *input_shape]

    updated_model = keras.models.model_from_json(json.dumps(model_json))
    updated_model.set_weights(model_weights)
    return updated_model
