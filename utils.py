import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import losses


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
            tf.keras.preprocessing.image.save_img(image_name, image)
            tf.keras.preprocessing.image.save_img(mask_name, mask)

        tf.keras.backend.clear_session()
        if i == batches:
            break


def load_files(image_path, mask_path, target_shape=(1920, 2560), classes=1, one_hot_encoded=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_shape)
    image = image / 255.

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, target_shape)

    if classes > 1:
        mask = mask - 1

    if one_hot_encoded:
        mask = tf.cast(mask, dtype=tf.int32)
        mask = tf.one_hot(mask, depth=classes, axis=2, dtype=tf.float32)
        mask = tf.squeeze(mask)

    return image, mask


def load_dataset(path, batch_size=1, target_shape=(1920, 2560), repeat=False, shuffle=False, classes=1, one_hot_encoded=False, seed=1145):
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
    masks_paths = [str(masks_path) for masks_path in masks_paths]
    dataset_files = tf.data.Dataset.from_tensor_slices((images_paths, masks_paths))
    dataset = dataset_files.map(lambda image_path, mask_path: load_files(image_path, mask_path, target_shape, classes, one_hot_encoded))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=int(len(images_paths) * 0.1), seed=seed)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def update_model(model, input_shape):
    model_weights = model.get_weights()
    model_json = json.loads(model.to_json())

    model_json["config"]["layers"][0]["config"]["batch_input_shape"] = [None, *input_shape]
    # model_json["config"]["layers"][1]["config"]["layers"][0]["config"]["batch_input_shape"] = [None, *input_shape]
    model_json["config"]["layers"][0]["config"]["batch_input_shape"] = [None, *input_shape]

    updated_model = tf.keras.models.model_from_json(json.dumps(model_json))
    updated_model.set_weights(model_weights)
    return updated_model


def evaluate(model, images_path, batch_size, input_shape=None, classes=1, one_hot_encoded=False):
    if Path(model).is_file():
        loaded_model = tf.keras.models.load_model(model, custom_objects={"dice_coef_loss": losses.dice_coef_loss, "dice_coef": losses.dice_coef})

        if input_shape:
            loaded_model = update_model(loaded_model, input_shape)

        loaded_model.compile(optimizer=Adam(lr=1e-5), loss=losses.dice_coef_loss, metrics=[losses.dice_coef])

        input_shape = loaded_model.input_shape[1:]
        height, width, channels = input_shape

        evaluate_dataset = load_dataset(images_path, batch_size=batch_size, target_shape=(height, width), classes=classes, one_hot_encoded=one_hot_encoded)
        loss, dice = loaded_model.evaluate(evaluate_dataset)
        print(f"Model {str(Path(model))}")
        print("  - Loss: %.4f" % loss)
        print("  - Dice: %.4f" % dice)
    else:
        models = [model_path for model_path in Path(model).glob("*.h5")]
        models.sort()

        loaded_model = tf.keras.models.load_model(str(models[0]), custom_objects={"dice_coef_loss": losses.dice_coef_loss, "dice_coef": losses.dice_coef})

        if input_shape:
            loaded_model = update_model(loaded_model, input_shape)

        input_shape = loaded_model.input_shape[1:]
        height, width, channels = input_shape

        evaluate_dataset = load_dataset(images_path, batch_size=batch_size, target_shape=(height, width), classes=classes, one_hot_encoded=one_hot_encoded)
        best_model = {}

        for i, model_path in enumerate(models):
            loaded_model = tf.keras.models.load_model(str(model_path), custom_objects={"dice_coef_loss": losses.dice_coef_loss, "dice_coef": losses.dice_coef})

            if input_shape:
                loaded_model = update_model(loaded_model, input_shape)

            loaded_model.compile(optimizer=Adam(lr=1e-5), loss=losses.dice_coef_loss, metrics=[losses.dice_coef])
            loss, dice = loaded_model.evaluate(evaluate_dataset)
            print(f"({i+1}/{len(models)}) Model {model_path.name}")
            print("  - Loss: %.4f" % loss)
            print("  - Dice: %.4f" % dice)

            if "model" in best_model:
                if dice > best_model["dice"]:
                    best_model["model"] = model_path.name
                    best_model["loss"] = loss
                    best_model["dice"] = dice
            else:
                best_model["model"] = model_path.name
                best_model["loss"] = loss
                best_model["dice"] = dice

            tf.keras.backend.clear_session()

        print(f"\nBest model: {Path(model).joinpath(best_model['model'])}")
        print("  - Loss: %.4f" % best_model['loss'])
        print("  - Dice: %.4f" % best_model['dice'])
        return best_model


def predict(model, images_path, batch_size, output_path="predictions", copy_images=False, new_input_shape=None):
    if isinstance(model, str) or isinstance(model, Path):
        loaded_model = tf.keras.models.load_model(str(model), custom_objects={"dice_coef_loss": losses.dice_coef_loss, "dice_coef": losses.dice_coef})
    else:
        loaded_model = model

    if new_input_shape:
        loaded_model = update_model(loaded_model, new_input_shape)

    input_shape = loaded_model.input_shape[1:]
    height, width, channels = input_shape

    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    if Path(images_path).is_dir():
        images = [image_path for image_path in Path(images_path).glob("*.*") if image_path.suffix.lower() in supported_types and not image_path.stem.endswith("_prediction")]
    elif Path(images_path).is_file():
        images = [Path(images_path)]

    if len(images) == 0:
        print(f"No images found at '{images_path}'.")
        return

    images_tensor = np.empty((1, height, width, channels))
    Path(output_path).mkdir(exist_ok=True, parents=True)

    for image_path in images:
        image = tf.io.read_file(str(image_path))
        image = tf.image.decode_jpeg(image, channels=3)
        original_shape = image.shape[:2]
        image = tf.image.resize(image, (height, width))

        images_tensor[0, :, :, :] = image

        prediction = loaded_model.predict(images_tensor, batch_size=batch_size, verbose=1)
        prediction = tf.image.resize(prediction[0], original_shape).numpy()

        prediction[prediction < 0.5] = 0.
        prediction[prediction >= 0.5] = 255.

        cv2.imwrite(os.path.join(output_path, f"{image_path.stem}_{loaded_model.name}_prediction.png"), prediction)

        if copy_images:
            shutil.copyfile(str(image_path), Path(output_path).joinpath(image_path.name))
        tf.keras.backend.clear_session()
