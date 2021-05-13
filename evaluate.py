import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


def load_files(image_path, target_shape=(1920, 2560)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.decode_image(image, channels=3)
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


def load_dataset(path, batch_size=32, target_shape=(1920, 2560), seed=1145):
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

    dataset = dataset.shuffle(buffer_size=len(images_paths), seed=seed)
    # dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset


def dice_coef(y_true, y_pred, smooth=1.):
    intersection = keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = keras.backend.sum(y_true, axis=[1,2,3]) + keras.backend.sum(y_pred, axis=[1,2,3])
    return keras.backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def main(model, images_path="dataset/test/", test_all=False):
    seed = 1145
    tf.random.set_seed(seed)
    np.random.seed(seed)

    if not bool(test_all):
        loaded_model = keras.models.load_model(model, custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
        loaded_model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

        input_shape = loaded_model.input_shape[1:]
        height, width, channels = input_shape

        evaluate_dataset = load_dataset(images_path, batch_size=1, target_shape=(height, width))
        loss, dice = loaded_model.evaluate(evaluate_dataset)
        print(f"Model {Path(model).name}")
        print("  - Loss: %.4f" % loss)
        print("  - Dice: %.4f" % dice)
    else:
        models = [model_path for model_path in Path(model).glob("*.h5")]

        loaded_model = keras.models.load_model(str(models[0]), custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
        input_shape = loaded_model.input_shape[1:]
        height, width, channels = input_shape

        evaluate_dataset = load_dataset(images_path, batch_size=1, target_shape=(height, width))

        for i, model_path in enumerate(models):
            loaded_model = keras.models.load_model(str(model_path), custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
            loaded_model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
            loss, dice = loaded_model.evaluate(evaluate_dataset)
            print(f"({i+1}/{len(models)}) Model {model_path.name}")
            print("  - Loss: %.4f" % loss)
            print("  - Dice: %.4f" % dice)
            keras.backend.clear_session()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        print("Please provide the model path")
