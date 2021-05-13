import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


def data_loader(path, batch_size=32, target_shape=(1920, 2560), seed=1145):
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    images = image_datagen.flow_from_directory(
        directory=Path(path),
        target_size=target_shape,
        classes=["images"],
        class_mode=None,
        color_mode="rgb",
        batch_size=batch_size,
        seed=seed
    )

    masks = mask_datagen.flow_from_directory(
        directory=Path(path),
        target_size=target_shape,
        classes=["masks"],
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        seed=seed
    )

    return images, masks


def data_generator(path, batch_size=32, target_shape=(1920, 2560), seed=1145):
    images, masks = data_loader(path, batch_size, target_shape, seed)
    for images, masks in zip(images, masks):
        yield (images, masks)


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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if not bool(test_all):
        loaded_model = keras.models.load_model(model, custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
        loaded_model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

        input_shape = loaded_model.input_shape[1:]
        height, width, channels = input_shape

        evaluate_dataset = data_generator(images_path, batch_size=1, target_shape=(height, width))
        loss, dice = loaded_model.evaluate(evaluate_dataset, steps=10)
        print(f"Model {Path(model).name}")
        print("  - Loss: %.4f" % loss)
        print("  - Dice: %.4f" % dice)
    else:
        models = [model_path for model_path in Path(model).glob("*.h5")]
        models.sort()

        loaded_model = keras.models.load_model(str(models[0]), custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
        input_shape = loaded_model.input_shape[1:]
        height, width, channels = input_shape

        evaluate_dataset = data_generator(images_path, batch_size=1, target_shape=(height, width))
        best = {}

        for i, model_path in enumerate(models):
            loaded_model = keras.models.load_model(str(model_path), custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
            loaded_model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
            loss, dice = loaded_model.evaluate(evaluate_dataset, steps=10)
            print(f"({i+1}/{len(models)}) Model {model_path.name}")
            print("  - Loss: %.4f" % loss)
            print("  - Dice: %.4f" % dice)

            if "model" in best:
                if dice > best["dice"]:
                    best["model"] = model_path.name
                    best["loss"] = loss
                    best["dice"] = dice
            else:
                best["model"] = model_path.name
                best["loss"] = loss
                best["dice"] = dice

            keras.backend.clear_session()

        print(f"\nBest model: {best['model']}")
        print("  - Loss: %.4f" % best['loss'])
        print("  - Dice: %.4f" % best['dice'])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        print("Please provide the model path")
