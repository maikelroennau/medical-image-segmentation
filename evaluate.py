import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


def load_images_and_masks(path, batch_size=16, target_size=(1920, 2560), seed=1145):
    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    images_paths = [image_path for image_path in Path(path).joinpath("images").glob("*.*") if image_path.suffix.lower() in supported_types and not image_path.stem.endswith("_prediction")]
    masks_paths = [mask_path for mask_path in Path(path).joinpath("masks").glob("*.*") if mask_path.suffix.lower() in supported_types and not mask_path.stem.endswith("_prediction")]

    assert len(images_paths) == len(masks_paths), f"Different quantity of images ({len(images_paths)}) and masks ({len(masks_paths)})"

    images_paths.sort()
    masks_paths.sort()

    for image, mask in zip(images_paths, masks_paths):
        assert image.stem.lower() == mask.stem.lower(), f"Image and mask do not correspond: {image.name} <==> {image.name}"

    height, width = target_size
    images = np.zeros((len(images_paths), height, width, 3), dtype="float32")
    masks = np.zeros((len(masks_paths), height, width, 1), dtype="float32")

    for i, (image, mask) in enumerate(zip(images_paths, masks_paths)):
        images[i, :, :, :] = keras.preprocessing.image.load_img(image, target_size=target_size)
        masks[i, :, :, 0] = keras.preprocessing.image.load_img(mask, target_size=target_size, color_mode="grayscale")

    np.random.seed(seed)
    randomize = np.random.permutation(len(images))
    images = np.asarray(images)[randomize]
    masks = np.asarray(masks)[randomize]

    print(f"Loaded from '{path}'")
    print(f"  - Images: {len(images)}")
    print(f"  - Masks: {len(masks)}")
    
    return images, masks


def dice_coef(y_true, y_pred, smooth=1.):
    intersection = keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = keras.backend.sum(y_true, axis=[1,2,3]) + keras.backend.sum(y_pred, axis=[1,2,3])
    return keras.backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def main(model, images_path="dataset/test/", test_all=False):
    if not test_all:
        loaded_model = keras.models.load_model(model, custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
        input_shape = loaded_model.input_shape[1:]
        height, width, channels = input_shape

        images, masks = load_images_and_masks(images_path, target_size=(height, width))
        loaded_model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
        loss, dice = loaded_model.evaluate(images, masks, batch_size=1)
        print(f"Model {Path(model).name}")
        print("  - Loss: %.2f" % loss)
        print("  - Dice: %.2f" % dice)
    else:
        models = [model_path for model_path in Path(model).glob("epoch*.h5")]

        loaded_model = keras.models.load_model(str(models[0]), custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
        input_shape = loaded_model.input_shape[1:]
        height, width, channels = input_shape

        images, masks = load_images_and_masks(images_path, target_size=(height, width))

        for model_path in models:
            loaded_model = keras.models.load_model(str(model_path), custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
            loaded_model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
            loss, dice = loaded_model.evaluate(images, masks, batch_size=1)
            print(f"Model {model_path.name}")
            print("  - Loss: %.2f" % loss)
            print("  - Dice: %.2f" % dice)
            keras.backend.clear_session()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        print("Please provide the model path")
