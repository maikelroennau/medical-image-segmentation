import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from losses import dice_coef, dice_coef_loss
from utils import update_model


def predict(model, images_path, batch_size, output_path, copy_images, input_shape=None):
    loaded_model = keras.models.load_model(model, custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})

    if input_shape:
        loaded_model = update_model(loaded_model, input_shape)

    input_shape = loaded_model.input_shape[1:]
    height, width, channels = input_shape

    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    if Path(images_path).is_dir():
        images = [image_path for image_path in Path(images_path).rglob("*.*") if image_path.suffix.lower() in supported_types and not image_path.stem.endswith("_prediction")]
    elif Path(images_path).is_file():
        images = [Path(images_path)]

    images_tensor = np.empty((1, height, width, channels))
    Path(output_path).mkdir(exist_ok=True, parents=True)

    for i, image_path in enumerate(images):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        original_shape = image.shape[:2][::-1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        images_tensor[0, :, :, :] = image

        prediction = loaded_model.predict(images_tensor, batch_size=batch_size, verbose=1)
        prediction = cv2.resize(prediction[0], original_shape)
        if len(prediction.shape) > 2:
            prediction[:, :, 0][prediction[:, :, 0] < 0.5] = 0
            prediction[:, :, 0][prediction[:, :, 0] >= 0.5] = 255

            prediction[:, :, 1][prediction[:, :, 1] == 0.] = 255
            prediction[:, :, 1][prediction[:, :, 1] < 255.] = 0

            output = np.zeros(prediction.shape[:2] + (3,))
            output[:, :, 0:2] = prediction.astype(np.uint8)

            cv2.imwrite(os.path.join(output_path, f"{image_path.stem}_{loaded_model.name}_{i}_prediction.png"), cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2RGB))
            if copy_images:
                shutil.copyfile(str(image_path), Path(output_path).joinpath(image_path.name))
        else:
            prediction[prediction < 0.5] = 0
            prediction[prediction >= 0.5] = 255
            cv2.imwrite(os.path.join(output_path, f"{image_path.stem}_{loaded_model.name}_prediction.png"), prediction)
            if copy_images:
                shutil.copyfile(str(image_path), Path(output_path).joinpath(image_path.name))
        keras.backend.clear_session()


def main():
    parser = argparse.ArgumentParser(description="Predicts using the indicated model on the indicated data.")

    parser.add_argument(
        "-m",
        "--model",
        help="Path to the model to use for prediction.",
        required=True,
        type=str)

    parser.add_argument(
        "-i",
        "--images",
        help="Path to the directory containing the images to predict or to a single image file.",
        required=True,
        type=str)

    parser.add_argument(
        "-o",
        "--output",
        help="Where to save the resulting predictions.",
        type=str)

    parser.add_argument(
        "--input-shape",
        help="Whether to replace the model's input shape with the given input shape. Expects input shape in the following format: `HEIGHTxWIDTH`.",
        type=str)

    parser.add_argument(
        "-b",
        "--batch-size",
        help="Batch size during evaluation.",
        default=1,
        type=int)

    parser.add_argument(
        "-c",
        "--copy-images",
        help="Copy predicted images to output directory. Only effective when `--output` is provided.",
        default=True,
        action="store_true")

    parser.add_argument(
        "-gpu",
        "--gpu",
        help="What GPU to use. Pass `-1` to use CPU.",
        default=0)

    parser.add_argument(
        "--memory-growth",
        help="Whether or not to allow GPU memory growth.",
        default=False,
        action="store_true")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = str(args.memory_growth).lower()

    if args.input_shape:
        input_shape = args.input_shape.lower().split("x")
        input_shape = (int(input_shape[0]), int(input_shape[1]), (3))
    else:
        input_shape = None

    if not args.output and Path(args.images).is_file():
        output = Path(args.images).parent
    elif not args.output and not Path(args.images).is_file():
        output = Path(args.images)
    else:
        output = args.output

    if args.output != output:
        args.copy_images = False

    predict(args.model, args.images, args.batch_size, output, args.copy_images, input_shape)


if __name__ == "__main__":
    main()
