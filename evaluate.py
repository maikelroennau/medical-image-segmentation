import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from losses import dice_coef, dice_coef_loss
from utils import load_dataset, update_model


def evaluate(model, images_path, batch_size, input_shape=None):
    if Path(model).is_file():
        loaded_model = keras.models.load_model(model, custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})

        if input_shape:
            loaded_model = update_model(loaded_model, input_shape)

        loaded_model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

        input_shape = loaded_model.input_shape[1:]
        height, width, channels = input_shape

        evaluate_dataset = load_dataset(images_path, batch_size=batch_size, target_shape=(height, width))
        loss, dice = loaded_model.evaluate(evaluate_dataset)
        print(f"Model {Path(model).name}")
        print("  - Loss: %.4f" % loss)
        print("  - Dice: %.4f" % dice)
    else:
        models = [model_path for model_path in Path(model).glob("*.h5")]
        models.sort()

        loaded_model = keras.models.load_model(str(models[0]), custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})

        if input_shape:
            loaded_model = update_model(loaded_model, input_shape)

        input_shape = loaded_model.input_shape[1:]
        height, width, channels = input_shape

        evaluate_dataset = load_dataset(images_path, batch_size=batch_size, target_shape=(height, width))
        best = {}

        for i, model_path in enumerate(models):
            loaded_model = keras.models.load_model(str(model_path), custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})

            if input_shape:
                loaded_model = update_model(loaded_model, input_shape)

            loaded_model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
            loss, dice = loaded_model.evaluate(evaluate_dataset)
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


def main():
    parser = argparse.ArgumentParser(description="Predicts using the indicated model on the indicated data.")

    parser.add_argument(
        "-m",
        "--model",
        help="Path to the model to evaluate. If path is a directory, will evaluate all models within it.",
        required=True,
        type=str)

    parser.add_argument(
        "-i",
        "--images",
        help="Path to the directory containing the images to predict or to a single image file.",
        required=True,
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
        "-gpu",
        "--gpu",
        help="What GPU to use. Pass `-1` to use CPU.",
        default=0)

    parser.add_argument(
        "--memory-growth",
        help="Whether or not to allow GPU memory growth.",
        default=False,
        action="store_true")

    parser.add_argument(
        "--seed",
        help="Seed for reproducibility.",
        default=1145,
        type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = str(args.memory_growth).lower()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    if args.input_shape:
        input_shape = args.input_shape.lower().split("x")
        input_shape = (int(input_shape[0]), int(input_shape[1]), (3))
    else:
        input_shape = None

    evaluate(args.model, args.images, args.batch_size, input_shape)


if __name__ == "__main__":
    main()
