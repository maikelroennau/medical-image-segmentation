import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from utils import losses
from utils.data_io import load_dataset
from utils.postprocess import post_process


CUSTOM_OBJECTS = {
    "dice_coef": losses.dice_coef,
    "dice_coef_loss": losses.dice_coef_loss,
    "jaccard_index": losses.jaccard_index,
    "jaccard_index_loss": losses.jaccard_index_loss,
    "weighted_categorical_crossentropy": losses.weighted_categorical_crossentropy,
    "categorical_focal_loss": losses.categorical_focal_loss,
    "unified_focal_loss": losses.unified_focal_loss,
    "categorical_crossentropy": sm.losses.categorical_crossentropy,
    "categorical_crossentropy_plus_dice_loss": sm.losses.cce_dice_loss,
    "focal_loss_plus_dice_loss": sm.losses.categorical_focal_dice_loss,
    "focal_loss": sm.losses.categorical_focal_loss,
    "f1-score": sm.metrics.f1_score,
    "iou_score": sm.metrics.iou_score,
}

METRICS = [
    sm.metrics.f1_score,
    sm.metrics.iou_score
]

def update_model(model, input_shape):
    model_weights = model.get_weights()
    model_json = json.loads(model.to_json())

    model_json["config"]["layers"][0]["config"]["batch_input_shape"] = [None, *input_shape]
    model_json["config"]["layers"][0]["config"]["batch_input_shape"] = [None, *input_shape]

    updated_model = tf.keras.models.model_from_json(json.dumps(model_json))
    updated_model.set_weights(model_weights)
    return updated_model


def evaluate(model, images_path, batch_size, input_shape=None, classes=1, one_hot_encoded=False):
    if Path(model).is_file():
        loaded_model = tf.keras.models.load_model(model, custom_objects=CUSTOM_OBJECTS)

        if input_shape:
            loaded_model = update_model(loaded_model, input_shape)

        loaded_model.compile(optimizer=Adam(learning_rate=1e-5), loss=sm.losses.cce_dice_loss, metrics=[METRICS])

        input_shape = loaded_model.input_shape[1:]
        height, width, channels = input_shape

        evaluate_dataset = load_dataset(images_path, batch_size=batch_size, target_shape=(height, width), classes=classes, one_hot_encoded=one_hot_encoded)
        evaluation_metrics = loaded_model.evaluate(evaluate_dataset)

        print(f"Model {str(model)}")
        print("  - Loss: %.4f" % evaluation_metrics[0])

        model = Path(model)
        model_metrics = {}
        model_metrics["model"] = str(model)
        model_metrics["loss"] = evaluation_metrics[0]

        for i, evaluation_metric in enumerate(evaluation_metrics[1:]):
            metric = METRICS[i] if isinstance(METRICS[i], str) else METRICS[i].__name__
            print(f"  - {metric}: {np.round(evaluation_metric, 4)}")
            model_metrics[metric] = evaluation_metric

        return model_metrics
    else:
        models = [model_path for model_path in Path(model).glob("*.h5")]
        models.sort()

        if len(models) > 0:
            loaded_model = tf.keras.models.load_model(str(models[0]), custom_objects=CUSTOM_OBJECTS)
        else:
            print("No models found")
            return None, None

        if input_shape:
            loaded_model = update_model(loaded_model, input_shape)

        input_shape = loaded_model.input_shape[1:]
        height, width, channels = input_shape

        evaluate_dataset = load_dataset(images_path, batch_size=batch_size, target_shape=(height, width), classes=classes, one_hot_encoded=one_hot_encoded)

        if models[0].parent.joinpath("train_config.json").is_file():
            with open(str(models[0].parent.joinpath("train_config.json")), "r") as config_file:
                epochs = json.load(config_file)["epochs"]
        else:
            epochs = int(str(models[-1]).split("AgNOR")[1].split("_")[1][1:])

        best_model = {}
        models_metrics = {}
        models_metrics["test_loss"] = [0] * epochs
        for i in range(len(METRICS)):
            metric = METRICS[i] if isinstance(METRICS[i], str) else METRICS[i].__name__
            models_metrics[f"test_{metric}"] = [0] * len(models_metrics["test_loss"])

        for i, model_path in enumerate(models):
            loaded_model = tf.keras.models.load_model(str(model_path), custom_objects=CUSTOM_OBJECTS)

            if input_shape:
                loaded_model = update_model(loaded_model, input_shape)

            loaded_model.compile(optimizer=Adam(learning_rate=1e-5), loss=sm.losses.cce_dice_loss, metrics=[METRICS])
            evaluation_metrics = loaded_model.evaluate(evaluate_dataset)
            print(f"Model {str(model_path)}")
            print(f"  - Loss: {np.round(evaluation_metrics[0], 4)}")
            for i, evaluation_metric in enumerate(evaluation_metrics[1:]):
                metric = METRICS[i] if isinstance(METRICS[i], str) else METRICS[i].__name__
                print(f"  - {metric}: {np.round(evaluation_metric, 4)}")

            # Add model metrics to dict
            if len(str(model_path).split("AgNOR")[1].split("_")[1:]) > 0:
                models_metrics["test_loss"][int(str(model_path).split("AgNOR")[1].split("_")[1][1:])-1] = evaluation_metrics[0]
            else:
                models_metrics["test_loss"][-1] = evaluation_metrics[0]
            for i, evaluation_metric in enumerate(evaluation_metrics[1:]):
                metric = METRICS[i] if isinstance(METRICS[i], str) else METRICS[i].__name__
                if len(str(model_path).split("AgNOR")[1].split("_")[1:]) > 0:
                    models_metrics[f"test_{metric}"][int(str(model_path).split("AgNOR")[1].split("_")[1][1:])-1] = evaluation_metric
                else:
                    models_metrics[f"test_{metric}"][-1] = evaluation_metric

            # Check for the best model
            if "model" in best_model:
                if evaluation_metrics[1] > best_model["f1-score"]:
                    best_model["model"] = model_path.name
                    best_model["loss"] = evaluation_metrics[0]
                    for i, evaluation_metric in enumerate(evaluation_metrics[1:]):
                        metric = METRICS[i] if isinstance(METRICS[i], str) else METRICS[i].__name__
                        best_model[metric] = evaluation_metric
            else:
                best_model["model"] = model_path.name
                best_model["loss"] = evaluation_metrics[0]
                for i, evaluation_metric in enumerate(evaluation_metrics[1:]):
                    metric = METRICS[i] if isinstance(METRICS[i], str) else METRICS[i].__name__
                    best_model[metric] = evaluation_metric

            tf.keras.backend.clear_session()

        print(f"\nBest model: {best_model['model']}")
        print(f"  - Loss: {np.round(best_model['loss'], 4)}")
        for metric, value in list(best_model.items())[2:]:
            print(f"  - {metric}: {np.round(value, 4)}")

        return best_model, models_metrics


def predict(model, images_path, batch_size, output_path="predictions", copy_images=False, new_input_shape=None, normalize=False, single_dir=False, verbose=1):
    if isinstance(model, str) or isinstance(model, Path):
        model = Path(model)
        if model.is_file():
            loaded_model = tf.keras.models.load_model(str(model), custom_objects=CUSTOM_OBJECTS)
        elif model.is_dir():
            models = [model_path for model_path in model.glob("*.h5")]
            if len(models) > 0:
                print(f"No model(s) found at {str(model)}")
                for model_path in tqdm(models):
                    predict(
                        model_path,
                        images_path,
                        batch_size,
                        output_path=str(Path(output_path)) if single_dir else str(Path(output_path).joinpath(model_path.name)),
                        copy_images=copy_images,
                        new_input_shape=new_input_shape,
                        normalize=normalize,
                        single_dir=single_dir,
                        verbose=0)
            return
    elif model != None:
        loaded_model = model
    else:
        print("No model(s) found")
        return

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
        image = tf.image.decode_png(image, channels=3)
        original_shape = image.shape[:2]
        image = tf.image.resize(image, (height, width), method="nearest")

        if normalize:
            image = tf.cast(image, dtype=tf.float32)
            image = image / 255.

        images_tensor[0, :, :, :] = image

        prediction = loaded_model.predict(images_tensor, batch_size=batch_size, verbose=verbose)
        prediction = tf.image.resize(prediction[0], original_shape, method="nearest").numpy()

        pixel_intensity = 127
        prediction[:, :, 0] = np.where(
            np.logical_and(prediction[:, :, 0] > prediction[:, :, 1], prediction[:, :, 0] > prediction[:, :, 2]), pixel_intensity, 0)
        prediction[:, :, 1] = np.where(
            np.logical_and(prediction[:, :, 1] > prediction[:, :, 0], prediction[:, :, 1] > prediction[:, :, 2]), pixel_intensity, 0)
        prediction[:, :, 2] = np.where(
            np.logical_and(prediction[:, :, 2] > prediction[:, :, 0], prediction[:, :, 2] > prediction[:, :, 1]), pixel_intensity, 0)

        if prediction.shape[-1] == 2:
            prediction_reshaped = np.zeros(tuple(prediction.shape[:2]) + (3,), dtype=np.uint8)
            prediction_reshaped[:, :, :2] = prediction
            prediction = prediction_reshaped

        prediction, _ = post_process(prediction)

        if single_dir:
            output_image_path = os.path.join(output_path, f"{model.stem.split('_l')[0]}_{image_path.stem}_prediction.png")
        else:
            output_image_path = os.path.join(output_path, f"{image_path.stem}_{loaded_model.name}_prediction.png")

        cv2.imwrite(output_image_path, cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB))

        if copy_images:
            shutil.copyfile(str(image_path), Path(output_path).joinpath(image_path.name))
        tf.keras.backend.clear_session()


def compute_classes_distribution(dataset, batches=1, plot=True, figsize=(20, 10), output=".", get_as_weights=False, classes=["Background", "Nucleus", "NOR"]):
    class_occurence = []
    batch_size = None

    for i, batch in enumerate(dataset):
        if i == batches:
            batch_size = batch[0].shape[0]
            break

        for mask in batch[1]:
            class_count = []
            for class_index in range(mask.shape[-1]):
                class_count.append(tf.math.reduce_sum(mask[:, :, class_index]))
            class_occurence.append(class_count)

    class_occurence = tf.convert_to_tensor(class_occurence, dtype=tf.int64)
    class_distribution = tf.reduce_sum(class_occurence, axis=0) / (batches * batch_size)
    class_distribution = class_distribution.numpy()
    class_distribution = class_distribution * 100 / (dataset.element_spec[0].shape[1] * dataset.element_spec[0].shape[2])
    if get_as_weights:
        class_distribution = (100 - class_distribution) / 100
        class_distribution = np.round(class_distribution, 2)

    distribution = {}
    for occurence, class_name in zip(class_distribution, classes):
        distribution[class_name] = float(occurence)

    if plot:
        import pandas as pd

        output_path = Path(output)
        output_path.mkdir(exist_ok=True, parents=True)

        class_occurence = class_occurence.numpy()
        df = pd.DataFrame(class_occurence, columns=classes)
        class_weights_figure = df.plot.bar(stacked=True, figsize=figsize)
        class_weights_figure.set(xlabel="Image instance", ylabel="Number of pixels per class", title="Image class distribution")
        class_weights_figure.axes.set_xticks([])
        class_weights_figure = class_weights_figure.get_figure()
        class_weights_figure.savefig(output_path.joinpath("classes_distribution.png"))

        df = pd.DataFrame(distribution.values()).transpose()
        df.columns = classes
        class_weights_figure = df.plot.bar(stacked=True, figsize=(10, 10))
        class_weights_figure.set(ylabel="Number of pixels per class", title="Dataset class distribution")
        class_weights_figure = class_weights_figure.get_figure()
        class_weights_figure.savefig(output_path.joinpath("classes_distribution_dataset.png"))

    return distribution
