import datetime
import json
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import imgviz
import numpy as np
import pandas as pd
import PIL.Image
import segmentation_models as sm
import tensorflow as tf
from tqdm import tqdm

from utils import contour_analysis
from utils.data import (list_files, load_dataset, load_image, one_hot_encode,
                        reset_class_values)
from utils.model import METRICS, get_model_input_shape, load_model


def evaluate_from_files(
    ground_truth_path: str,
    predictions_path: str,
    classes: Optional[int] = None) -> dict:
    """Evaluate prediction using the metrics defined in `utils.model.METRICS`.

    Args:
        ground_truth_path (str): The path to the directory containing the ground truth masks.
        predictions_path (str): The path to the directory containing the predicted masks.
        classes (Optional[int], optional): The number of classes in the data. If `None`, will infer from the ground truth data on an instance basis.

    Raises:
        FileNotFoundError: In case the `predictions_path` does not exists.
        FileNotFoundError: In case the `ground_truth_path` does not exists.
        ValueError: If the number of predictions does not match the number of ground truth masks.

    Returns:
        dict: A dictionary with the metrics score for every evaluation.
    """
    ground_truth = list_files(ground_truth_path, as_numpy=True)
    predictions = list_files(predictions_path, as_numpy=True)

    if len(ground_truth) == 0:
        raise FileNotFoundError(f"No ground truth files were found at `{ground_truth}`.")
    if len(predictions) == 0:
        raise FileNotFoundError(f"No prediction files were found at `{predictions}`.")
    if len(ground_truth) != len(predictions):
        raise ValueError(
            f"The number of ground truth and prediction files do not not match: '{len(ground_truth)}' != '{len(predictions)}'")

    classes_undefined = True if classes is None else False
    metrics = { metric.name: [] for metric in METRICS }
    
    for ground_truth_file_path, prediction_file_path in tqdm(zip(ground_truth, predictions), total=len(ground_truth)):
        ground_truth = load_image(ground_truth_file_path, as_gray=True)
        prediction = load_image(prediction_file_path, as_gray=True)

        if classes_undefined:
            classes = np.unique(ground_truth).size

        ground_truth = one_hot_encode(ground_truth, classes=classes, as_numpy=True)
        ground_truth = ground_truth.reshape((1,) + ground_truth.shape).astype(np.float32)

        prediction = one_hot_encode(prediction, classes=classes, as_numpy=True)
        prediction = prediction.reshape((1,) + prediction.shape).astype(np.float32)

        for metric in METRICS:
            metrics[metric.name].append(metric(ground_truth, prediction).numpy())

    print("Evaluation results:")
    print(f"  - Number of images: {len(predictions)}")
    for metric in METRICS:
        print(f"  - {metric.name}")
        print(f"    - Mean: " + str(np.round(np.mean(metrics[metric.name]), 4)))
        print(f"    - STD.: " + str(np.round(np.std(metrics[metric.name]), 4)))

    return metrics


def evaluate(
    models_paths: List[str],
    images_path: str,
    batch_size: Optional[int] = 1,
    classes: Optional[int] = 3,
    one_hot_encoded: Optional[bool] = True,
    input_shape: Optional[tuple] = None,
    loss_function: Optional[sm.losses.Loss] = sm.losses.cce_dice_loss,
    model_name: Optional[str] = "AgNOR") -> Tuple[dict, dict]:
    """Evaluates a list of `tf.keras.Model` objects.

    Args:
        models_paths (List[str]): A list of paths of `tf.keras.Models` to be evaluated.
        images_path (str): The path to the directory containing the `images` and `masks` subdirectories.
        batch_size (Optional[int], optional): The number of images per batch. Defaults to 1.
        classes (Optional[int], optional): The number of classes. Affects the one hot encoding. Defaults to 3.
        one_hot_encoded (Optional[bool], optional): Whether or not to one hot encode the masks. Defaults to True.
        input_shape (Optional[tuple], optional): The input shape to use to evaluate the model. If `None`, uses the model's default input shape. In the format `(HEIGHT, WIDTH, CHANNELS)`. Defaults to None.
        loss_function (Optional[sm.losses.Loss], optional): The loss function to be used to evaluate the mode. Defaults to sm.losses.cce_dice_loss.
        model_name (Optional[str]): The name of the model. Defaults to AgNOR.

    Returns:
        Tuple[dict, dict]: A tuple of dictionaries where the first contain the evaluation of the best model, and the second the evaluation of all models.
    """
    if Path(models_paths).is_file():
        models_paths = [models_paths]
    elif Path(models_paths).is_dir():
        models_paths = [str(path) for path in Path(models_paths).glob("*.h5")]
    else:
        raise FileNotFoundError(f"No file models were found at `{models_paths}`.")

    models_paths.sort()
    evaluate_dataset = load_dataset(
        images_path, batch_size=batch_size, shape=input_shape[:2], classes=classes, mask_one_hot_encoded=one_hot_encoded)

    config_file_path = f"train_config_{Path(models_paths[0]).parent.name}.json"
    if Path(models_paths[0]).parent.joinpath(config_file_path).is_file():
        with open(str(Path(models_paths[0]).parent.joinpath(config_file_path)), "r") as config_file:
            epochs = json.load(config_file)["epochs"]
    else:
        epochs = re.search("_e\d{3}", models_paths[-1]).group()
        epochs = int(re.search("\d{3}", epochs).group())

    best_model = {}
    models_metrics = {}
    models_metrics["test_loss"] = [0] * epochs
    for i in range(len(METRICS)):
        metric = METRICS[i] if isinstance(METRICS[i], str) else METRICS[i].__name__
        models_metrics[f"test_{metric}"] = [0] * len(models_metrics["test_loss"])

    for i, model_path in enumerate(models_paths):
        model = load_model(model_path=model_path, input_shape=input_shape, loss_function=loss_function)
        evaluation_metrics = model.evaluate(evaluate_dataset, return_dict=True)

        print(f"Model {str(model_path)}")
        print(f"  - Loss: {np.round(evaluation_metrics['loss'], 4)}")

        for metric, value in evaluation_metrics.items():
            if metric != "loss":
                print(f"  - {metric}: {np.round(value, 4)}")

        # Add model metrics to dict
        if len(model_path.split(model_name)[1].split("_")[1:]) > 0:
            models_metrics["test_loss"][int(model_path.split(model_name)[1].split("_")[1][1:])-1] = evaluation_metrics["loss"]
        else:
            models_metrics["test_loss"][-1] = evaluation_metrics["loss"]

        for metric, value in evaluation_metrics.items():
            if metric != "loss":
                if len(model_path.split(model_name)[1].split("_")[1:]) > 0:
                    models_metrics[f"test_{metric}"][int(str(model_path).split(model_name)[1].split("_")[1][1:])-1] = value
                else:
                    models_metrics[f"test_{metric}"][-1] = value

        # Check for the best model
        if "model" in best_model:
            if evaluation_metrics["f1-score"] > best_model["f1-score"]:
                best_model["model"] = Path(model_path).name
                best_model["loss"] = evaluation_metrics["loss"]
                for metric, value in evaluation_metrics.items():
                    if metric != "loss":
                        best_model[metric] = value
        else:
            best_model["model"] = Path(model_path).name
            best_model["loss"] = evaluation_metrics["loss"]
            for metric, value in evaluation_metrics.items():
                if metric != "loss":
                    best_model[metric] = value

    if len(models_paths) > 1:
        print(f"\nBest model: {best_model['model']}")
        print(f"  - Loss: {np.round(best_model['loss'], 4)}")
        for metric, value in list(best_model.items())[2:]:
            print(f"  - {metric}: {np.round(value, 4)}")

    return best_model, models_metrics


def collapse_probabilities(
    prediction: Union[np.ndarray, tf.Tensor],
    pixel_intensity: Optional[int] = 127) -> Union[np.ndarray, tf.Tensor]:
    """Converts the Softmax probability of each each pixel class to the class with the highest probability.

    Args:
        prediction (Union[np.ndarray, tf.Tensor]): A prediction in the format `(HEIGHT, WIDTH, CLASSES)`.
        pixel_intensity (Optional[int], optional): The intensity each pixel class will be assigned. Defaults to 127.

    Returns:
        Union[np.ndarray, tf.Tensor]: The prediction with the collapsed probabilities into the classes.
    """
    classes = prediction.shape[-1]
    for i in range(classes):
        prediction[:, :, i] = np.where(
            np.logical_and.reduce(
                np.array([prediction[:, :, i] > prediction[:, :, j] for j in range(classes) if j != i])), pixel_intensity, 0)

    return prediction


def color_classes(prediction: np.ndarray) -> np.ndarray:
    """Color a n-dimensional array of one-hot-encoded semantic segmentation image.

    Args:
        prediction (np.ndarray): The one-hot-encoded array image.

    Returns:
        np.ndarray: A RGB image with colored pixels per class.
    """
    prediction = PIL.Image.fromarray(prediction.astype(np.uint8), mode="P")

    colormap = imgviz.label_colormap()
    prediction.putpalette(colormap.flatten())

    prediction = np.asarray(prediction.convert())
    return prediction


def patch_predict(
    model: tf.keras.Model,
    image: Union[np.ndarray, tf.Tensor],
    patch_shape: Tuple[int, int, int]) -> Union[np.ndarray, tf.Tensor]:
    """Performs image prediction in patches of the original image for reduced GPU memory usage.

    This function will slice down the `image` and perform predictions in smaller portions of it (`patches`).
    The predicted `patches` are put back together in the same shape as the `image`.

    Args:
        model (tf.keras.Model): The model to be used for prediction.
        image (Union[np.ndarray, tf.Tensor]): The image to be predicted.
        patch_shape (Tuple[int, int, int]): The shape of the patch to be used for the predictions.

    Raises:
        ValueError: If the model's input shape an the patch shape do not match.

    Returns:
        Union[np.ndarray, tf.Tensor]: The predicted image constructed from the patch predictions.
    """
    input_shape = get_model_input_shape(model)
    if input_shape != patch_shape:
        raise ValueError("The model's input shape does not match the patch shape: model '{input_shape}' != patch '{patch_shape}'.")
    else:
        height = image.shape[0]
        width = image.shape[1]
        x_range = range(0, height, patch_shape[0])
        y_range = range(0, width, patch_shape[1])

        patch_prediction = np.zeros(image.shape)

        for x in x_range:
            for y in y_range:
                slice = image[x:x+patch_shape[0], y:y+patch_shape[1]]
                batch = slice.reshape((1,) + slice.shape)
                patch_prediction[x:x+patch_shape[0], y:y+patch_shape[1]] = model(batch, training=False)[0].numpy()

        return patch_prediction


def predict(
    model: Union[str, tf.keras.Model],
    images: str,
    normalize: Optional[bool] = True,
    input_shape: Optional[Tuple[int, int, int]] = None,
    copy_images: Optional[bool] = False,
    grayscale: Optional[bool] = False,
    analyze_contours: Optional[bool] = False,
    output_predictions: Optional[str] = "predictions",
    output_contour_analysis: Optional[str] = None,
    record_id: Optional[str] = None,
    record_class: Optional[str] = None,
    measures_only: Optional[bool] = False,
    current_time: Optional[str] = time.strftime('%Y%m%d%H%M%S')) -> None:
    """Predicts the segmentation mask of the input image(s).

    Args:
        model (Union[str, tf.keras.Model]): The model to be used to perform the prediction(s).
        images (str): A path to an image file, or a path to a directory containing images, or a path to a directory containing subdirectories of classes.
        normalize (Optional[bool], optional): Whether or not to put the image values between zero and one ([0,1]). Defaults to True.
        input_shape (Optional[Tuple[int, int, int]], optional): The input shape the loaded model and images should have, in format `(HEIGHT, WIDTH, CHANNELS)`. If `model` is a `tf.keras.model` with an input shape different from `input_shape`, then its input shape will be changed to `input_shape`. Defaults to None.
        copy_images (Optional[bool], optional): Whether or not to copy the input images to the predictions output directory. Defaults to False.
        grayscale (Optional[bool], optional): Whether or not to save the predicted masks as grayscale images with values for classes starting from zero. Defaults to `False`.
        analyze_contours (Optional[bool], optional): Whether or not to apply the contour analysis algorithm. If `True`, it will also write the contour measurements to a `.csv` file. Defaults to False.
        output_predictions (Optional[str], optional): The path where to save the predicted segmentation masks. Defaults to "predictions".
        output_contour_analysis (Optional[str], optional): The path where to save the `.csv` file containing the contour measurements. Only effective if `analyze_contour` is `True`. Defaults to None.
        record_id (Optional[str], optional): An ID that will identify the contour measurements. Defaults to None.
        record_class (Optional[str], optional): The class the contour measurements belong to. Defaults to None.
        measures_only (Optional[bool], optional): Do not save the predicted images or copy the input images to the output path. If `True`, it will override the effect of `output_predictions`. Defaults to False.
        current_time (Optional[str], optional): A timestamp to be added to the contour measurements, in the format `YYYYMMDDHHMMSS`. Defaults to time.strftime('%Y%m%d%H%M%S').

    Raises:
        FileNotFoundError: If `images` is not a path to file or a directory that exist.
        ValueError: If `images` is not a `str`.
        ValueError: If `model` is not a path to a file.
    """
    if isinstance(images, str):
        if Path(images).is_dir():
            files = list_files(images, as_numpy=True)
        elif Path(images).is_file():
            files = [images]
        else:
            raise FileNotFoundError(f"The directory or file was not found at `{images}`.")
    elif not isinstance(images, np.ndarray):
        raise ValueError(f"`images` must be a `str`. Given `{type(images)}`.")

    if isinstance(model, str):
        model = load_model(model_path=model, input_shape=input_shape)
    elif not isinstance(model, tf.keras.Model):
        raise ValueError(f"`model` must be a `str` or `tf.keras.Model`. Given `{type(model)}`.")

    if not input_shape:
        input_shape = get_model_input_shape(model)

    if not output_contour_analysis:
        output_contour_analysis = output_predictions

    output_predictions = Path(output_predictions)
    output_predictions.mkdir(exist_ok=True, parents=True)

    for file in tqdm(files, desc=record_id):
        image = load_image(image_path=file, normalize=normalize, as_numpy=True)

        if image.shape != input_shape:
            prediction = patch_predict(model, image, input_shape)
        else:
            batch = image.reshape((1,) + image.shape)
            prediction = model(batch, training=False)[0].numpy()

        prediction = collapse_probabilities(prediction=prediction, pixel_intensity=127)

        if prediction.shape[-1] > 3:
            prediction = color_classes(prediction)

        file = Path(file)
        if analyze_contours:
            prediction, detail = contour_analysis.analyze_contours(mask=prediction)
            prediction, parent_contours, child_contours = prediction
            detail, discarded_parent_contours, discarded_child_contours = detail

            if record_id:
                parent_measurements, child_measurements = contour_analysis.get_contour_measurements(
                    parent_contours=parent_contours,
                    child_contours=child_contours,
                    shape=input_shape[:2],
                    mask_name=Path(file).name,
                    record_id=record_id,
                    record_class=record_class)

                contour_analysis.write_contour_measurements(
                    parent_measurements=parent_measurements,
                    child_measurements=child_measurements,
                    output_path=output_contour_analysis,
                    datetime=current_time)

                if len(discarded_parent_contours) > 0 or len(discarded_child_contours) > 0:
                    discarded_parent_measurements, discarded_child_measurements = contour_analysis.get_contour_measurements(
                        parent_contours=discarded_parent_contours,
                        child_contours=discarded_child_contours,
                        shape=input_shape[:2],
                        mask_name=Path(file).name,
                        record_id=record_id,
                        record_class=record_class,
                        start_index=len(parent_measurements),
                        contours_flag="invalid")

                    contour_analysis.write_contour_measurements(
                        parent_measurements=discarded_parent_measurements,
                        child_measurements=discarded_child_measurements,
                        output_path=output_contour_analysis,
                        datetime=current_time)

            if detail is not None and not measures_only:
                filtered_objects = output_predictions.joinpath("filtered_objects")
                filtered_objects.mkdir(exist_ok=True, parents=True)

                cv2.imwrite(
                    str(filtered_objects.joinpath(f"{file.stem}_detail.png")), cv2.cvtColor(detail, cv2.COLOR_BGR2RGB))
                if copy_images:
                    shutil.copyfile(str(file), filtered_objects.joinpath(file.name))

        if not measures_only:
            if grayscale:
                prediction = reset_class_values(prediction)
            else:
                prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)

            cv2.imwrite(str(output_predictions.joinpath(f"{file.stem}_prediction.png")), prediction)
            if copy_images:
                shutil.copyfile(str(file), output_predictions.joinpath(file.name))


def plot_metrics(
    metrics_file_path: str,
    output: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = (15, 15)) -> None:
    """Generates graphs displaying the training, validation, and test metrics.

    Args:
        metrics (str): The path to the `train_config.json` file.
        output (Optional[str], optional): The path where to save the graphs. If `None`, it will save in the same location as `metrics_file_path`. Defaults to None.
        figsize (Optional[Tuple[int, int]], optional): The dimensions of the graphs. Defaults to (15, 15).
    """
    metrics = Path(metrics_file_path)
    if metrics.is_file():
        with metrics.open() as f:
            metrics_file = json.load(f)

        metrics_data = {
            "Training metrics": {
                "loss": metrics_file["train_metrics"]["loss"],
                "f1-score": metrics_file["train_metrics"]["f1-score"],
                "iou-score": metrics_file["train_metrics"]["iou_score"]
            },
            "Validation metrics": {
                "val_loss": metrics_file["train_metrics"]["val_loss"],
                "val_f1-score": metrics_file["train_metrics"]["val_f1-score"],
                "val_iou-score": metrics_file["train_metrics"]["val_iou_score"]
            },
            "Test metrics": {
                "test_loss": metrics_file["test_metrics"]["test_loss"],
                "test_f1-score": metrics_file["test_metrics"]["test_f1-score"],
                "test_iou-score": metrics_file["test_metrics"]["test_iou_score"]
            },
            "Learning rage": {
                "lr": metrics_file["train_metrics"]["lr"]
            }
        }

        if output:
            output_path = Path(output)
        else:
            output_path = Path(metrics.parent)
        output_path.mkdir(exist_ok=True, parents=True)

        for i, (title, data) in enumerate(metrics_data.items()):
            df = pd.DataFrame(data)
            df.index = range(1, len(df.index) + 1)

            image = df.plot(grid=True, figsize=figsize)
            image.set(xlabel="Epoch", title=title)
            image.set_ylim(ymin=0)

            for column in df.columns:
                if "loss" in column:
                    text = f"e{np.argmin(list(df[column])) + 1}"
                    value = (np.argmin(list(df[column])) + 1, df[column].min())
                else:
                    text = f"e{np.argmax(list(df[column])) + 1}"
                    value = (np.argmax(list(df[column])) + 1, df[column].max())

                if column != "lr":
                    image.annotate(text, value, arrowprops=dict(facecolor='black', shrink=0.05))

            image = image.get_figure()
            image.savefig(str(output_path.joinpath(f"0{i+1}_{title.lower().replace('', '')}.png")))


def compute_classes_distribution(
    dataset: tf.data.Dataset,
    batches: Optional[int] = 1,
    plot: Optional[bool] = True,
    figsize: Optional[Tuple[int, int]] = (20, 10),
    output: Optional[str] = ".",
    get_as_weights: Optional[bool] = False,
    classes: Optional[list] = ["Background", "Nucleus", "NOR"]) -> dict:
    """Computes the class distribution in a `tf.data.Dataset` considering the number of pixels.

    Args:
        dataset (tf.data.Dataset): A `tf.data.Dataset` containing the images and segmentation masks.
        batches (Optional[int], optional): The number of batches the dataset contains. Defaults to 1.
        plot (Optional[bool], optional): Whether or not to plot and save a Matplotlib bars graph with the classes distribution. Defaults to True.
        figsize (Optional[Tuple[int, int]], optional): The size of the figure to be ploted. Defaults to (20, 10).
        output (Optional[str], optional): The path where to save the figure. Defaults to ".".
        get_as_weights (Optional[bool], optional): Converts the number of pixels per class to percentage over all classes. Defaults to False.
        classes (Optional[list], optional): The name of the classes. Defaults to ["Background", "Nucleus", "NOR"].

    Returns:
        dict: A dictionary with an entry per class containing the class distribution.
    """
    class_occurrence = []
    batch_size = None

    for i, batch in enumerate(dataset):
        if i == batches:
            batch_size = batch[0].shape[0]
            break

        for mask in batch[1]:
            class_count = []
            for class_index in range(mask.shape[-1]):
                class_count.append(tf.math.reduce_sum(mask[:, :, class_index]))
            class_occurrence.append(class_count)

    class_occurrence = tf.convert_to_tensor(class_occurrence, dtype=tf.int64)
    class_distribution = tf.reduce_sum(class_occurrence, axis=0) / (batches * batch_size)
    class_distribution = class_distribution.numpy()
    class_distribution = class_distribution * 100 / (dataset.element_spec[0].shape[1] * dataset.element_spec[0].shape[2])
    if get_as_weights:
        class_distribution = (100 - class_distribution) / 100
        class_distribution = np.round(class_distribution, 2)

    distribution = {}
    for occurrence, class_name in zip(class_distribution, classes):
        distribution[class_name] = float(occurrence)

    if plot:
        output_path = Path(output)
        output_path.mkdir(exist_ok=True, parents=True)

        class_occurrence = class_occurrence.numpy()
        df = pd.DataFrame(class_occurrence, columns=classes)
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


def get_duration(start: float, end: float) -> str:
    """Calculates the time delta between a starting time and a ending time from `time.time()`.

    Args:
        start (float): The starting time.
        end (float): The ending time.

    Returns:
        str: The time delta in format `HH:MM:SS`.
    """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    duration = "{:0>2}:{:0>2}:{:02.0f}".format(int(hours), int(minutes), seconds)
    return duration


def add_time_delta(duration1: str, duration2: str) -> str:
    """Adds a duration to another duration.

    Args:
        duration1 (str): The duration to be summed to another one, in format `HH:MM:SS`.
        duration2 (str): The other duration to be summed, in format `HH:MM:SS`.

    Returns:
        str: The duration sum of `duration1` and `duration2`, in format `HH:MM:SS`.
    """
    hours, minutes, seconds = duration1.split(":")
    duration1 = datetime.timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds))

    hours, minutes, seconds = duration2.split(":")
    duration2 = datetime.timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds))

    total_seconds = (duration1 + duration2).total_seconds()
    duration = "%d:%02d:%02d" % (total_seconds / 3600, total_seconds / 60 % 60, total_seconds % 60)
    return duration
