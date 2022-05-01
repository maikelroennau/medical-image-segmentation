import datetime
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple, Union

import imgviz
import numpy as np
import pandas as pd
import PIL.Image
import tensorflow as tf


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
    if prediction.shape[-1] <= 3:
        color_map = [
            [130, 130, 130],
            [255, 128, 0],
            [0, 0, 255]
        ]

        map_classes = []
        for i in range(prediction.shape[-1]):
            map_classes.append(prediction[:, :, i] > 0)

        for i in range(prediction.shape[-1]):
            for j in range(prediction.shape[-1]):
                prediction[:, :, j] = np.where(map_classes[i], color_map[i][j], prediction[:, :, j])
    else:
        prediction = PIL.Image.fromarray(prediction.astype(np.uint8), mode="P")

        colormap = imgviz.label_colormap()
        prediction.putpalette(colormap.flatten())

        prediction = np.asarray(prediction.convert())
    return prediction


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
        print(metrics_file)
        metrics_data = {
            "Training metrics": {
                "segmentation_loss": metrics_file["train_metrics"]["softmax_loss"],
                "counts_loss": metrics_file["train_metrics"]["nuclei_nor_counts_loss"],
                "f1-score": metrics_file["train_metrics"]["softmax_f1-score"],
                "iou-score": metrics_file["train_metrics"]["softmax_iou_score"],
                "counts_mae": metrics_file["train_metrics"]["nuclei_nor_counts_mae"]
            },
            "Validation metrics": {
                "val_segmentation_loss": metrics_file["validation_metrics"]["val_softmax_loss"],
                "val_counts_loss": metrics_file["validation_metrics"]["val_nuclei_nor_counts_loss"],
                "val_f1-score": metrics_file["validation_metrics"]["val_f1-score"],
                "val_iou-score": metrics_file["validation_metrics"]["val_iou_score"],
                "val_counts_mae": metrics_file["validation_metrics"]["val_nuclei_nor_counts_mae"]
            },
            "Test metrics": {
                "test_segmentation_loss": metrics_file["test_metrics"]["softmax_test_softmax_loss"],
                "test_counts_loss": metrics_file["test_metrics"]["softmax_test_nuclei_nor_counts_loss"],
                "test_f1-score": metrics_file["test_metrics"]["softmax_test_f1-score"],
                "test_iou-score": metrics_file["test_metrics"]["softmax_test_iou_score"],
                "test_counts_mae": metrics_file["test_metrics"]["softmax_test_nuclei_nor_counts_mae"]
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

        i = 0
        for title, data in metrics_data.items():
            try:
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
                i += 1
            except:
                pass


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


def pad_along_axis(array: np.ndarray, size: int, axis: int = 0, mode="reflect"):
    """Pad an image along a specific axis.

    Args:
        array (np.ndarray): The image to be padded.
        size (int): The size the padded axis must have.
        axis (int, optional): Which axis to apply the padding. Defaults to 0.
        mode (str, optional): How to fill the padded pixels. Defaults to "reflect".

    Returns:
        np.ndarray: The padded image.
    """
    pad_size = size - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode=mode)


def get_hash_file(path: str) -> str:
    """Obtains the hash of the a file.

    Args:
        path (str): The path to the file.

    Returns:
        str: The file hash.
    """
    with open(path, "rb") as f:
        bytes = f.read()
        hash_file = hashlib.sha256(bytes).hexdigest()
    return hash_file
