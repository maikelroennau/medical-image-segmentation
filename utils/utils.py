import datetime
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import imgviz
import numpy as np
import pandas as pd
import segmentation_models as sm
import tensorflow as tf


def get_color_map(colormap: Optional[str] = "agnor"):
    """Provides the default color map for the specified number of classes.

    The number of classes refers to the number of dimentions outputted by the model.

    Args:
        colormap (str): What color map to use. Pass `agnor` for a color map for 3 classes plus background, or pass `papanicolaou` for a color map for 7 classes plus background. Defaults to `agnor`.
    """
    if colormap == "agnor":
        # AgNOR
        color_map = [
            [130, 130, 130], # Gray         _background_
            [255, 128,   0], # Orange       citoplasma
            [  0,   0, 255], # Blue         AgNOR
            [128,   0,  64], # Purple       Satellite
        ]
    elif colormap == "papanicolaou":
        # Papanicolaou
        color_map = [
            [130, 130, 130], # Gray         _background_
            [ 78, 121, 167], # Blue         aglomerado
            [242, 142,  43], # Orange       citoplasma
            [ 44, 160,  44], # Green        escama
            [200,  82,   0], # Brown        superficial
            [ 23, 190, 207], # Turquoise    intermediaria
            [188, 189,  34], # Mustard      suspeita
            [148, 103, 189], # Purple       binucleacao
        ]
    else:
        return None

    return color_map

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

    return prediction.astype(np.uint8)


def color_classes(prediction: np.ndarray, colormap: Optional[str] = "agnor") -> np.ndarray:
    """Color a n-dimensional array of one-hot-encoded semantic segmentation image.

    Args:
        prediction (np.ndarray): The one-hot-encoded array image.
        colormap (str): What color map to use. Pass `agnor` for a color map for 3 classes plus background, or pass `papanicolaou` for a color map for 7 classes plus background. Defaults to `agnor`.

    Returns:
        np.ndarray: A RGB image with colored pixels per class.
    """
    # Check if the image encodes the number of classes.
    if len(prediction.shape) >= 3:

        # Extend color map if necessary.
        n_classes = prediction.shape[-1]
        if n_classes > 4:
            color_map = get_color_map(colormap="papanicolaou")
        else:
            color_map = get_color_map(colormap="agnor")

        if n_classes > len(color_map):
            color_map.extend(imgviz.label_colormap(n_label=n_classes))

        # Obtain color map before changing the array.
        class_maps = []
        for i in range(n_classes):
            class_maps.append(prediction[:, :, i] > 0)

        # Recolor classes.
        for i in range(n_classes):
            for j in range(3): # 3 color channels
                prediction[:, :, j] = np.where(class_maps[i], color_map[i][j], prediction[:, :, j])

        # Remove any extra channels so the array can be saved as an image.
        prediction = prediction[:, :, :3]

        return prediction
    else:
        class_values = np.unique(prediction)
        n_classes = len(class_values)

        color_map = get_color_map(colormap=colormap)

        # Extend color map if necessary.
        if n_classes > len(color_map):
            color_map.extend(imgviz.label_colormap(n_label=n_classes))

        # Obtain color map.
        class_maps = []
        for class_value in class_values:
            class_maps.append(prediction == class_value)

        colored = np.zeros(prediction.shape + (3,), dtype=np.uint8)

        # Recolor classes.
        for i, class_value in enumerate(class_values):
            for j in range(3): # 3 color channels
                colored[:, :, j] = np.where(class_maps[i], color_map[class_value][j], colored[:, :, j])

        return colored


def one_hot_encoded_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert one-hot-encoded arrays to RGB images.

    Args:
        image (np.ndarray): The input array, with an arbitrary number of dimensions.

    Returns:
        np.ndarray: The RGB image mapping each dimension to a color.
    """
    # Extend color map if necessary.
    n_classes = len(np.unique(image))
    if n_classes > 4:
        color_map = get_color_map(colormap="papanicolaou")
    else:
        color_map = get_color_map()
    if n_classes > len(color_map):
        color_map.extend(imgviz.label_colormap(n_label=n_classes))

    # Add new axis to represent RGB colors.
    overlay_mask = imgviz.label2rgb(image, colormap=color_map)

    return overlay_mask


def get_intersection(
    expected_contour: np.ndarray,
    predicted_contour: np.ndarray,
    shape: Tuple[int, int]) -> float:
    """Get the intersection value for the input contours.

    The function uses the Intersection Over Union (IoU) metric from the `Segmentation Models` library.

    Args:
        expected_contour (np.ndarray): The first contour.
        predicted_contour (np.ndarray): The second contour.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extracted, in the format `(HEIGHT, WIDTH)`.

    Returns:
        float: The intersection value in range [0, 1].
    """
    expected = np.zeros(shape, dtype=np.uint8)
    predicted = np.zeros(shape, dtype=np.uint8)

    expected = cv2.drawContours(expected, contours=[expected_contour], contourIdx=-1, color=1, thickness=cv2.FILLED)
    predicted = cv2.drawContours(predicted, contours=[predicted_contour], contourIdx=-1, color=1, thickness=cv2.FILLED)

    expected = expected.reshape((1,) + expected.shape).astype(np.float32)
    predicted = predicted.reshape((1,) + predicted.shape).astype(np.float32)

    iou = sm.metrics.iou_score(expected, predicted).numpy()
    return iou


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
            "Learning rate": {
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
            image.savefig(str(output_path.joinpath(f"0{i+1}_{title.lower().replace(' ', '_')}.png")))


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
        figsize (Optional[Tuple[int, int]], optional): The size of the figure to be plotted. Defaults to (20, 10).
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


def pad_along_axis(array: np.ndarray, size: int, axis: int = 0, mode: Optional[str] = "constant"):
    """Pad an image along a specific axis.

    Args:
        array (np.ndarray): The image to be padded.
        size (int): The size the padded axis must have.
        axis (int, optional): Which axis to apply the padding. Defaults to 0.
        mode (str, optional): How to fill the padded pixels. Defaults to "constant".

    Returns:
        np.ndarray: The padded image.
    """
    pad_size = size - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    if mode == "constant":
        return np.pad(array, pad_width=npad, mode=mode, constant_values=0)
    else:
        return np.pad(array, pad_width=npad, mode=mode)


def get_labelme_shapes(annotation_path: str, shape_types: Optional[List[str]] = None) -> List[dict]:
    """Get all shapes of the given types from a `labelme` file.

    Args:
        annotation_path (str): The path to the .json `labelme` annotation file.
        shape_types (Optional[list], optional): List of shape types to return. Example: `["polygon", "circle"]. Defaults to None.

    Raises:
        FileNotFoundError: If the annotation file is not found.

    Returns:
        List[dict]: List containing the shapes in the `annotation_file` that are of any type in `shape_types`.
    """
    annotation_path = Path(annotation_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"The annotation file `{annotation_path}` was not found.")
    else:
        with open(str(annotation_path), "r", encoding="utf-8") as annotation_pointer:
            annotation_file = json.load(annotation_pointer)

        shapes = []
        for shape in annotation_file["shapes"]:
            if shape_types is not None:
                if shape["shape_type"] in shape_types:
                    shapes.append(shape)
            else:
                shapes.append(shape)
        return shapes


def get_labelme_points(annotation_path, shape_types: Optional[list] = None, reshape_for_opencv: Optional[bool] = False) -> List[List[int]]:
    """Get points from shapes from a `labelme` annotation file.

    Args:
        annotation_path (str): The path to the .json `labelme` annotation file.
        shape_types (Optional[list], optional): List of shape types to return. Example: `["polygon", "circle"]. Defaults to None.
        reshape (Optional[bool], optional): Whether or not to reshape points to be compatible with OpenCV. Defaults to False.

    Returns:
        List[List[int]]: List containing the points of the shapes in the `annotation_file` that are of any type in `shape_types`.
    """
    shapes = get_labelme_shapes(annotation_path, shape_types)
    points = []
    for shape in shapes:
        shape_points = shape["points"]
        shape_points = np.array(shape_points, dtype=np.int32)

        if reshape_for_opencv:
            shape_points = shape_points.reshape((shape_points.shape[0], 1, shape_points.shape[1]))
        if shape_types is not None:
            if shape["shape_type"] in shape_types:
                points.append(shape_points)
        else:
            points.append(shape_points)
    return points


def convert_bbox_to_contour(bbox: np.ndarray) -> np.array:
    """Converts a `labelme` bounding box points to a contour.

    Bounding boxes are represented with two points by `labelme`, the top left, and bottom right points.
    When checking if another contour is within a given `labelme` bounding box, OpenCV assumes the two points represent a line.
    This function convert the `labelme` bounding box into a proper contour that will be interpreted as a rectangle in OpenCV.

    Args:
        bbox (np.array): The bounding box points.

    Returns:
        np.array: The contour in the format of a polygon.
    """
    bbox.insert(1, [bbox[1][0], bbox[0][1]])
    bbox.insert(3, [bbox[0][0], bbox[2][1]])
    bbox = np.array(bbox, dtype=np.int32)
    return bbox


def get_object_classes(annotation_path):
    annotation_path = Path(annotation_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"The annotation file `{annotation_path}` was not found.")
    else:
        with open(str(annotation_path), "r") as annotation_pointer:
            annotation_file = json.load(annotation_pointer)

        classes = []
        for shape in annotation_file["shapes"]:
            classes.append(shape["label"])
        return classes


def get_mean_rgb_values(contour: np.ndarray, image: np.ndarray) -> List[Union[float, float, float]]:
    """Obtains the mean RGB values of a given contour in an image.

    Args:
        contour (np.ndarray): The contour that delimits the pixels that will be considered.
        image (np.ndarray): The image from the RGB values will be extracted.

    Returns:
        List[Union[float, float, float]]: List containing the average RGB value, per channel.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = cv2.drawContours(mask, contours=[contour], contourIdx=-1, color=1, thickness=cv2.FILLED)
    total = np.sum(mask)
    red = np.round(np.sum(mask * image[:, :, 0]) / total, 4)
    green = np.round(np.sum(mask * image[:, :, 1]) / total, 4)
    blue = np.round(np.sum(mask * image[:, :, 2]) / total, 4)
    return red, green, blue


def optimize_models_storage(directory: str) -> None:
    directory = Path(directory)
    train_config_file = directory.joinpath(f"train_config_{directory.name}.json")
    if not train_config_file.is_file():
        train_config_file = directory.joinpath(f"train_config.json")
        if not train_config_file.is_file():
            print(f"Train config file not found at '{str(directory)}'.")
            return

    with open(str(train_config_file)) as f:
        config_file = json.load(f)

        if "best_model" not in config_file:
            print(f"No best model information found in `{str(directory.joinpath(train_config_file.name))}`")
            return

        best_model = config_file["best_model"]["model"]

        models = [model.name for model in directory.glob("*.h5")]
        if best_model in models:
            models.remove(best_model)
        else:
            print(f"The best model '{best_model}' was not found in the directory '{str(directory.joinpath(train_config_file.name))}'")
            return

        for model in models:
            os.remove(str(directory.joinpath(model)))
