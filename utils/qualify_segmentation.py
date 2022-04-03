from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import segmentation_models as sm
from tqdm import tqdm

from utils.contour_analysis import get_contours
from utils.data import list_files, load_image, one_hot_encode


COLUMNS = [
    "source_image",
    "expected_nuclei",
    "expected_nors",
    "predicted_nuclei",
    "predicted_nors",
    "true_positive_nuclei",
    "false_positive_nuclei",
    "false_negative_nuclei",
    "true_positive_nors",
    "false_positive_nors",
    "false_negative_nors"
]


def get_intersection(
    expected_contour: np.ndarray,
    predicted_contour: np.ndarray,
    shape: Tuple[int, int]) -> float:
    """Get the intersection value for the input contours.

    The function uses the Intersection Over Union (IoU) metric from the `Segmentation Models` library.

    Args:
        expected_contour (np.ndarray): The first contour.
        predicted_contour (np.ndarray): The second contour.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extrated, in the format `(HEIGHT, WIDTH)`.

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


def count_intersect_contours(
    ground_truth_contours: List[np.ndarray],
    predicted_contours: List[np.ndarray],
    shape: Tuple[int, int]) -> Union[int, int]:
    """Count the number of contours that intersect with the ground truth contours.

    Args:
        ground_truth_contours (List[np.ndarray]): The list of ground truth contours.
        predicted_contours (List[np.ndarray]): The list of predicted contours.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extrated, in the format `(HEIGHT, WIDTH)`.

    Returns:
        Union[int, int]: A tuple where the first element is the number of predicted contours intersecting with the ground truth contours (true positives), and the second number is the number of contours that were predicted but are not in the ground truth (false positives).
    """
    intersect_contours = 0
    for predicted_contour in predicted_contours:
        intersected = False
        for ground_truth_contour in ground_truth_contours:
            intersection = get_intersection(ground_truth_contour, predicted_contour, shape=shape)
            if np.round(intersection, 2) > 0.:
                intersected = True
                break

        if intersected:
            intersect_contours += 1

    false_positives = len(predicted_contours) - intersect_contours

    return intersect_contours, false_positives


def qualify_segmentation(
    ground_truth_path: str,
    predictions_path: str,
    output_qualification: Optional[str] = "qualification.csv",
    classes: Optional[int] = None) -> list:
    """Qualifies the predicted contours against the ground truth contours.

    This function works by checking if the segmented nuclei and NORs match to nuclei and NORs in the ground truth.

    Args:
        ground_truth_path (str): The path to the ground truth masks.
        predictions_path (str): The path to the predicted masks.
        output_qualification (Optional[str], optional): The path were to save the qualification information. Defaults to "qualification.csv".
        classes (Optional[int], optional): The number of classes in the data. Defaults to None.

    Raises:
        FileNotFoundError: If the ground truth masks are not found.
        FileNotFoundError: If the predicted masks are not found.
        ValueError: If the number of ground truth files and prediction files do not match.

    Returns:
        list: A list of dictionaries containing the expected number of contours and the number of true positives, false positives, and false negatives.
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
    data = []

    for ground_truth_file_path, prediction_file_path in tqdm(zip(ground_truth, predictions), total=len(ground_truth)):
        ground_truth = load_image(ground_truth_file_path, as_gray=True)
        prediction = load_image(prediction_file_path, as_gray=True)

        if classes_undefined:
            classes = np.unique(ground_truth).size

        ground_truth = one_hot_encode(ground_truth, classes=classes, as_numpy=True)
        prediction = one_hot_encode(prediction, classes=classes, as_numpy=True)

        expected_nuclei = get_contours(ground_truth[:, :, 1] + ground_truth[:, :, 2])
        expected_nors = get_contours(ground_truth[:, :, 2])
        predicted_nuclei = get_contours(prediction[:, :, 1] + prediction[:, :, 2])
        predicted_nors = get_contours(prediction[:, :, 2])

        false_negative_nuclei = max(0, len(expected_nuclei) - len(predicted_nuclei))
        false_negative_nors = max(0, len(expected_nors) - len(predicted_nors))

        true_positive_nuclei, false_positive_nuclei = count_intersect_contours(
            expected_nuclei,
            predicted_nuclei,
            shape=ground_truth.shape[:2])

        true_positive_nors, false_positive_nors = count_intersect_contours(
            expected_nors,
            predicted_nors,
            shape=ground_truth.shape[:2])

        data = [{
            "source_image": Path(ground_truth_file_path).name,
            "expected_nuclei": len(expected_nuclei),
            "expected_nors": len(expected_nors),
            "predicted_nuclei": len(predicted_nuclei),
            "predicted_nors": len(predicted_nors),
            "true_positive_nuclei": true_positive_nuclei,
            "false_positive_nuclei": false_positive_nuclei,
            "false_negative_nuclei": false_negative_nuclei,
            "true_positive_nors": true_positive_nors,
            "false_positive_nors": false_positive_nors,
            "false_negative_nors": false_negative_nors
        }]

        df = pd.DataFrame(data, columns=COLUMNS)

        output_qualification = Path(output_qualification)
        output_qualification.parent.mkdir(exist_ok=True, parents=True)

        if output_qualification.is_file():
            df.to_csv(str(output_qualification), mode="a", header=False, index=False)
        else:
            df.to_csv(str(output_qualification), mode="w", header=True, index=False)

    return data
