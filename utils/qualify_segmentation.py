from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.contour_analysis import (analyze_contours,
                                    discard_contours_outside_contours,
                                    get_contours)
from utils.data import list_files, load_image, one_hot_encode
from utils.utils import (convert_bbox_to_contour, get_intersection,
                         get_labelme_points)


COLUMNS = [
    "source_image",
    "expected_nuclei",
    "expected_clusters",
    "expected_satellites"
    "predicted_nuclei",
    "predicted_clusters",
    "predicted_satellites",
    "true_positive_nuclei",
    "false_positive_nuclei",
    "false_negative_nuclei",
    "true_positive_clusters",
    "false_positive_clusters",
    "false_negative_clusters",
    "true_positive_satellites",
    "false_positive_satellites",
    "false_negative_satellites",
    "bboxes"
]

iou_threshold = 0.50

def get_false_positive_contours(
    ground_truth_contours: List[np.ndarray],
    predicted_contours: List[np.ndarray],
    shape: Tuple[int, int],
    drop: bool = False) -> Union[list, list]:
    """Gets the false positive contours based on the ground truth contours.

    Args:
        ground_truth_contours (List[np.ndarray]): List containing the ground truth contours.
        predicted_contours (List[np.ndarray]): List containing the predicted contours.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extracted, in the format `(HEIGHT, WIDTH)`.
        drop (bool): Whether to drop contours from `ground_truth_contours` after evaluation. Defaults to False.

    Returns:
        Union[list, list]: The false positive contours.
    """

    false_positives = []

    if drop:
        ground_truth_contours = { str(key): value for key, value in enumerate(ground_truth_contours) }

        for predicted_contour in predicted_contours:
            if len(ground_truth_contours) == 0:
                break

            intersected = False
            for key, ground_truth_contour in ground_truth_contours.items():
                intersection = get_intersection(ground_truth_contour, predicted_contour, shape=shape)
                if np.round(intersection, 2) >= iou_threshold:
                    intersected = True
                    ground_truth_contours.pop(key)
                    break

            if not intersected:
                false_positives.append(predicted_contour)
    else:
        for predicted_contour in predicted_contours:
            intersected = False
            for ground_truth_contour in ground_truth_contours:
                intersection = get_intersection(ground_truth_contour, predicted_contour, shape=shape)
                if np.round(intersection, 2) >= iou_threshold:
                    intersected = True
                    break

            if not intersected:
                false_positives.append(predicted_contour)

    return false_positives


def count_intersect_contours(
    ground_truth_contours: List[np.ndarray],
    predicted_contours: List[np.ndarray],
    shape: Tuple[int, int],
    drop: bool = False) -> Union[int, int]:
    """Count the number of contours that intersect with the ground truth contours.

    Args:
        ground_truth_contours (List[np.ndarray]): The list of ground truth contours.
        predicted_contours (List[np.ndarray]): The list of predicted contours.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extracted, in the format `(HEIGHT, WIDTH)`.
        drop (bool): Whether to drop contours from `ground_truth_contours` after evaluation. Defaults to False.

    Returns:
        Union[int, int]: A tuple where the first element is the number of predicted contours intersecting with the ground truth contours (true positives), and the second number is the number of contours that were predicted but are not in the ground truth (false positives).
    """
    intersecting_contours = 0
    if drop:
        ground_truth_contours = { str(key): value for key, value in enumerate(ground_truth_contours) }

        for predicted_contour in predicted_contours:
            if len(ground_truth_contours) == 0:
                break

            intersected = False
            for key, ground_truth_contour in ground_truth_contours.items():
                intersection = get_intersection(ground_truth_contour, predicted_contour, shape=shape)
                if np.round(intersection, 2) >= iou_threshold:
                    intersected = True
                    ground_truth_contours.pop(key)
                    break

            if intersected:
                intersecting_contours += 1
    else:
        for predicted_contour in predicted_contours:
            intersected = False
            for ground_truth_contour in ground_truth_contours:
                intersection = get_intersection(ground_truth_contour, predicted_contour, shape=shape)
                if np.round(intersection, 2) >= iou_threshold:
                    intersected = True
                    break

            if intersected:
                intersecting_contours += 1

    return intersecting_contours


def qualify_segmentation(
    ground_truth_path: str,
    predictions_path: str,
    classes: Optional[int] = None,
    output_qualification: Optional[str] = None,
    output_visualization: Optional[str] = None,
    bbox_annotations_path: Optional[str] = None,
    intersection_precision: Optional[float] = 4,
    apply_contour_analysis_on_ground_truth: Optional[bool] = False) -> list:
    """Qualifies the predicted contours against the ground truth contours.

    This function works by checking if the segmented nuclei and clusters match to nuclei and clusters in the ground truth.

    Args:
        ground_truth_path (str): The path to the ground truth masks.
        predictions_path (str): The path to the predicted masks.
        classes (Optional[int], optional): The number of classes in the data. Defaults to None.
        output_qualification (Optional[str], optional): The path were to save the qualification information. Defaults to `predictions_path`.
        output_visualization (Optional[bool], optional). Path where to save the visualization showing the differences in respect to the ground truth. Does not generate visualization if `None`.
        bbox_annotations_path (Optional[str], optional): Path to the `labelme` annotations containing bounding boxes for the nuclei to be considered.
        apply_contour_analysis_on_ground_truth (Optional[bool], optional): Whether or not to apply the contour analysis algorithm to the ground truth. Defaults to `False`.

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
    if bbox_annotations_path is not None:
        annotations = list_files(bbox_annotations_path, as_numpy=True, file_types=[".json"])
        if len(annotations) == 0:
            raise FileNotFoundError(f"No annotation files were found at `{annotations}`.")

    classes_undefined = True if classes is None else False
    data = []

    nucleus_records = pd.DataFrame({
        "source_image": pd.Series(dtype="str"),
        "nucleus": pd.Series(dtype=np.int32),
        "iou": pd.Series(dtype="float"),
        "type": pd.Series(dtype="str")
    })

    cluster_records = pd.DataFrame({
        "source_image": pd.Series(dtype="str"),
        "nucleus": pd.Series(dtype=np.int32),
        "cluster": pd.Series(dtype=np.int32),
        "iou": pd.Series(dtype="float"),
        "type": pd.Series(dtype="str")
    })

    satellite_records = pd.DataFrame({
        "source_image": pd.Series(dtype="str"),
        "nucleus": pd.Series(dtype=np.int32),
        "satellite": pd.Series(dtype=np.int32),
        "iou": pd.Series(dtype="float"),
        "type": pd.Series(dtype="str")
    })

    expected_records = pd.DataFrame({
        "source_image": pd.Series(dtype="str"),
        "expected_nuclei": pd.Series(dtype=np.int32),
        "expected_clusters": pd.Series(dtype=np.int32),
        "expected_satellites": pd.Series(dtype=np.int32)
    })

    expected_nuclei_count = 0
    expected_cluster_count = 0
    expected_satellite_count = 0
    predicted_nuclei_count = 0
    predicted_cluster_count = 0
    predicted_satellite_count = 0

    for ground_truth_file_path, prediction_file_path in tqdm(zip(ground_truth, predictions), total=len(ground_truth)):
        ground_truth = load_image(ground_truth_file_path, as_gray=True)
        prediction = load_image(prediction_file_path, as_gray=True)

        if classes_undefined:
            classes = np.unique(ground_truth).size

        ground_truth = one_hot_encode(ground_truth, classes=classes, as_numpy=True)
        prediction = one_hot_encode(prediction, classes=classes, as_numpy=True)

        if apply_contour_analysis_on_ground_truth:
            ground_truth[:, :, 2] += ground_truth[:, :, 3]
            ground_truth, _ = analyze_contours(ground_truth[:, :, :3])
            ground_truth, _, _ = ground_truth

        expected_nuclei = get_contours(ground_truth[:, :, 1] + ground_truth[:, :, 2] + ground_truth[:, :, 3])
        expected_clusters = get_contours(ground_truth[:, :, 2])
        expected_satellites = get_contours(ground_truth[:, :, 3])
        predicted_nuclei = get_contours(prediction[:, :, 1] + prediction[:, :, 2] + prediction[:, :, 3])
        predicted_clusters = get_contours(prediction[:, :, 2])
        predicted_satellites = get_contours(prediction[:, :, 3])

        expected_nuclei_count += len(expected_nuclei)
        expected_cluster_count += len(expected_clusters)
        expected_satellite_count += len(expected_satellites)
        predicted_nuclei_count += len(predicted_nuclei)
        predicted_cluster_count += len(predicted_clusters)
        predicted_satellite_count += len(predicted_satellites)

        if bbox_annotations_path is not None:
            annotation = str(Path(bbox_annotations_path).joinpath(f"{Path(ground_truth_file_path).stem.replace('_mask', '')}.json"))
            bboxes = get_labelme_points(annotation, shape_types=["rectangle"])

            # Convert bboxes into contours with four points.
            for i in range(len(bboxes)):
                bboxes[i] = convert_bbox_to_contour(bboxes[i].tolist())

            expected_nuclei, _ = discard_contours_outside_contours(bboxes, expected_nuclei)
            expected_clusters, _ = discard_contours_outside_contours(expected_nuclei, expected_clusters)
            expected_satellites, _ = discard_contours_outside_contours(expected_nuclei, expected_satellites)
            predicted_nuclei, _ = discard_contours_outside_contours(bboxes, predicted_nuclei)
            predicted_clusters, _ = discard_contours_outside_contours(predicted_nuclei, predicted_clusters)
            predicted_satellites, _ = discard_contours_outside_contours(predicted_nuclei, predicted_satellites)

        record = {
            "source_image": Path(ground_truth_file_path).name,
            "expected_nuclei": len(expected_nuclei),
            "expected_clusters": len(expected_clusters),
            "expected_satellites": len(expected_satellites)
        }
        expected_records = expected_records.append(record, ignore_index=True)

        for i, predicted_nucleus in enumerate(predicted_nuclei):
            nucleus_intersected = False
            for expected_nucleus in expected_nuclei:
                intersection = get_intersection(expected_nucleus, predicted_nucleus, shape=ground_truth.shape[:2])
                if np.round(intersection, intersection_precision) > 0:
                    record = {
                        "source_image": Path(ground_truth_file_path).name,
                        "nucleus": i,
                        "iou": intersection,
                        "type": "True positive",
                    }
                    nucleus_records = nucleus_records.append(record, ignore_index=True)
                    nucleus_intersected = True

                    j = -1
                    for predicted_cluster in predicted_clusters:
                        predicted_cluster, _ = discard_contours_outside_contours([predicted_nucleus], [predicted_cluster])
                        if len(predicted_cluster) == 0:
                            continue
                        j += 1
                        cluster_intersected = False
                        for expected_cluster in expected_clusters:
                            intersection = get_intersection(expected_cluster, predicted_cluster[0], shape=ground_truth.shape[:2])
                            if np.round(intersection, intersection_precision) > 0:
                                record = {
                                    "source_image": Path(ground_truth_file_path).name,
                                    "nucleus": i,
                                    "cluster": j,
                                    "iou": intersection,
                                    "type": "True positive"
                                }
                                cluster_records = cluster_records.append(record, ignore_index=True)
                                cluster_intersected = True
                                break

                        if not cluster_intersected:
                            record = {
                                "source_image": Path(ground_truth_file_path).name,
                                "nucleus": i,
                                "cluster": j,
                                "iou": 0.0,
                                "type": "False positive"
                            }
                            cluster_records = cluster_records.append(record, ignore_index=True)

                    j = -1
                    for predicted_satellite in predicted_satellites:
                        predicted_satellite, _ = discard_contours_outside_contours([predicted_nucleus], [predicted_satellite])
                        if len(predicted_satellite) == 0:
                            continue
                        j += 1
                        satellite_intersected = False
                        for expected_satellite in expected_satellites:
                            intersection = get_intersection(expected_satellite, predicted_satellite[0], shape=ground_truth.shape[:2])
                            if np.round(intersection, intersection_precision) > 0:
                                record = {
                                    "source_image": Path(ground_truth_file_path).name,
                                    "nucleus": i,
                                    "satellite": j,
                                    "iou": intersection,
                                    "type": "True positive"
                                }
                                satellite_records = satellite_records.append(record, ignore_index=True)
                                satellite_intersected = True
                                break

                        if not satellite_intersected:
                            record = {
                                "source_image": Path(ground_truth_file_path).name,
                                "nucleus": i,
                                "satellite": j,
                                "iou": 0.0,
                                "type": "False positive"
                            }
                            satellite_records = satellite_records.append(record, ignore_index=True)
                    break

            if not nucleus_intersected:
                record = {
                    "source_image": Path(ground_truth_file_path).name,
                    "nucleus": i,
                    "iou": intersection,
                    "type": "False positive"
                }
                nucleus_records = nucleus_records.append(record, ignore_index=True)

        for i, expected_nucleus in enumerate(expected_nuclei):
            nucleus_intersected = False
            for predicted_nucleus in predicted_nuclei:
                intersection = get_intersection(expected_nucleus, predicted_nucleus, shape=ground_truth.shape[:2])
                if np.round(intersection, intersection_precision) > 0:
                    nucleus_intersected = True

                    j = cluster_records.loc[(cluster_records["source_image"] == Path(ground_truth_file_path).name) & (cluster_records["nucleus"] == i)]["cluster"].max()
                    j = 0 if j != j else j
                    for expected_cluster in expected_clusters:
                        expected_cluster, _ = discard_contours_outside_contours([expected_nucleus], [expected_cluster])
                        if len(expected_cluster) == 0:
                            continue
                        j += 1
                        cluster_intersected = False
                        for predicted_cluster in predicted_clusters:
                            intersection = get_intersection(expected_cluster[0], predicted_cluster, shape=ground_truth.shape[:2])
                            if np.round(intersection, intersection_precision) > 0:
                                cluster_intersected = True
                                break

                        if not cluster_intersected:
                            cluster = cluster_records.loc[(cluster_records["source_image"] == Path(ground_truth_file_path).name) & (cluster_records["nucleus"] == i)]["cluster"].max() + 1
                            cluster = 0 if cluster != cluster else cluster
                            record = {
                                "source_image": Path(ground_truth_file_path).name,
                                "nucleus": i,
                                "cluster": cluster,
                                "iou": 0.0,
                                "type": "False negative"
                            }
                            cluster_records = cluster_records.append(record, ignore_index=True)

                    j = satellite_records.loc[(satellite_records["source_image"] == Path(ground_truth_file_path).name) & (satellite_records["nucleus"] == i)]["satellite"].max()
                    j = 0 if j != j else j
                    for expected_satellite in expected_satellites:
                        expected_satellite, _ = discard_contours_outside_contours([expected_nucleus], [expected_satellite])
                        if len(expected_satellite) == 0:
                            continue
                        j += 1
                        satellite_intersected = False
                        for predicted_satellite in predicted_satellites:
                            intersection = get_intersection(expected_satellite[0], predicted_satellite, shape=ground_truth.shape[:2])
                            if np.round(intersection, intersection_precision) > 0:
                                satellite_intersected = True
                                break

                        if not satellite_intersected:
                            satellite = satellite_records.loc[(satellite_records["source_image"] == Path(ground_truth_file_path).name) & (satellite_records["nucleus"] == i)]["satellite"].max() + 1,
                            if isinstance(satellite, tuple):
                                satellite = satellite[0]
                            satellite = 0 if satellite != satellite else satellite
                            record = {
                                "source_image": Path(ground_truth_file_path).name,
                                "nucleus": i,
                                "satellite": satellite,
                                "iou": 0.0,
                                "type": "False negative"
                            }
                            satellite_records = satellite_records.append(record, ignore_index=True)
                    break

            if not nucleus_intersected:
                record = {
                    "source_image": Path(ground_truth_file_path).name,
                    "nucleus": i,
                    "iou": 0.0,
                    "type": "False negative"
                }
                nucleus_records = nucleus_records.append(record, ignore_index=True)

    print(f"expected_nuclei_count: {expected_nuclei_count}")
    print(f"expected_cluster_count: {expected_cluster_count}")
    print(f"expected_satellite_count: {expected_satellite_count}")
    print(f"predicted_nuclei_count: {predicted_nuclei_count}")
    print(f"predicted_cluster_count: {predicted_cluster_count}")
    print(f"predicted_satellite_count: {predicted_satellite_count}")

    nucleus_file = Path(output_qualification).joinpath("nucleus_iou.csv")
    cluster_file = Path(output_qualification).joinpath("cluster_iou.csv")
    satellite_file = Path(output_qualification).joinpath("satellite_iou.csv")
    expected_file = Path(output_qualification).joinpath("expected_values.csv")

    Path(output_qualification).mkdir(exist_ok=True, parents=True)

    if output_visualization is not None:
        output_visualization = Path(output_visualization)
        Path(output_visualization).mkdir(exist_ok=True, parents=True)

    if nucleus_file.is_file():
        nucleus_records.to_csv(str(nucleus_file), mode="a", header=False, index=False)
    else:
        nucleus_records.to_csv(str(nucleus_file), mode="w", header=True, index=False)

    if cluster_file.is_file():
        cluster_records.to_csv(str(cluster_file), mode="a", header=False, index=False)
    else:
        cluster_records.to_csv(str(cluster_file), mode="w", header=True, index=False)

    if satellite_file.is_file():
        satellite_records.to_csv(str(satellite_file), mode="a", header=False, index=False)
    else:
        satellite_records.to_csv(str(satellite_file), mode="w", header=True, index=False)

    if expected_file.is_file():
        expected_records.to_csv(str(expected_file), mode="a", header=False, index=False)
    else:
        expected_records.to_csv(str(expected_file), mode="w", header=True, index=False)

    return data
