import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import splev, splprep

from utils.utils import color_classes


NUCLEUS_COLUMNS = [
    "patient_id",
    "source_image",
    "class",
    "flag",
    "nucleus",
    "nucleus_pixel_count"]

NOR_COLUMNS = [
    "patient_id",
    "source_image",
    "class",
    "flag",
    "nucleus",
    "nor",
    "nor_pixel_count"]

CLASSES = [
    "control",
    "leukoplakia",
    "carcinoma",
    "unknown"]


def smooth_contours(contours: List[np.ndarray], points: Optional[int] = 30) -> List[np.ndarray]:
    """Smooth a list of contours using a B-spline approximation.

    Args:
        contours (List[np.ndarray]): The contours to be smoothed.
        points (Optional[int], optional): The number of points the smoothed contour should have. Defaults to 30.

    Returns:
        List[np.ndarray]: The smoothed contours.
    """
    smoothened_contours = []
    for contour in contours:
        try:
            x, y = contour.T

            # Convert from numpy arrays to normal arrays
            x = x.tolist()[0]
            y = y.tolist()[0]

            # Find the B-spline representation of an N-dimensional curve.
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
            tck, u = splprep([x, y], u=None, s=1.0, per=1, k=1)

            # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
            u_new = np.linspace(u.min(), u.max(), points)

            # Given the knots and coefficients of a B-spline representation, evaluate the value of the smoothing polynomial and its derivatives.
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
            x_new, y_new = splev(u_new, tck, der=0)

            # Convert it back to Numpy format for OpenCV to be able to display it
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
            smoothened_contours.append(np.asarray(res_array, dtype=np.int32))
        except Exception as e:
            print(f"The smoothing of a contour caused a failure: {e}")
    return smoothened_contours


def get_contours(mask: np.ndarray) -> List[np.ndarray]:
    """Find the contours in a binary segmentation mask.

    Args:
        mask (np.ndarray): The segmentation mask.

    Returns:
        List[np.ndarray]: The list of contours found in the mask.
    """
    mask = mask.copy()
    mask[mask > 0] = 255
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_contour_pixel_count(contour: np.ndarray, shape: List[np.ndarray]) -> int:
    """Counts the number of pixels in a given contour.

    Args:
        contour (np.ndarray): The contour to be evaluated.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extrated, in the format `(HEIGHT, WIDTH)`.

    Returns:
        int: The number of pixels in the contour.
    """
    image = np.zeros(shape)
    cv2.drawContours(image, contours=[contour], contourIdx=-1, color=1, thickness=cv2.FILLED)
    return int(image.sum())


def dilate_contours(
    contours: List,
    structuring_element: Optional[int] = cv2.MORPH_ELLIPSE,
    kernel: Optional[Tuple[int, int]] = (3, 3),
    iterations: Optional[int] = 1,
    shape: Optional[Tuple[int, int]]= None) -> np.ndarray:
    """Dilate a list of contours using the specified morphological operator and kernel size.

    Args:
        contours (List): The list of contours to be dilated.
        structuring_element (Optional[int], optional): The morphological transform to apply. Defaults to `cv2.MORPH_ELLIPSE`.
        kernel (Optional[Tuple[int, int]], optional): The size of the kernel to be used. Defaults to `(3, 3)`.
        iterations (Optional[int], optional): Number of iterations to apply the dilatation. Defaults to 1.
        shape (Optional[Tuple[int, int]], optional): The dimensions of the image from where the contours were extrated, in the format `(HEIGHT, WIDTH)`.

    Returns:
        np.ndarray: _description_
    """
    if shape is not None:
        mask = np.zeros(shape, dtype=np.uint8)
    else:
        max_value = 0
        for contour in contours:
            max_value = contour.max() if contour.max() > max_value else max_value
        mask = np.zeros((max_value * 2, max_value * 2), dtype=np.uint8)

    mask = cv2.drawContours(mask, contours=contours, contourIdx=-1, color=[255, 255, 255], thickness=cv2.FILLED)
    kernel = cv2.getStructuringElement(structuring_element, kernel)
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    return mask


def discard_contours_by_size(
    contours: List[np.ndarray],
    shape: Tuple[int, int],
    min_pixel_count: Optional[int] = 5000,
    max_pixel_count: Optional[int] = 40000) -> Union[List[np.ndarray], List[np.ndarray]]:
    """Discards contours smaller or bigger than the given thresholds.

    Args:
        contours (List[np.ndarray]): The contours to be evaluated.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extrated, in the format `(HEIGHT, WIDTH)`.
        min_pixel_count (Optional[int], optional): The minimum number of pixels the contour must have. Defaults to 5000.
        max_pixel_count (Optional[int], optional): The maximum number of pixels the contour must have. Defaults to 40000.

    Returns:
        Union[List[np.ndarray], List[np.ndarray]]: The `kept` array contains the contours that are withing the size specification. The `discarded` array contains the contours that are not withing the size specification.
    """
    kept = []
    discarded = []
    for contour in contours:
        # contour_area = cv2.contourArea(contour)
        contour_area = get_contour_pixel_count(contour, shape=shape)
        if min_pixel_count <= contour_area and contour_area <= max_pixel_count:
            kept.append(contour)
        else:
            discarded.append(contour)
    return kept, discarded


def discard_contours_without_contours(
    parent_contours: List[np.ndarray],
    child_contours: List[np.ndarray]) -> Union[List[np.ndarray], List[np.ndarray]]:
    """Discards contours that do not have other contours inside.

    Args:
        parent_contours (List[np.ndarray]): The list of contours to be considered as the parent contours.
        child_contours (List[np.ndarray]): The list of contours to be considered as child contours of the parent contours.

    Returns:
        Union[List[np.ndarray], List[np.ndarray]]: The `kept` array contains the contours that have other contours inside. The `discarded` array contains the contours that do not have other contours inside.
    """
    kept = []
    discarded = []
    for parent in parent_contours:
        keep_parent = False
        for child in child_contours:
            for child_point in child:
                if cv2.pointPolygonTest(parent, tuple(child_point[0]), False) >= 0:
                    keep_parent = True
        if keep_parent:
            kept.append(parent)
        else:
            discarded.append(parent)
    return kept, discarded


def discard_contours_outside_contours(
    parent_contours: List[np.ndarray],
    child_contours: List[np.ndarray]) -> Union[List[np.ndarray], List[np.ndarray]]:
    """Discards contours that are outside other contours.

    This function iterates over all child contours and checks if at least one point is inside of at least one parent contour.

    Args:
        parent_contours (List[np.ndarray]): The list of contours to be considered as the parent contours.
        child_contours (List[np.ndarray]): The list of contours to be considered as child contours of the parent contours.

    Returns:
        Union[List[np.ndarray], List[np.ndarray]]: The `kept` array contains the child contours that are within a parent contour. The `discarded` array contains the child contours that are not in any parent contour.
    """
    kept = []
    discarded = []
    for child in child_contours:
        keep_child = False
        for parent in parent_contours:
            for child_point in child:
                if cv2.pointPolygonTest(parent, tuple(child_point[0]), False) >= 0:
                    keep_child = True
                    break
            if keep_child:
                break
        if keep_child:
            kept.append(child)
        else:
            discarded.append(child)
    return kept, discarded


def discard_overlapping_deformed_contours(
    contours: List[np.ndarray],
    shape: Tuple[int, int],
    diff: Optional[int] = 1000) -> Union[List[np.ndarray], List[np.ndarray]]:
    """Discards contours overlapping with other and defformed contours.

    This function verifies if contours are overlapping with others by computing the difference in the number of pixels between the contour and the convex hull of that contour.
    If the difference exceeds the given threshold (`diff`), then the contour is discarded as it is likely overlapping with another one.
    This assumes the nature of the objects being segmented, which are nuclei and NORs in AgNOR-stained images, which tend to have circular shapes.
    Deformed contours caused by obstruction or fragmented segmentation also get discarded under the same criterion.

    Args:
        contours (List[np.ndarray]): The list of contours to be evaluated.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extrated, in the format `(HEIGHT, WIDTH)`.
        diff (Optional[int], optional): The mim number of pixel difference between the contour and its convex hull. If the difference is over `diff`, then the contour is discarded. Defaults to 1000.

    Returns:
        Union[List[np.ndarray], List[np.ndarray]]: The `kept` array contains the contours that are not overlapping with other or are not deformed. The `discarded` array contains the overlapping and deforemed nuclei.
    """
    kept = []
    discarded = []
    for contour in contours:
        contour_pixel_count = get_contour_pixel_count(contour, shape)
        contour_convex_pixel_count = get_contour_pixel_count(cv2.convexHull(contour), shape)
        if contour_convex_pixel_count - contour_pixel_count < diff:
            kept.append(contour)
        else:
            discarded.append(contour)
    return kept, discarded


def draw_contour_lines(image: np.ndarray, contours: List[np.ndarray], type: Optional[str] = "multiple") -> np.ndarray:
    """Draw the line of contours.

    Args:
        image (np.ndarray): The image to draw the contours on.
        contours (List[np.ndarray]): The list of the contours to be drawn.
        type (Optional[str]): The type of contorus to draw. If `multiple`, draws the segmented contour, the convex hull contour and the overlapp between the segmented and convex contours. If `single`, draws only the segmented contour. Defaults to "multiple".

    Raises:
        ValueError: If `type` is not in [`multiple`, `single`].

    Returns:
        np.ndarray: The image with the contours drawn on it.
    """
    if type == "multiple":
        for discarded_contour in contours:
            contour = np.zeros(image.shape, dtype=np.uint8)
            contour_convex = np.zeros(image.shape, dtype=np.uint8)

            cv2.drawContours(contour, contours=[discarded_contour], contourIdx=-1, color=[1, 1, 1], thickness=1)
            cv2.drawContours(
                contour_convex, contours=[cv2.convexHull(discarded_contour)], contourIdx=-1, color=[1, 1, 1], thickness=1)

            diff = contour + contour_convex
            diff[diff < 2] = 0
            diff[diff == 2] = 1

            # Yellow = Smoothed contour
            cv2.drawContours(image, contours=[discarded_contour], contourIdx=-1, color=[255, 255, 0], thickness=1)
            # Cyan = Convex hull of the smoothed contour
            cv2.drawContours(
                image, contours=[cv2.convexHull(discarded_contour)], contourIdx=-1, color=[0, 255, 255], thickness=1)

            # White = Smoothed contour equals to Convex hull of the smoothed contour
            image = np.where(diff > 0, [255, 255, 255], image)
    elif type == "single":
        image = cv2.drawContours(image, contours=contours, contourIdx=-1, color=[255, 255, 255], thickness=1)
    else:
        raise ValueError("Argument `type` must be either `multiple` or `single`.")

    return image.astype(np.uint8)


def analyze_contours(
    mask: Union[np.ndarray, tf.Tensor],
    smooth: Optional[bool] = False) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Analyze the segmented contours and updates the segmentation mask.

    Args:
        mask (Union[np.ndarray, tf.Tensor]): The mask containing the contours to be analyzed.
        smooth (Optional[bool], optional): Whether or not to smooth the contours..

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]: The first tuple contrains the updated mask, and the nuclei and NORs contours. The second one contains mask with the discarded nuclei contours, and the discarded nuclei and NORs contours.
        and the detail mask showing discarded objects and their contour analysis.
    """
    # Obtain and filter nuclei and NORs contours
    nuclei_contours = get_contours(mask[:, :, 1] + mask[:, :, 2])
    nuclei_contours, nuclei_size_discarded = discard_contours_by_size(nuclei_contours, shape=mask.shape[:2])

    nors_contours = get_contours(mask[:, :, 2])
    # nors_contours, _ = discard_contours_by_size(nors_contours, shape=mask.shape[:2])

    if smooth:
        nuclei_contours = smooth_contours(nuclei_contours, points=40)
        nors_contours = smooth_contours(nors_contours, 16)

    nuclei_with_nors, nuclei_without_nors = discard_contours_without_contours(nuclei_contours, nors_contours)
    nuclei_contours_adequate, nuclei_overlapping_deformed = discard_overlapping_deformed_contours(
        nuclei_with_nors, shape=mask.shape[:2])

    nors_in_adequate_nuclei, _ = discard_contours_outside_contours(nuclei_contours_adequate, nors_contours)
    nors_in_overlapping_deformed, _ = discard_contours_outside_contours(nuclei_overlapping_deformed, nors_contours)

    # Create a new mask with the filtered nuclei and NORs
    pixel_intensity = int(np.max(np.unique(mask)))
    background = np.ones(mask.shape[:2], dtype=np.uint8)
    nucleus = np.zeros(mask.shape[:2], dtype=np.uint8)
    nor = np.zeros(mask.shape[:2], dtype=np.uint8)

    cv2.drawContours(nucleus, contours=nuclei_contours_adequate, contourIdx=-1, color=pixel_intensity, thickness=cv2.FILLED)
    cv2.drawContours(nor, contours=nors_in_adequate_nuclei, contourIdx=-1, color=pixel_intensity, thickness=cv2.FILLED)

    nucleus = np.where(nor, 0, nucleus)
    background = np.where(np.logical_and(nucleus == 0, nor == 0), pixel_intensity, 0)
    updated_mask = np.stack([background, nucleus, nor], axis=2).astype(np.uint8)

    contour_detail = mask.copy()
    contour_detail = color_classes(contour_detail)

    if len(nuclei_size_discarded) > 0:
        contour_detail = draw_contour_lines(contour_detail, nuclei_size_discarded, type="single")

    if len(nuclei_without_nors) > 0:
        contour_detail = draw_contour_lines(contour_detail, nuclei_without_nors, type="single")

    if len(nuclei_overlapping_deformed) > 0:
        contour_detail = draw_contour_lines(contour_detail, nuclei_overlapping_deformed)
    else:
        nuclei_overlapping_deformed, nors_in_overlapping_deformed = [], []
        if len(nuclei_size_discarded) == 0 and len(nuclei_without_nors) == 0:
            contour_detail = None

    return (updated_mask, nuclei_contours_adequate, nors_in_adequate_nuclei),\
        (contour_detail, nuclei_overlapping_deformed, nors_in_overlapping_deformed)


def get_contour_measurements(
    parent_contours: List[np.ndarray],
    child_contours: List[np.ndarray],
    shape: Tuple[int, int],
    mask_name: str,
    record_id: Optional[str] = "unknown",
    record_class: Optional[str] = "unknown class",
    start_index: Optional[int] = 0,
    contours_flag: Optional[str] = "valid") -> Union[List[dict], List[dict]]:
    """Calculate the number of pixels per contour and create a record for each of them.

    Args:
        parent_contours (List[np.ndarray]): The parent contours.
        child_contours (List[np.ndarray]): The child contours.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extrated, in the format `(HEIGHT, WIDTH)`.
        mask_name (str): The name of the mask from where the contorus were extracted.
        record_id (Optional[str]): The unique ID of the record. Defaults to "unknown",
        record_class (Optional[str]): The class the record belongs to. Must be one of `["control", "leukoplakia", "carcinoma", "unknown"]`. Defaults to "unknown".
        start_index (Optional[int], optional): The index to start the parent contour ID assignment. Usually it will not be `0` when discarded records are being measure for record purposes. Defaults to 0.
        contours_flag (Optional[str], optional): A string value identifying the characteristis of the record. Usually it will be `valid`, but it can be `discarded` or anything else. Defaults to "valid".

    Returns:
        Union[List[dict], List[dict]]: The parent and child measurements.
    """
    parent_measurements = []
    child_measurements = []

    for parent_id, parent_contour in enumerate(parent_contours, start=start_index):
        parent_pixel_count = get_contour_pixel_count(parent_contour, shape)
        parent_features = [record_id, mask_name, record_class, contours_flag, parent_id, parent_pixel_count]
        parent_measurements.append({ key: value for key, value in zip(NUCLEUS_COLUMNS, parent_features) })

        child_id = 0
        for child_contour in child_contours:
            for child_point in child_contour:
                if cv2.pointPolygonTest(parent_contour, tuple(child_point[0]), False) >= 0:
                    child_pixel_count = get_contour_pixel_count(child_contour, shape)
                    child_features = [
                        record_id, mask_name, record_class, contours_flag, parent_id, child_id, child_pixel_count
                    ]
                    child_measurements.append({ key: value for key, value in zip(NOR_COLUMNS, child_features) })
                    child_id += 1
                    break

    return parent_measurements, child_measurements


def write_contour_measurements(
    parent_measurements: List[dict],
    child_measurements: List[dict],
    output_path: str,
    datetime: Optional[str] = time.strftime('%Y%m%d%H%M%S')) -> None:
    """Writes contour measurements to `.csv` files.

    Args:
        parent_measurements (List[dict]): The parent contours.
        child_measurements (List[dict]): The child contours.
        output_path (str): The path where the files should be written to.
        datetime (Optional[str], optional): A date and time identification for when the files were generated. Defaults to time.strftime('%Y%m%d%H%M%S').
    """
    df_parent = pd.DataFrame(parent_measurements, columns=NUCLEUS_COLUMNS)
    df_child = pd.DataFrame(child_measurements, columns=NOR_COLUMNS)

    df_parent["datetime"] = datetime
    df_child["datetime"] = datetime

    parent_measurements_output = Path(output_path).joinpath(f"nuclei_measurements.csv")
    child_measurements_output = Path(output_path).joinpath(f"nor_measurements.csv")

    if Path(parent_measurements_output).is_file():
        df_parent.to_csv(str(parent_measurements_output), mode="a", header=False, index=False)
    else:
        df_parent.to_csv(str(parent_measurements_output), mode="w", header=True, index=False)

    if Path(child_measurements_output).is_file():
        df_child.to_csv(str(child_measurements_output), mode="a", header=False, index=False)
    else:
        df_child.to_csv(str(child_measurements_output), mode="w", header=True, index=False)
