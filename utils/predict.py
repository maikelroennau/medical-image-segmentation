import shutil
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import contour_analysis
from utils.data import list_files, load_image, reset_class_values
from utils.model import get_model_input_shape, load_model
from utils.utils import collapse_probabilities, color_classes


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

        x_range = []
        y_range = []
        start, end = 0, 0
        for x in range(int(np.ceil(height / patch_shape[0]))):
            end = start + patch_shape[0]
            if end > height:
                end = height
                start = end - patch_shape[0]
            x_range.append([start, end])
            start = end
        start, end = 0, 0
        for y in range(int(np.ceil(width / patch_shape[1]))):
            end = start + patch_shape[1]
            if end > width:
                end = width
                start = end - patch_shape[1]
            y_range.append([start, end])
            start = end

        patch_prediction = np.zeros((image.shape[0], image.shape[1], 8), dtype=np.float32)

        for x in x_range:
            for y in y_range:
                patch = image[x[0]:x[1], y[0]:y[1]]
                print(f"{patch.min()}, {patch.max()}")
                patch = patch.reshape((1,) + patch.shape)
                patch_prediction[x[0]:x[1], y[0]:y[1]] = model(patch, training=False)[0].numpy()

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
    bboxes: Optional[str] = None,
    classify_agnor: Optional[bool] = False,
    decision_tree_model_path: Optional[str] = "agnor_classifier.joblib",
    record_id: Optional[str] = None,
    record_class: Optional[str] = None,
    measures_only: Optional[bool] = False,
    current_time: Optional[str] = time.strftime('%Y%m%d%H%M%S'),
    use_bias_layer: Optional[bool] = False,
    papanicolaou_post_process: Optional[bool] = False
    ) -> None:
    """_summary_

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
        bboxes (Optional[str], optional): Path to `labelme` annotations containing bounding boxes that indicate what objects must be considered by the contour analysis algorithm. Defaults to None.
        classify_agnor (Optional[bool], optional): Whether or not to classify AgNORs in `clusters` or `satellites`. Defaults to False.
        decision_tree_model_path (Optional[str], optional): The path to the decision tree model to be employed in the AgNOR classification. Defaults to "agnor_classifier.joblib".
        record_id (Optional[str], optional): An ID that will identify the contour measurements. Defaults to None.
        record_class (Optional[str], optional): The class the contour measurements belong to. Defaults to None.
        measures_only (Optional[bool], optional): Do not save the predicted images or copy the input images to the output path. If `True`, it will override the effect of `output_predictions`. Defaults to False.
        current_time (Optional[str], optional): A timestamp to be added to the contour measurements, in the format `YYYYMMDDHHMMSS`. Defaults to time.strftime('%Y%m%d%H%M%S').
        use_bias_layer (Optional[bool], optional): Whether or not to use the add the PabBias layer the model. Defaults to False.
        papanicolaou_post_process (Optional[bool], optional): Whether or not to apply the Papanicolaou post-processing algorithm. Defaults to False.

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
        model = load_model(model_path=model, input_shape=input_shape, use_bias_layer=use_bias_layer)
    elif not isinstance(model, tf.keras.Model):
        raise ValueError(f"`model` must be a `str` or `tf.keras.Model`. Given `{type(model)}`.")

    if not input_shape:
        input_shape = get_model_input_shape(model)

    if not output_contour_analysis:
        output_contour_analysis = output_predictions

    output_predictions = Path(output_predictions)
    output_predictions.mkdir(exist_ok=True, parents=True)

    files = list(set(files))
    files.sort()

    for file in tqdm(files, desc=record_id):
        image, original_shape = load_image(image_path=file, normalize=normalize, shape=input_shape[:2], as_numpy=True, return_original_shape=True)

        if image.shape != input_shape:
            prediction = patch_predict(model, image, input_shape)
        else:
            batch = image.reshape((1,) + image.shape)
            prediction = model(batch, training=False)[0].numpy()

        if not use_bias_layer and papanicolaou_post_process:
            prediction = contour_analysis.adjust_probability(prediction=prediction.copy())
        
        prediction = collapse_probabilities(prediction=prediction, pixel_intensity=127)

        prediction = cv2.resize(prediction, original_shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        if papanicolaou_post_process:
            prediction = contour_analysis.post_process_papanicolaou(prediction)
            prediction = contour_analysis.remove_objects(prediction=prediction)

        file = Path(file)
        if analyze_contours:
            prediction, detail = contour_analysis.analyze_contours(mask=prediction)

            if bboxes is not None:
                annotation = Path(bboxes).joinpath(f"{Path(file).stem.replace('_mask', '')}.json")
                prediction = contour_analysis.discard_unboxed_contours(*prediction, annotation=annotation)

            prediction, parent_contours, child_contours = prediction
            detail, discarded_parent_contours, discarded_child_contours = detail

            if record_id:
                for i, parent_contour in enumerate(parent_contours):
                    parent_measurements, child_measurements, min_contour_size, max_contour_size = contour_analysis.get_contour_measurements(
                        parent_contours=[parent_contour],
                        child_contours=child_contours,
                        shape=input_shape[:2],
                        mask_name=Path(file).name,
                        record_id=record_id,
                        record_class=record_class,
                        start_index=i)

                    _, child_measurements, _, _ = contour_analysis.get_contour_measurements(
                        parent_contours=[parent_contour],
                        child_contours=child_contours,
                        shape=input_shape[:2],
                        mask_name=Path(file).name,
                        record_id=record_id,
                        record_class=record_class,
                        min_contour_size=min_contour_size,
                        max_contour_size=max_contour_size)

                    if classify_agnor:
                        child_measurements = contour_analysis.classify_agnor(decision_tree_model_path, child_measurements)

                        prediction = np.stack([
                            prediction[:, :, 0],
                            prediction[:, :, 1],
                            prediction[:, :, 2],
                            np.zeros(prediction.shape[:2], dtype=np.uint8)
                        ], axis=2)

                        filtered_child_contour, _ = contour_analysis.discard_contours_outside_contours([parent_contour], child_contours)

                        # OpenCV's `drawContours` fails using array slices, so a new matrix must be created, drawn on and assigned to `predictions`.
                        satellites = prediction[:, :, 3].copy()
                        for classified_measurement, classified_contour in zip(child_measurements, filtered_child_contour):
                            if classified_measurement["agnor_type"] == "satellite":
                                cv2.drawContours(satellites, contours=[classified_contour], contourIdx=-1, color=1, thickness=cv2.FILLED)
                        prediction[:, :, 3] = satellites

                    contour_analysis.write_contour_measurements(
                        parent_measurements=parent_measurements,
                        child_measurements=child_measurements,
                        output_path=output_contour_analysis,
                        datetime=current_time)

                    if len(discarded_parent_contours) > 0 or len(discarded_child_contours) > 0:
                        measurements = contour_analysis.get_contour_measurements(
                            parent_contours=discarded_parent_contours,
                            child_contours=discarded_child_contours,
                            shape=input_shape[:2],
                            mask_name=Path(file).name,
                            record_id=record_id,
                            record_class=record_class,
                            start_index=len(parent_contours) + i,
                            contours_flag="invalid")
                        discarded_parent_measurements, discarded_child_measurements, min_contour_size, max_contour_size = measurements

                        _, discarded_child_measurements, _, _ = contour_analysis.get_contour_measurements(
                            parent_contours=discarded_parent_contours,
                            child_contours=discarded_child_contours,
                            shape=input_shape[:2],
                            mask_name=Path(file).name,
                            record_id=record_id,
                            record_class=record_class,
                            start_index=len(parent_contours) + i,
                            contours_flag="invalid",
                            min_contour_size=min_contour_size,
                            max_contour_size=max_contour_size)

                        if classify_agnor:
                            discarded_child_measurements = contour_analysis.classify_agnor(
                                decision_tree_model_path, discarded_child_measurements)

                        contour_analysis.write_contour_measurements(
                            parent_measurements=discarded_parent_measurements,
                            child_measurements=discarded_child_measurements,
                            output_path=output_contour_analysis,
                            datetime=current_time)

            if detail is not None and not measures_only:
                filtered_objects = output_predictions.joinpath("filtered_objects")
                filtered_objects.mkdir(exist_ok=True, parents=True)

                cv2.imwrite(str(filtered_objects.joinpath(f"{file.stem}_image_detail.png")), cv2.cvtColor(detail, cv2.COLOR_BGR2RGB))

                if copy_images:
                    image = contour_analysis.draw_contour_lines(
                        load_image(image_path=str(file), normalize=False, as_numpy=True),
                        discarded_parent_contours)
                    cv2.imwrite(str(filtered_objects.joinpath(f"{file.stem}_image.png")), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not measures_only:
            if grayscale:
                prediction = reset_class_values(prediction)
            else:
                prediction = color_classes(prediction)
                prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)

            cv2.imwrite(str(output_predictions.joinpath(f"{file.stem}_prediction.png")), prediction)
            if copy_images:
                shutil.copyfile(str(file), output_predictions.joinpath(file.name))
