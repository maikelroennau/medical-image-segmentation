import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import segmentation_models as sm
from tqdm import tqdm

from utils.data import list_files, load_dataset, load_image, one_hot_encode
from utils.model import METRICS, load_model


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
