import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import losses


CUSTOM_OBJECTS = {
    "dice_coef": losses.dice_coef,
    "dice_coef_loss": losses.dice_coef_loss,
    "jaccard_index": losses.jaccard_index,
    "jaccard_index_loss": losses.jaccard_index_loss,
    "weighted_categorical_crossentropy": losses.weighted_categorical_crossentropy
}

METRICS = [
    "accuracy",
    losses.dice_coef,
    losses.jaccard_index,
]

def write_dataset(dataset, output_path="dataset_visualization", max_batches=None, same_dir=False):
    output = Path(output_path)
    images_path = output.joinpath("images")

    if same_dir:
        masks_path = output.joinpath("images")
    else:
        masks_path = output.joinpath("masks")

    images_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)

    if max_batches:
        if max_batches > len(dataset):
            batches = len(dataset)
        else:
            batches = max_batches
    else:
        batches = len(dataset)

    for i, batch in tqdm(enumerate(dataset), total=batches):
        for j, (image, mask) in enumerate(zip(batch[0], batch[1])):
            image_name = str(images_path.joinpath(f"batch_{i}_{j}.png"))
            mask_name = str(masks_path.joinpath(f"batch_{i}_{j}.png"))
            tf.keras.preprocessing.image.save_img(image_name, image)
            tf.keras.preprocessing.image.save_img(mask_name, mask)

        tf.keras.backend.clear_session()
        if i == batches:
            break


def load_files(image_path, mask_path, target_shape=(1920, 2560), classes=1, one_hot_encoded=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, target_shape, method="nearest")
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, target_shape, method="nearest")

    if one_hot_encoded:
        mask = tf.cast(mask, dtype=tf.int32)
        mask = tf.one_hot(mask, depth=classes, axis=2, dtype=tf.int32)
        mask = tf.squeeze(mask)

    mask = tf.cast(mask, dtype=tf.float32)

    return image, mask


def load_dataset(path, batch_size=1, target_shape=(1920, 2560), repeat=False, shuffle=False, classes=1, one_hot_encoded=False, validate_masks=False, seed=7613):
    if validate_masks:
        images_path = Path(path).joinpath("images")
        masks_path = Path(path).joinpath("masks")

        supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        images_paths = [image_path for image_path in images_path.glob("*.*") if image_path.suffix.lower() in supported_types and not image_path.stem.endswith("_prediction")]
        masks_paths = [mask_path for mask_path in masks_path.glob("*.*") if mask_path.suffix.lower() in supported_types and not mask_path.stem.endswith("_prediction")]

        images_paths.sort()
        masks_paths.sort()

        assert len(images_paths) == len(masks_paths), f"Different quantity of images ({len(images_paths)}) and masks ({len(masks_paths)})"

        for image_path, mask_path in zip(images_paths, masks_paths):
            assert image_path.stem.lower() == mask_path.stem.lower(), f"Image and mask do not correspond: {image_path.name} <==> {mask_path.name}"

        print(f"Dataset '{str(images_path.parent)}' contains {len(images_paths)} images and masks.")

        images_paths = [str(image_path) for image_path in images_paths]
        masks_paths = [str(masks_path) for masks_path in masks_paths]
        dataset = tf.data.Dataset.from_tensor_slices((images_paths, masks_paths))
    else:
        images_path = Path(path).joinpath("images").joinpath("*.*")
        masks_path = Path(path).joinpath("masks").joinpath("*.*")

        images = tf.data.Dataset.list_files(str(images_path), shuffle=True, seed=seed)
        masks = tf.data.Dataset.list_files(str(masks_path), shuffle=True, seed=seed)
        dataset = tf.data.Dataset.zip((images, masks))
        print(f"Dataset '{str(images_path.parent)}' contains {len(dataset)} images and masks.")

    dataset = dataset.map(lambda image_path, mask_path: load_files(image_path, mask_path, target_shape, classes, one_hot_encoded))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 2, seed=seed)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


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

        loaded_model.compile(optimizer=Adam(lr=1e-5), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[METRICS])

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
        models_metrics = {}
        best_model = {}

        for i, model_path in enumerate(models):
            loaded_model = tf.keras.models.load_model(str(model_path), custom_objects=CUSTOM_OBJECTS)

            if input_shape:
                loaded_model = update_model(loaded_model, input_shape)

            loaded_model.compile(optimizer=Adam(lr=1e-5), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[METRICS])
            evaluation_metrics = loaded_model.evaluate(evaluate_dataset)
            print(f"Model {str(model_path)}")
            print(f"  - Loss: {np.round(evaluation_metrics[0], 4)}")
            for i, evaluation_metric in enumerate(evaluation_metrics[1:]):
                metric = METRICS[i] if isinstance(METRICS[i], str) else METRICS[i].__name__
                print(f"  - {metric}: {np.round(evaluation_metric, 4)}")

            # Add model metrics to dict
            models_metrics[model_path.name] = {}
            models_metrics[model_path.name]["model"] = str(model_path)
            models_metrics[model_path.name]["loss"] = evaluation_metrics[0]
            for i, evaluation_metric in enumerate(evaluation_metrics[1:]):
                metric = METRICS[i] if isinstance(METRICS[i], str) else METRICS[i].__name__
                models_metrics[model_path.name][metric] = evaluation_metric

            # Check for the best model
            if "model" in best_model:
                if evaluation_metrics[0] < best_model["loss"]:
                    best_model["model"] = str(model_path)
                    best_model["loss"] = evaluation_metrics[0]
                    for i, evaluation_metric in enumerate(evaluation_metrics[1:]):
                        metric = METRICS[i] if isinstance(METRICS[i], str) else METRICS[i].__name__
                        best_model[metric] = evaluation_metric
            else:
                best_model["model"] = str(model_path)
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


def predict(model, images_path, batch_size, output_path="predictions", copy_images=False, new_input_shape=None):
    if isinstance(model, str) or isinstance(model, Path):
        model = Path(model)
        if model.is_file():
            loaded_model = tf.keras.models.load_model(str(model), custom_objects=CUSTOM_OBJECTS)
        elif model.is_dir():
            models = [model_path for model_path in model.glob("*.h5")]
            if len(models) > 0:
                print(f"No model(s) found at {str(model)}")
                for model_path in models:
                    predict(model_path, images_path, batch_size, output_path=str(model.joinpath("predictions").joinpath(model_path.name)), copy_images=copy_images, new_input_shape=new_input_shape)
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
        image = tf.image.decode_jpeg(image, channels=3)
        original_shape = image.shape[:2]
        image = tf.image.resize(image, (height, width))

        images_tensor[0, :, :, :] = image

        prediction = loaded_model.predict(images_tensor, batch_size=batch_size, verbose=1)
        prediction = tf.image.resize(prediction[0], original_shape).numpy()

        # prediction[:, :, 0] = 0
        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 255
        cv2.imwrite(os.path.join(output_path, f"{image_path.stem}_{loaded_model.name}_prediction.png"), cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB))

        if copy_images:
            shutil.copyfile(str(image_path), Path(output_path).joinpath(image_path.name))
        tf.keras.backend.clear_session()


def plot_metrics(history, output=".", figsize=(15, 15)):
    import pandas as pd

    if not isinstance(history, dict):
        if not isinstance(history, str):
            print("\nOject type not suported for plotting training history.")
            return
        else:
            if Path(history).is_file():
                with open(str(history)) as json_file:
                    history = json.load(json_file)

    validation_metrics = [key for key in history.keys() if key.startswith("val_")]
    train_metrics = [key.replace("val_", "") for key in validation_metrics]
    history_keys = train_metrics + validation_metrics + ["lr"]

    df_data = { key: history[key] for key in history_keys }
    df = pd.DataFrame(df_data)
    df.index = range(1, len(df.index) + 1)

    output_path = Path(output)
    output_path.mkdir(exist_ok=True, parents=True)

    train_image = df[train_metrics].plot(grid=True, figsize=figsize)
    train_image.set(xlabel="Epoch", title="Train metrics")
    train_image = train_image.get_figure()
    train_image.savefig(output_path.joinpath("01_train_metrics.png"))

    validation_image = df[validation_metrics].plot(grid=True, figsize=figsize)
    validation_image.set(xlabel="Epoch", title="Validation metrics")
    validation_image = validation_image.get_figure()
    validation_image.savefig(output_path.joinpath("02_validation_metrics.png"))

    lr_image = df[["lr"]].plot(grid=True, figsize=figsize)
    lr_image.set(xlabel="Epoch", title="Learning rate")
    lr_image = lr_image.get_figure()
    lr_image.savefig(output_path.joinpath("03_learning_rate.png"))


def compute_classes_weights(dataset, batches=1, plot=True, figsize=(20, 10), output="."):
    class_occurence = []

    for i, batch in enumerate(dataset):
        if i == batches:
            break

        for mask in batch[1]:
            class_count = []
            for class_index in range(mask.shape[-1]):
                class_count.append(tf.math.reduce_sum(mask[:, :, class_index]))
            class_occurence.append(class_count)

    class_occurence = tf.convert_to_tensor(class_occurence, dtype=tf.int64)
    class_occurence_mean = tf.math.reduce_mean(class_occurence, axis=0)

    class_weights = {}
    for i, occurence in enumerate(class_occurence_mean):
        class_weights[i] = float(occurence)

    if plot:
        import pandas as pd

        output_path = Path(output)
        output_path.mkdir(exist_ok=True, parents=True)

        df = pd.DataFrame(class_occurence.numpy(), columns=["Background", "Nucleus", "NOR"])
        class_weights_figure = df.plot.bar(stacked=True, figsize=figsize)
        class_weights_figure.set(xlabel="Image instance", ylabel="Number of pixels", title="Pixel class distribution (dataset)")
        class_weights_figure = class_weights_figure.get_figure()
        class_weights_figure.savefig(output_path.joinpath("classes_distribution.png"))

        df = pd.DataFrame(class_weights.values()).transpose()
        df.columns = ["Background", "Nucleus", "NOR"]
        class_weights_figure = df.plot.bar(stacked=True, figsize=(10, 10))
        class_weights_figure.set(ylabel="Number of pixels", title="Pixel class distribution (dataset mean)")
        class_weights_figure = class_weights_figure.get_figure()
        class_weights_figure.savefig(output_path.joinpath("classes_distribution_dataset_mean.png"))

    return class_weights
