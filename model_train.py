import argparse
import glob
import json
import os
import time
import types
from pathlib import Path
from typing import Optional

import numpy as np
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from utils.data import list_files, load_dataset
from utils.evaluate import evaluate
from utils.model import METRICS, load_model, make_model
from utils.utils import add_time_delta, get_duration, plot_metrics

sm.set_framework("tf.keras")


def show_train_config(
    train_config: dict,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    duration: Optional[str] = None) -> None:
    """Shows the training configuration details, and starting, ending and duration times.

    Args:
        train_config (dict): The training configuration dictionary.
        start_time (Optional[str], optional): The time the training started. Defaults to None.
        end_time (Optional[str], optional): The time the training ended. Defaults to None.
        duration (Optional[str], optional): The duration of the training. Defaults to None.
    """
    if start_time:
        print(f"\nTraining start: {start_time}")
    if end_time:
        print(f"Training end: {end_time}")
    if duration:
        print(f"Duration: {duration}")

    print(f"  - Directory: {train_config['directory']}")
    print(f"  - Backbone: {train_config['backbone']}")
    print(f"  - Decoder: {train_config['decoder']}")
    print(f"  - Loss function: {train_config['loss_function']}")
    print(f"  - Initial learning rate: {train_config['initial_learning_rate']}")
    print(f"  - Learning rate factor: {train_config['learning_rate_factor']}")
    print(f"  - Classes: {train_config['classes']}")
    print(f"  - Epochs: {train_config['epochs']}")
    print(f"  - Batch size: {train_config['batch_size']}")
    print(f"  - Steps per epoch: {train_config['steps_per_epoch']}")
    print(f"  - Input shape: {train_config['input_shape']}")
    print(f"  - One hot encoded: {train_config['one_hot_encoded']}")
    print(f"  - GPU(s): {train_config['gpu']}")
    print(f"  - Metrics: {train_config['metrics']}")
    print(f"  - Save all weights: {train_config['save_all']}")
    print(f"  - Seed: {train_config['seed']}")

    if train_config["model_name"]:
        print(f"  - Model name: {train_config['model_name']}")

    if train_config["resume"]:
        print(f"  - Resume model: {train_config['resume']}")
        print(f"  - Resume epoch: {train_config['resume_epoch']}")

    print(f"  - Dataset:")
    print(f"    - Train: {train_config['train_dataset']} ({train_config['train_samples']} samples)")
    # print(f"    - Validation: {train_config['validation_dataset']} ({train_config['validation_samples']} samples)")
    print(f"    - Test: {train_config['test_dataset']} ({train_config['test_samples']} samples)\n")


def update_best_model_and_metrics(train_config: dict, best_model: dict, models_metrics: dict) -> dict:
    """Update the best model information and the models metrics in a training config dictionary.

    Args:
        train_config (dict): The training config dictionary.
        best_model (dict): The best model dictionary.
        models_metrics (dict): The models metrics dictionary.

    Returns:
        dict: The updated training config dictionary.
    """
    if "best_model" in train_config.keys():
        if best_model["f1-score"] > train_config["best_model"]["f1-score"]:
            train_config["best_model"] = best_model
    else:
        train_config["best_model"] = best_model

    if "test_metrics" in train_config.keys():
        for k in train_config["test_metrics"].keys():
            train_config["test_metrics"][k].extend(models_metrics[k][len(train_config["test_metrics"][k]):])
    else:
        train_config["test_metrics"] = models_metrics

    return train_config


def train(
    backbone: str,
    decoder: str,
    dataset: str,
    loss: str,
    learning_rate: int,
    learning_rate_factor: float,
    classes: int,
    epochs: int,
    batch_size: int,
    steps_per_epoch: int,
    height: int,
    width: int,
    rgb: bool,
    save_all: bool,
    gpu: str,
    model_name: str,
    description: str,
    seed: int,
    resume: str,
    resume_epoch: int) -> None:
    """Trains a segmentation model using the specified architecture and hyperparameters.

    This function will train a segmentation model using the specified backbone and decoder and hyperparameters.
    The trained model is saved to a directory named `checkpoints`, in a subdirectory with named after the date and time the training started, following this pattern: `YYYYMMDDHHMMSS`.
    A `JSON` file is also saved containing all specifications used to train the model, including paths to the dataset and number of samples in each subset.
    The `JSON` file can be used as an argument to te `document_experiments.py` script to produce a `.csv` file summarizing the experiment. Multiple `JSON` files from different experiments can be processed and be saved to a single `.csv` file.

    Args:
        backbone (str): The backbone to be used in the model's architecture. See the available backbones at `https://github.com/qubvel/segmentation_models`.
        decoder (str): The decoder to be used in the model's architecture. Must be one of `U-Net`, `Linknet`, `FPN`, or `PSPNet`.
        dataset (str): The path to the directory containing the images. It must contain three subdirectories: `train`, `validation`, and `test`. And each subdirectory must contain other two subdirectories: `images`, and `masks`.
        loss (str): The loss function to be used to train the model. Must be one of `dice`, `focal`, or `categorical`.
        learning_rate (str): The learning rate to train the model.
        learning_rate_factor (float): A value to multiple to the learning rate and decrease it during the training.
        classes (int): The number of classes to segment.
        epochs (int): The number of epochs to train the model for.
        batch_size (int): The number of images per batch.
        steps_per_epoch (int): The number of steps per epoch (number of batches per epoch).
        height (int): The height of the images.
        width (int): The width of the images.
        rgb (bool): Whether or not the dataset images are RGB.
        save_all (bool): Whether or not to save the model weights after each epoch. If `False`, overrites the previous model with the new one if it scores better given a metric.
        gpu (str): What GPUs to use during training. For multi-GPU, use the format `GPU0,GPU1,GPU2,...`. For CPU, pass `-1`.
        model_name (str): The name of the model file.
        description (str): A description describing the trained model.
        seed (int): A seed for reproducibility.
        resume (str): The path to the model to be loaded to continue training.
        resume_epoch (int): The number of the epoch the model was last trained to.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if seed:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    if rgb:
        input_shape = (height, width, 3)
    else:
        input_shape = (height, width)

    one_hot_encoded = True if classes > 1 else False

    if loss == "dice":
        loss_function = sm.losses.cce_dice_loss
    if loss == "focal":
        loss_function = sm.losses.categorical_focal_dice_loss
    elif loss == "categorical":
        loss_function = sm.losses.categorical_crossentropy

    losses = {
        "softmax": loss_function,
        "nuclei_nor_counts": "mse",
    }

    loss_weights = {
        "softmax": 1.0,
        "nuclei_nor_counts": 1.0
    }

    metrics = {
        "softmax": METRICS,
        "nuclei_nor_counts": "mae"
    }

    if resume:
        checkpoint_directory = Path(resume).parent
    else:
        checkpoint_directory = Path("checkpoints").joinpath(f"{time.strftime('%Y%m%d%H%M%S')}")
        checkpoint_directory.mkdir(exist_ok=True, parents=True)

    if save_all:
        # checkpoint_model = str(checkpoint_directory.joinpath(model_name + "_e{epoch:03d}_l{loss:.4f}_vl{val_loss:.4f}.h5"))
        checkpoint_model = str(checkpoint_directory.joinpath(model_name + "_e{epoch:03d}_l{softmax_loss:.4f}.h5"))
    else:
        checkpoint_model = str(checkpoint_directory.joinpath(model_name + ".h5"))

    train_dataset_path = Path(dataset).joinpath('train')
    # validation_dataset_path = Path(dataset).joinpath('validation')
    test_dataset_path = Path(dataset).joinpath('test')

    train_dataset = load_dataset(
        dataset_path=str(train_dataset_path),
        batch_size=batch_size,
        shape=(height, width),
        classes=classes,
        mask_one_hot_encoded=one_hot_encoded,
        repeat=True,
        shuffle=True
    )

    # validation_dataset = load_dataset(
    #     dataset_path=str(validation_dataset_path),
    #     batch_size=1,
    #     shape=(height, width),
    #     classes=classes,
    #     mask_one_hot_encoded=one_hot_encoded,
    #     repeat=False,
    #     shuffle=True
    # )

    train_config_path = str(checkpoint_directory.joinpath(f"train_config_{checkpoint_directory.name}.json"))
    train_config = {
        "directory": checkpoint_directory.name,
        "backbone": backbone,
        "decoder": decoder,
        "loss_function": loss_function.__name__ if isinstance(loss_function, types.FunctionType) else loss_function.name,
        "initial_learning_rate": learning_rate,
        "learning_rate_factor": learning_rate_factor,
        "classes": classes,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "input_shape": input_shape,
        "one_hot_encoded": one_hot_encoded,
        "gpu": str(gpu),
        "metrics": [metric if isinstance(metric, str) else metric.__name__ for metric in metrics],
        "train_dataset": str(train_dataset_path),
        # "validation_dataset": str(validation_dataset_path),
        "test_dataset": str(test_dataset_path),
        "train_samples": len(list_files(train_dataset_path.joinpath("images"), as_numpy=True)),
        # "validation_samples": len(list_files(validation_dataset_path.joinpath("images"), as_numpy=True)),
        "test_samples": len(list_files(test_dataset_path.joinpath("images"), as_numpy=True)),
        "save_all": save_all,
        "model_name": model_name,
        "description": description,
        "seed": seed,
        "resume": resume,
        "resume_epoch": resume_epoch,
    }

    if resume:
        with open(train_config_path, "r") as config_file:
            previous_train_config = json.load(config_file)
            train_config = { **previous_train_config, **train_config }
    else:
        with open(train_config_path, "w") as config_file:
            json.dump(train_config, config_file, indent=4)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="softmax_f1-score", factor=learning_rate_factor, min_delta=1e-3, min_lr=1e-8, patience=10, verbose=1, mode="max"),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_model, monitor="softmax_f1-score", mode="max", save_best_only=False if save_all else True),
        # tf.keras.callbacks.TensorBoard(
        #     log_dir=str(checkpoint_directory.joinpath("logs")), histogram_freq=1, update_freq="batch", write_images=False)
    ]

    start = time.time()
    start_time = time.strftime('%x %X')

    try:
        if "," in str(gpu):
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            with strategy.scope():
                if resume:
                    model = load_model(
                        model_path=str(Path(resume)),
                        input_shape=input_shape,
                        loss_function=loss_function,
                        optimizer=Adam(learning_rate=learning_rate))
                    model.summary()
                    show_train_config(train_config, start_time)

                    history = model.fit(
                        train_dataset,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        # validation_data=validation_dataset,
                        initial_epoch=int(resume_epoch),
                        callbacks=callbacks)
                else:
                    model = make_model(backbone, decoder, input_shape, classes, learning_rate, loss_function, metrics, model_name)
                    model.summary()
                    show_train_config(train_config, start_time)

                    history = model.fit(
                        train_dataset,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        # validation_data=validation_dataset,
                        callbacks=callbacks)
        else:
            if resume:
                model = load_model(
                    model_path=str(Path(resume)),
                    input_shape=input_shape,
                    loss_function=loss_function,
                    optimizer=Adam(learning_rate=learning_rate))

                flatten = tf.keras.layers.Flatten()(model.layers[594].output)
                count_layer = tf.keras.layers.Dense(2, name="nuclei_nor_counts")(flatten)
                model = tf.keras.Model(
                    name=model.name,
                    inputs=[model.input],
                    outputs=[model.layers[-1].output, count_layer]
                )

                model.compile(optimizer=Adam(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights, metrics=metrics)

                model.summary()
                show_train_config(train_config, start_time)

                history = model.fit(
                    train_dataset,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    # validation_data=validation_dataset,
                    initial_epoch=int(resume_epoch),
                    callbacks=callbacks)
            else:
                model = make_model(backbone, decoder, input_shape, classes, learning_rate, loss_function, metrics, model_name)
                model.summary()
                show_train_config(train_config, start_time)

                history = model.fit(
                    train_dataset,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    # validation_data=validation_dataset,
                    callbacks=callbacks)
    except Exception as e:
        print(f"\nThere was an error during training that caused it to stop: \n{e}")
        train_config["error"] = str(e)
        history = None

    end = time.time()
    end_time = time.strftime('%x %X')
    duration = get_duration(start, end)

    if "duration" in train_config.keys():
        duration = add_time_delta(train_config["duration"], duration)

    train_config["duration"] = duration
    show_train_config(train_config, start_time, end_time, duration)

    if history:
        history_data = history.history

        if "train_metrics" in train_config.keys():
            for k, v in history_data.items():
                v = [float(i) for i in v]
                if k in train_config["train_metrics"].keys():
                    train_config["train_metrics"][k].extend(v)
                else:
                    nan_history = np.zeros(epochs)
                    nan_history[-len(v):] = v
                    train_config["train_metrics"][k] = list(nan_history)
        else:
            train_config["train_metrics"] = {}
            for k, v in history_data.items():
                v = [float(i) for i in v]
                train_config["train_metrics"][k] = v

    with open(train_config_path, "w") as config_file:
        json.dump(train_config, config_file, indent=4)

    if len(glob.glob(str(checkpoint_directory.joinpath("*.h5")))) > 0:
        print(f"\nEvaluate all saved models on test data '{str(test_dataset_path)}'")
        best_model, models_metrics = evaluate(
            models_paths=str(checkpoint_directory),
            images_path=str(test_dataset_path),
            batch_size=1,
            classes=classes,
            one_hot_encoded=one_hot_encoded,
            input_shape=(1920, 2560, 3), # TODO: Update to be the same value specified in the arguments.
            loss_function=losses,
            model_name=model_name)

        if best_model is not None or models_metrics is not None:
            train_config = update_best_model_and_metrics(train_config, best_model, models_metrics)

            with open(train_config_path, "w") as config_file:
                json.dump(train_config, config_file, indent=4)

            plot_metrics(train_config_path)
    else:
        print(f"\nSkipping evaluation, no models were found at `{str(checkpoint_directory)}`.")


def main():
    parser = argparse.ArgumentParser(description="Train a model accordingly to the arguments.")

    parser.add_argument(
        "--backbone",
        help="The feature extractor of the model.",
        type=str)

    parser.add_argument(
        "--decoder",
        help="The decoder of the model.",
        type=str)

    parser.add_argument(
        "--dataset",
        help="Path to a directory containing 'train', 'validation' and 'test' as sub-directories, each containing an 'images' and 'masks' sub-directories.",
        type=str)

    parser.add_argument(
        "--loss",
        default="dice",
        type=str)

    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float)

    parser.add_argument(
        "--lr-factor",
        default=0.75,
        help="Factor to reduce the learning rate when training is stale.",
        type=float)

    parser.add_argument(
        "--classes",
        default=3,
        type=int)

    parser.add_argument(
        "--epochs",
        type=int)

    parser.add_argument(
        "--batch-size",
        type=int)

    parser.add_argument(
        "--steps",
        type=int)

    parser.add_argument(
        "--height",
        type=int)

    parser.add_argument(
        "--width",
        type=int)

    parser.add_argument(
        "--rgb",
        default=True,
        action="store_true")

    parser.add_argument(
        "--save-all",
        default=False,
        help="Whether or no to save all weights. If `True` saves weights after each epoch, if `False`, saves only if better than the previous epoch.",
        action="store_true")

    parser.add_argument(
        "--gpu",
        default="0",
        type=str)

    parser.add_argument(
        "--name",
        required=False,
        default="AgNOR",
        help="Name of the model.",
        type=str)

    parser.add_argument(
        "--description",
        required=False,
        default="",
        type=str)

    parser.add_argument(
        "--seed",
        default=None,
        type=int)

    parser.add_argument(
        "--resume",
        default=None,
        help="Path to the model to be loaded and trained.",
        type=str)

    parser.add_argument(
        "--resume-epoch",
        default=None,
        help="The last epoch the model trained.",
        type=int)

    args = parser.parse_args()

    train(
        backbone=args.backbone,
        decoder=args.decoder,
        dataset=args.dataset,
        loss=args.loss,
        learning_rate=args.lr,
        learning_rate_factor=args.lr_factor,
        classes=args.classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps,
        height=args.height,
        width=args.width,
        rgb=args.rgb,
        save_all=args.save_all,
        gpu=args.gpu,
        model_name=args.name,
        description=args.description,
        seed=args.seed,
        resume=args.resume,
        resume_epoch=args.resume_epoch,
    )

if __name__ == "__main__":
    main()
