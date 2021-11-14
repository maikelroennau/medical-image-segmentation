import argparse
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from utils.data_io import list_files, load_dataset
from utils.post_process import plot_metrics
from utils.utils import (CUSTOM_OBJECTS, compute_classes_distribution,
                         evaluate, predict, update_model)


def train(
    model_name="AgNOR",
    description="",
    backbone="vgg16",
    decoder="U-Net",
    dataset="dataset/v15/",
    loss="dice",
    learning_rate=1e-4,
    learning_rate_change_factor=0.75,
    classes=3,
    epochs=10,
    batch_size=1,
    steps_per_epoch=420,
    height=960,  # 240 480 960 1152 1440 1920
    width=1280,  # 320 640 1280 1536 1920 2560
    rgb=True,
    predict_images=False,
    gpu=0,
    save_all=False,
    seed=None,
    resume=None,
    resume_epoch=None
):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if seed:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    description = f"""GPU{gpu}; Backbone '{backbone}' with decoder '{decoder}'; {description}"""

    if rgb:
        input_shape = (height, width, 3)
    else:
        input_shape = (height, width)

    one_hot_encoded = True if classes > 1 else False
    class_distribution = False

    train_dataset_path = f"{Path(dataset).joinpath('train')}"
    validation_dataset_path = f"{Path(dataset).joinpath('validation')}"
    test_dataset_path = f"{Path(dataset).joinpath('test')}"

    if loss == "dice":
        loss_function = sm.losses.cce_dice_loss
    if loss == "focal":
        loss_function = sm.losses.categorical_focal_dice_loss
    elif loss == "categorical":
        loss_function = sm.losses.categorical_crossentropy

    metrics = [sm.metrics.iou_score, sm.metrics.f1_score]

    ########
    ########

    train_dataset = load_dataset(
        train_dataset_path,
        batch_size=batch_size,
        target_shape=(height, width),
        repeat=True,
        shuffle=True,
        classes=classes,
        one_hot_encoded=one_hot_encoded,
        validate_masks=False)

    validation_dataset = load_dataset(
        validation_dataset_path,
        batch_size=1,
        target_shape=(height, width),
        classes=classes,
        one_hot_encoded=one_hot_encoded,
        validate_masks=False)

    ########
    ########

    def make_model(decoder, backbone, input_shape, classes, model_name="AgNOR"):
        if decoder == "U-Net":
            model = sm.Unet(
                backbone_name=backbone,
                input_shape=input_shape,
                classes=classes,
                activation="softmax" if classes > 1 else "sigmoid",
                encoder_weights="imagenet",
                encoder_freeze=False,
                decoder_block_type="transpose",
                decoder_filters=(512, 256, 128, 64, 32),
                decoder_use_batchnorm=True
            )
        elif decoder == "FPN":
            model = sm.FPN(
                backbone_name=backbone,
                input_shape=input_shape,
                classes=classes,
                activation="softmax" if classes > 1 else "sigmoid",
                encoder_weights="imagenet",
                encoder_freeze=False,
                pyramid_block_filters=256,
                pyramid_use_batchnorm=True,
                pyramid_aggregation="concat",
                pyramid_dropout=None
            )
        elif decoder == "Linknet":
            model = sm.Linknet(
                backbone_name=backbone,
                input_shape=input_shape,
                classes=classes,
                activation="softmax" if classes > 1 else "sigmoid",
                encoder_weights="imagenet",
                encoder_freeze=False,
                decoder_filters=(None, None, None, None, 16),
                decoder_use_batchnorm=True,
                decoder_block_type="transpose"
            )
        elif decoder == "PSPNet":
            model = sm.PSPNet(
                backbone_name=backbone,
                input_shape=input_shape,
                classes=classes,
                activation="softmax" if classes > 1 else "sigmoid",
                encoder_weights="imagenet",
                encoder_freeze=False,
                downsample_factor=8,
                psp_conv_filters=512,
                psp_pooling_type="avg",
                psp_use_batchnorm=True,
                psp_dropout=None,
            )

        model._name = model_name
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=metrics)
        return model

    if "," in str(gpu):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = make_model(decoder, backbone, input_shape, classes, model_name)
    else:
        model = make_model(decoder, backbone, input_shape, classes, model_name)

    model.summary()

    ########
    ########

    if resume:
        checkpoint_directory = Path(resume).parent
    else:
        checkpoint_directory = Path("checkpoints").joinpath(f"{time.strftime('%Y%m%d%H%M%S')}")
        checkpoint_directory.mkdir(exist_ok=True, parents=True)

    if save_all:
        checkpoint_model = str(checkpoint_directory.joinpath(model_name + "_e{epoch:03d}_l{loss:.4f}_vl{val_loss:.4f}.h5"))
    else:
        checkpoint_model = str(checkpoint_directory.joinpath(model_name + ".h5"))

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_f1-score", factor=learning_rate_change_factor, min_delta=1e-3, min_lr=1e-8, patience=10, verbose=1, mode="max"),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_model, monitor="val_f1-score", mode="max", save_best_only=False if save_all else True),
        # tf.keras.callbacks.TensorBoard(log_dir=str(checkpoint_directory.joinpath("logs")), histogram_freq=1, update_freq="batch", write_images=False)
    ]

    ########
    ########

    train_config = {
        "model_name": model.name,
        "description": description,
        "backbone": backbone,
        "decoder": decoder,
        "seed": seed,
        "classes": classes,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "input_shape": input_shape,
        "one_hot_encoded": one_hot_encoded,
        "loss_function": loss_function.__name__ if isinstance(loss_function, types.FunctionType) else loss_function.name,
        "metrics": [metric if isinstance(metric, str) else metric.__name__ for metric in metrics],
        "initial_learning_rate": model.optimizer.get_config()['learning_rate'],
        "learning_rate_change_factor": learning_rate_change_factor,
        "directory": checkpoint_directory.name,
        "train_dataset": train_dataset_path,
        "validation_dataset": validation_dataset_path,
        "test_dataset": test_dataset_path,
        "train_samples": len(list_files(path=train_dataset_path)[0]),
        "validation_samples": len(list_files(path=validation_dataset_path)[0]),
        "test_samples": len(list_files(path=test_dataset_path)[0])
    }

    # TODO: Fix 'utils.compute_classes_distribution' function which currently only supports multiclass masks
    if class_distribution and classes > 1:
        class_distribution = compute_classes_distribution(
            train_dataset,
            batches=steps_per_epoch // batch_size,
            plot=True,
            output=str(checkpoint_directory))

        train_config["class_distribution"] = class_distribution

        print("\nClass distribution (pixels):")
        for class_name, value in class_distribution.items():
            print(f"  - {str(round(value, 2)).zfill(5)} ==> {class_name}")

    if resume:
        with open(str(checkpoint_directory.joinpath("train_config.json")), "r") as config_file:
            previous_train_config = json.load(config_file)
            train_config = { **previous_train_config, **train_config }
    else:
        with open(str(checkpoint_directory.joinpath("train_config.json")), "w") as config_file:
            json.dump(train_config, config_file, indent=4)

    ########
    ########

    start = time.time()
    print(f"\nTraining start: {time.strftime('%x %X')}")
    print(f"  - Model name: {model.name}")
    print(f"  - Backbone: {backbone}")
    print(f"  - Decoder: {decoder}")
    print(f"  - Seed: {seed}")
    print(f"  - Classes: {classes}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Input shape: {input_shape}")
    print(f"  - One hot encoded: {one_hot_encoded}")
    print(f"  - Loss function: {loss_function.__name__ if isinstance(loss_function, types.FunctionType) else loss_function.name}")
    print(f"  - Metrics: {[metric if isinstance(metric, str) else metric.__name__ for metric in metrics]}")
    print(f"  - Initial Learning rate: {model.optimizer.get_config()['learning_rate']}")
    print(f"  - Checkpoints saved at: {str(checkpoint_directory)}")
    print(f"  - Dataset:")
    print(f"    - Train: {train_dataset_path}")
    print(f"    - Validation: {validation_dataset_path}")
    print(f"    - Test: {test_dataset_path}\n")

    tf.keras.backend.clear_session()

    try:
        if not resume:
            history = model.fit(
                train_dataset,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_dataset,
                callbacks=callbacks)
        else:
            if "," in str(gpu):
                # strategy = tf.distribute.MirroredStrategy()
                strategy = tf.distribute.MultiWorkerMirroredStrategy()
                with strategy.scope():
                    model = tf.keras.models.load_model(str(Path(resume)), custom_objects=CUSTOM_OBJECTS)
                    if previous_train_config["input_shape"] != input_shape:
                        model = update_model(model, input_shape)
                    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=metrics)

                    history = model.fit(
                      train_dataset,
                      epochs=epochs,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=validation_dataset,
                      initial_epoch=int(resume_epoch),
                      callbacks=callbacks)
            else:
                model = tf.keras.models.load_model(str(Path(resume)), custom_objects=CUSTOM_OBJECTS)
                if previous_train_config["input_shape"] != input_shape:
                    model = update_model(model, input_shape)
                model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=metrics)

                history = model.fit(
                    train_dataset,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_dataset,
                    initial_epoch=int(resume_epoch),
                    callbacks=callbacks)
    except Exception as e:
        print(f"\nThere was an error during training that caused it to stop: \n{e}")
        train_config["error"] = str(e)
        history = None

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    duration = "{:0>2}:{:0>2}:{:02.0f}".format(int(hours), int(minutes), seconds)

    if "duration" in train_config.keys():
        import datetime
        old_duration = datetime.datetime.strptime(train_config["duration"], "%H:%M:%S")
        new_duration = datetime.datetime.strptime(duration, "%H:%M:%S")
        new_duration = datetime.timedelta(minutes=new_duration.minute, seconds=new_duration.second, microseconds=new_duration.microsecond)
        train_config["duration"] = (old_duration + new_duration).strftime("%H:%M:%S")
    else:
        train_config["duration"] = duration

    print(f"Training start: {time.strftime('%x %X')}")
    print(f"Training end: {time.strftime('%x %X')}")
    print(f"Duration: {duration}")
    print(f"  - Model name: {model.name}")
    print(f"  - Backbone: {backbone}")
    print(f"  - Decoder: {decoder}")
    print(f"  - Seed: {seed}")
    print(f"  - Classes: {classes}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Input shape: {input_shape}")
    print(f"  - One hot encoded: {one_hot_encoded}")
    print(f"  - Loss function: {loss_function.__name__ if isinstance(loss_function, types.FunctionType) else loss_function.name}")
    print(f"  - Metrics: {[metric if isinstance(metric, str) else metric.__name__ for metric in metrics]}")
    print(f"  - Initial Learning rate: {model.optimizer.get_config()['learning_rate']}")
    print(f"  - Final learning rate: {model.optimizer.get_config()['learning_rate']}")
    print(f"  - Checkpoints saved at: {str(checkpoint_directory)}")
    print(f"  - Dataset:")
    print(f"    - Train: {train_dataset_path}")
    print(f"    - Validation: {validation_dataset_path}")
    print(f"    - Test: {test_dataset_path}\n")

    if history:
        history_data = history.history

        if "train_metrics" in train_config.keys():
            for k, v in history_data.items():
                v = [float(i) for i in v]
                train_config["train_metrics"][k].extend(v)
        else:
            train_config["train_metrics"] = {}
            for k, v in history_data.items():
                v = [float(i) for i in v]
                train_config["train_metrics"][k] = v

    with open(str(checkpoint_directory.joinpath("train_config.json")), "w") as config_file:
        json.dump(train_config, config_file, indent=4)

    ########
    ########

    print(f"\nEvaluate all saved models on test data '{test_dataset_path}'")
    best_model, models_metrics = evaluate(
        str(checkpoint_directory),
        test_dataset_path,
        batch_size=1,
        loss_function=loss_function,
        input_shape=(1920, 2560, 3),
        classes=classes,
        one_hot_encoded=one_hot_encoded)

    if best_model is not None or models_metrics is not None:
        model_path = best_model["model"]

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

        with open(str(checkpoint_directory.joinpath("train_config.json")), "w") as config_file:
            json.dump(train_config, config_file, indent=4)

        plot_metrics(
            { key: value for key, value in train_config["train_metrics"].items()
              if not key.startswith("val_") and not key.startswith("lr")},
            title="Training metrics",
            output=str(checkpoint_directory.joinpath("01_train.png")))
        plot_metrics(
            { key: value for key, value in train_config["train_metrics"].items() if key.startswith("val_") },
            title="Validation metrics",
            output=str(checkpoint_directory.joinpath("02_validation.png")))
        plot_metrics(
            { key: value for key, value in train_config["test_metrics"].items() },
            title="Test metrics",
            output=str(checkpoint_directory.joinpath("03_test.png")))
        plot_metrics(
            { key: value for key, value in train_config["train_metrics"].items() if key == "lr" },
            title="Learning rate",
            output=str(checkpoint_directory.joinpath("04_learning_rate.png")))
    else:
        model_path = None

    ########
    ########

    if predict_images:
        print(f"\n\nPredict with best model on test data '{test_dataset_path}'")
        predict(
            str(Path(checkpoint_directory).joinpath(model_path)),
            images_path=Path(test_dataset_path).joinpath("images"),
            batch_size=1,
            output_path=Path(test_dataset_path).joinpath("images"),
            copy_images=False,
            new_input_shape=None,
            normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicts using the indicated model on the indicated data.")

    parser.add_argument(
        "--name",
        default="AgNOR",
        help="Name of the model.",
        type=str)

    parser.add_argument(
        "--description",
        required=False,
        default="",
        help="Details about the model.",
        type=str)

    parser.add_argument(
        "--backbone",
        type=str)

    parser.add_argument(
        "--decoder",
        type=str)

    parser.add_argument(
        "--dataset",
        help="""Path to a directory containing 'train', 'validation' and 'test' as sub-directories, each containing an 'images'
                and 'masks' sub-directories.""",
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
        "--predict",
        default=False,
        action="store_true")

    parser.add_argument(
        "--save-all",
        default=False,
        action="store_true")

    parser.add_argument(
        "--gpu",
        default="0",
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

    parser.add_argument(
        "--default",
        default=False,
        help="Ignores arguments and use the default values.",
        action="store_true")

    args = parser.parse_args()

    if args.default:
        train()
    else:
        train(
            model_name=args.name,
            description=args.description,
            backbone=args.backbone,
            decoder=args.decoder,
            dataset=args.dataset,
            loss=args.loss,
            learning_rate=args.lr,
            learning_rate_change_factor=args.lr_factor,
            classes=args.classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps,
            height=args.height,
            width=args.width,
            rgb=args.rgb,
            predict_images=args.predict,
            save_all=args.save_all,
            gpu=args.gpu,
            seed=args.seed,
            resume=args.resume,
            resume_epoch=args.resume_epoch)
