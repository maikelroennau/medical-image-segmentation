import json
import os
import time
import types
from pathlib import Path

import numpy as np
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import losses
import utils

########
########

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

########
########

seed = 7613
np.random.seed(seed)
tf.random.set_seed(seed)

model_name = "AgNOR"
description = """Experiment description."""

epochs = 3
batch_size = 1
steps_per_epoch = 420

height = 960 # 240 480 960 1920
width = 1280 # 320 640 1280 2560
input_shape = (height, width, 3)

classes = 3
learning_rate = 1e-4
learning_rate_change_factor = 0.75
one_hot_encoded = True if classes > 1 else False
class_distribution = False

train_dataset_path = "dataset/v10/train/"
validation_dataset_path = "dataset/v10/validation/"
test_dataset_path = "dataset/v10/test/"

DECODER = "U-Net" # U-Net FPN Linknet PSPNet
BACKBONE = "resnet34"

loss_function = sm.losses.cce_dice_loss
metrics = [sm.metrics.iou_score, losses.dice_coef]

########
########

train_dataset = utils.load_dataset(
    train_dataset_path,
    batch_size=batch_size,
    target_shape=(height, width),
    repeat=True,
    shuffle=True,
    classes=classes,
    one_hot_encoded=one_hot_encoded,
    validate_masks=True)

validation_dataset = utils.load_dataset(
    validation_dataset_path,
    batch_size=batch_size,
    target_shape=(height, width),
    classes=classes,
    one_hot_encoded=one_hot_encoded,
    validate_masks=True)

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


model = make_model(DECODER, BACKBONE, input_shape, classes, model_name)
model.summary()

########
########

checkpoint_directory = os.path.join("checkpoints", f"{time.strftime('%Y%m%d%H%M%S')}")
os.makedirs(checkpoint_directory, exist_ok=True)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="iou_score", factor=learning_rate_change_factor, min_delta=0.5, patience=10, verbose=1, mode="auto", cooldown=1),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_directory, model_name + "_e{epoch:03d}_l{loss:.4f}_vl{val_loss:.4f}.h5"), monitor="loss", save_best_only=True),
    # tf.keras.callbacks.TensorBoard(log_dir=os.path.join(checkpoint_directory, "logs"), histogram_freq=1, update_freq="batch", write_images=False)
]

########
########

train_config = {
    "model_name": model.name,
    "description": description,
    "backbone": BACKBONE,
    "seed": seed,
    "classes": classes,
    "epochs": epochs,
    "batch_size": batch_size,
    "steps_per_epoch": steps_per_epoch,
    "input_shape": input_shape,
    "one_hot_encoded": one_hot_encoded,
    "loss_fuction": loss_function.__name__ if isinstance(loss_function, types.FunctionType) else loss_function.name,
    "metrics": [metric if isinstance(metric, str) else metric.__name__ for metric in metrics],
    "initial_learning_rate": model.optimizer.get_config()['learning_rate'],
    "learning_rate_change_factor": learning_rate_change_factor,
    "directory": checkpoint_directory,
    "train_dataset": train_dataset_path,
    "validation_dataset": validation_dataset_path,
    "test_dataset": test_dataset_path,
    "train_samples": len(utils.list_files(path=train_dataset_path)[0]),
    "validation_samples": len(utils.list_files(path=validation_dataset_path)[0]),
    "test_samples": len(utils.list_files(path=test_dataset_path)[0])
}

# TODO: Fix 'utils.compute_classes_distribution' function which currently only supports multiclass masks
if class_distribution and classes > 1:
    class_distribution = utils.compute_classes_distribution(
        train_dataset,
        batches=steps_per_epoch // batch_size,
        plot=True,
        output=checkpoint_directory)

    train_config["class_distribution"] = class_distribution

    print("\nClass distribution (pixels):")
    for class_name, value in class_distribution.items():
        print(f"  - {str(round(value, 2)).zfill(5)} ==> {class_name}")

with open(os.path.join(checkpoint_directory, "train_config.json"), "w") as config_file:
    json.dump(train_config, config_file, indent=4)

########
########

start = time.time()
print(f"\nTraining start: {time.strftime('%x %X')}")
print(f"  - Model name: {model.name}")
print(f"  - Decoder: {DECODER}")
print(f"  - Backbone: {BACKBONE}")
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
print(f"  - Checkpoints saved at: {checkpoint_directory}")
print(f"  - Dataset:")
print(f"    - Train: {train_dataset_path}")
print(f"    - Validation: {validation_dataset_path}")
print(f"    - Test: {test_dataset_path}\n")

tf.keras.backend.clear_session()

try:
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        callbacks=callbacks)
except Exception as e:
    print(f"\nThere was an error during training that caused it to stop: \n{e}")
    train_config["error"] = str(e)
    history = None

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
duration = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
train_config["duration"] = duration

print(f"Training start: {time.strftime('%x %X')}")
print(f"Training end: {time.strftime('%x %X')}")
print(f"Duration: {duration}")
print(f"  - Model name: {model.name}")
print(f"  - Decoder: {DECODER}")
print(f"  - Backbone: {BACKBONE}")
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
print(f"  - Checkpoints saved at: {checkpoint_directory}")
print(f"  - Dataset:")
print(f"    - Train: {train_dataset_path}")
print(f"    - Validation: {validation_dataset_path}")
print(f"    - Test: {test_dataset_path}\n")

if history:
    train_config["train_metrics"] = {}
    history_data = history.history
    for k, v in history_data.items():
        v = [float(i) for i in v]
        train_config["train_metrics"][k] = v

with open(os.path.join(checkpoint_directory, "train_config.json"), "w") as config_file:
    json.dump(train_config, config_file, indent=4)

########
########

print(f"\nEvaluate all saved models on test data '{test_dataset_path}'")
best_model, models_metrics = utils.evaluate(
    checkpoint_directory,
    test_dataset_path,
    batch_size,
    input_shape=None,
    classes=classes,
    one_hot_encoded=one_hot_encoded)

if best_model is not None or models_metrics is not None:
    train_config["best_model"] = best_model
    train_config["models_metrics"] = models_metrics
    model_path = best_model["model"]

    with open(os.path.join(checkpoint_directory, "train_config.json"), "w") as config_file:
        json.dump(train_config, config_file, indent=4)

    utils.plot_metrics(history.history, output=checkpoint_directory)
else:
    model_path = None

########
########

print(f"\n\nPredict with best model on test data '{test_dataset_path}'")
utils.predict(
    model_path,
    images_path=Path(test_dataset_path).joinpath("images"),
    batch_size=batch_size,
    output_path=Path(test_dataset_path).joinpath("images"),
    copy_images=False,
    new_input_shape=None,
    normalize=True)
