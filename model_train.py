import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam

import losses
import utils

########
########

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

########
########

seed = 1145
tf.random.set_seed(seed)
np.random.seed(seed)

model_name = "AgNOR"

epochs = 50
batch_size = 10
steps_per_epoch = 60

height = 960 # 240 480 960 1920
width = 1280 # 320 640 1280 2560
input_shape = (height, width, 3)

classes = 3
learning_rate = 1e-4
one_hot_encoded = True if classes > 1 else False
find_best_model = True

train_dataset_path = "dataset/train/"
validation_dataset_path = "dataset/validation/"
test_dataset_path = "dataset/test/"

########
########

train_dataset = utils.load_dataset(
    train_dataset_path,
    batch_size=batch_size,
    target_shape=(height, width),
    repeat=True,
    shuffle=True,
    classes=classes,
    one_hot_encoded=one_hot_encoded)

validation_dataset = utils.load_dataset(
    validation_dataset_path,
    batch_size=batch_size,
    target_shape=(height, width),
    classes=classes,
    one_hot_encoded=one_hot_encoded)

########
########

def make_model(input_shape, classes, model_name="U-Net"):
    inputs = tf.keras.Input(shape=input_shape)

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = layers.concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv5), conv4], axis=3)
    up6 = BatchNormalization()(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = layers.concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv6), conv3], axis=3)
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = layers.concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv7), conv2], axis=3)
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = layers.concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(conv8), conv1], axis=3)
    up9 = BatchNormalization()(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)
    conv9 = BatchNormalization()(conv9)

    outputs = Conv2D(classes, (1, 1), activation="sigmoid")(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

    model.compile(optimizer=Adam(lr=learning_rate), loss=losses.dice_coef_loss, metrics=[losses.dice_coef])

    return model

model = make_model(input_shape=input_shape, classes=classes, model_name=model_name)
model.summary()

########
########

checkpoint_directory = os.path.join("checkpoints", f"{time.strftime('%Y%m%d%H%M%S')}")
os.makedirs(checkpoint_directory, exist_ok=True)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.25, patience=10, verbose=1,  mode="auto", cooldown=1),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_directory, model_name + "_e{epoch:03d}_l{loss:.4f}_vl{val_loss:.4f}.h5"), monitor="loss", save_best_only=True),
    # tf.keras.callbacks.TensorBoard(log_dir=os.path.join(checkpoint_directory, "logs"), histogram_freq=1, update_freq="batch", write_images=False)
]

########
########

train_config = {
    "model_name": model.name,
    "seed": seed,
    "classes": classes,
    "epochs": epochs,
    "batch_size": batch_size,
    "steps_per_epoch": steps_per_epoch,
    "input_shape": input_shape,
    "one_hot_encoded": one_hot_encoded,
    "initial_learning_rate": model.optimizer.get_config()['learning_rate'],
    "train_dataset": train_dataset_path,
    "validation_dataset": validation_dataset_path,
    "test_dataset": test_dataset_path,
}

with open(os.path.join(checkpoint_directory, "train_config.json"), "w") as config_file:
    json.dump(train_config, config_file)

########
########

start = time.time()
print(f"Training start - {time.strftime('%x %X')}")
print(f"  - Model name: {model.name}")
print(f"  - Seed: {seed}")
print(f"  - Classes: {classes}")
print(f"  - Epochs: {epochs}")
print(f"  - Steps per epoch: {steps_per_epoch}")
print(f"  - Batch size: {batch_size}")
print(f"  - Input shape: {input_shape}")
print(f"  - One hot encoded: {one_hot_encoded}")
print(f"  - Initial Learning rate: {model.optimizer.get_config()['learning_rate']}")
print(f"  - Checkpoints saved at: {checkpoint_directory}")
print(f"  - Dataset:")
print(f"    - Train: {train_dataset_path}")
print(f"    - Validation: {validation_dataset_path}")
print(f"    - Test: {test_dataset_path}\n")

tf.keras.backend.clear_session()

history = model.fit(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_dataset,
    callbacks=callbacks)

end = time.time()
print(f"\nTraining end - {time.strftime('%x %X')}")
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
duration = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
print(f"Duration: {duration}")
print(f"  - Model name: {model.name}")
print(f"  - Classes: {classes}")
print(f"  - Epochs: {epochs}")
print(f"  - Steps per epoch: {steps_per_epoch}")
print(f"  - Batch size: {batch_size}")
print(f"  - Input shape: {input_shape}")
print(f"  - One hot encoded: {one_hot_encoded}")
print(f"  - Checkpoints saved at: {checkpoint_directory}")
print(f"  - Final learning rate: {model.optimizer.get_config()['learning_rate']}")
print(f"  - Dataset:")
print(f"    - Train: {train_dataset_path}")
print(f"    - Validation: {validation_dataset_path}")
print(f"    - Test: {test_dataset_path}\n")

train_config["duration"] = duration
history_data = history.history

for k, v in history_data.items():
    v = [float(i) for i in v]
    train_config[k] = v

with open(os.path.join(checkpoint_directory, "train_config.json"), "w") as config_file:
    json.dump(train_config, config_file)

########
########

if find_best_model:
    print("\nEvaluate all models on test data")
    best_model = utils.evaluate(checkpoint_directory, test_dataset_path, batch_size, input_shape=None, classes=classes, one_hot_encoded=one_hot_encoded)
    model_path = Path(checkpoint_directory).joinpath(best_model["model"])
else:
    model_path = [path for path in Path(checkpoint_directory).glob("*.h5")][-1]
    print("\nEvaluate last model on test data")
    utils.evaluate(model_path, test_dataset_path, batch_size, input_shape=None, classes=classes, one_hot_encoded=one_hot_encoded)

########
########

print("\n\nPredict on test data")
utils.predict(
    model_path,
    images_path=Path(test_dataset_path).joinpath("images"),
    batch_size=batch_size,
    output_path=Path(test_dataset_path).joinpath("images"),
    copy_images=False,
    new_input_shape=None)
