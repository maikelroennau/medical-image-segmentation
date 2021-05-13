import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

########
########

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

########
########

seed = 1145
tf.random.set_seed(seed)
np.random.seed(seed)

model_name = "AgNOR-NOR"

epochs = 100
batch_size = 1
steps_per_epoch = 600
validation_steps = 10

height = 960 # 240 480 960 1920
width = 1280 # 320 640 1280 2560
input_shape = (height, width, 3)

learning_rate = 1e-5

train_dataset_path = "dataset/nucleus/train/"
validation_dataset_path = "dataset/nucleus/validation/"
test_dataset_path = "dataset/nucleus/test/"

########
########

def data_loader(path, batch_size=32, target_shape=(1920, 2560), seed=1145):
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    images = image_datagen.flow_from_directory(
        directory=Path(path),
        target_size=target_shape,
        classes=["images"],
        class_mode=None,
        color_mode="rgb",
        batch_size=batch_size,
        seed=seed
    )

    masks = mask_datagen.flow_from_directory(
        directory=Path(path),
        target_size=target_shape,
        classes=["masks"],
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        seed=seed
    )

    return images, masks


def data_generator(path, batch_size=32, target_shape=(1920, 2560), seed=1145):
    images, masks = data_loader(path, batch_size, target_shape, seed)
    for images, masks in zip(images, masks):
        yield (images, masks)

########
########

train_dataset = data_generator(train_dataset_path, batch_size=batch_size, target_shape=(height, width))
validation_dataset = data_generator(validation_dataset_path, batch_size=batch_size, target_shape=(height, width))

########
########

def dice_coef(y_true, y_pred, smooth=1.):
    intersection = keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
    union = keras.backend.sum(y_true, axis=[1, 2, 3]) + keras.backend.sum(y_pred, axis=[1, 2, 3])
    return keras.backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)

########
########

data_augmentation = keras.Sequential(
    [
        # layers.experimental.preprocessing.RandomRotation(0.2, seed=seed),
        layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical", seed=seed),
        # layers.experimental.preprocessing.RandomContrast(0.1, seed=seed),
        # layers.experimental.preprocessing.Rescaling(1./255.)

    ]
)

########
########

def make_model(input_shape, model_name="AgNOR"):
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs)

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(conv9)

    model = keras.Model(inputs=[inputs], outputs=[outputs], name=model_name)

    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

model = make_model(input_shape=input_shape, model_name=model_name)
model.summary()

########
########

checkpoint_directory = os.path.join("checkpoints", f"{time.strftime('%Y%m%d%H%M%S')}")
os.makedirs(checkpoint_directory, exist_ok=True)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.25, patience=10, verbose=1,  mode="auto", cooldown=1),
    keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_directory, model_name + "_e{epoch:03d}_l{loss:.4f}_vl{val_loss:.4f}.h5"), monitor="val_dice_coef", save_best_only=False),
    # keras.callbacks.TensorBoard(log_dir=os.path.join(checkpoint_directory, "logs"), histogram_freq=1, update_freq="batch", write_images=False)
]

########
########

train_config = {
    "model_name": model.name,
    "seed": seed,
    "epochs": epochs,
    "batch_size": batch_size,
    "input_shape": input_shape,
    "initial_learning_rate": model.optimizer.get_config()['learning_rate']
}

with open(os.path.join(checkpoint_directory, "train_config.json"), "w") as config_file:
    json.dump(train_config, config_file)

########
########

start = time.time()
print(f"Training start - {time.strftime('%x %X')}")
print(f"  - Model name: {model.name}")
print(f"  - Seed: {seed}")
print(f"  - Epochs: {epochs}")
print(f"  - Batch size: {batch_size}")
print(f"  - Input shape: {input_shape}")
print(f"  - Learning rate: {model.optimizer.get_config()['learning_rate']}")
print(f"  - Checkpoints saved at: {checkpoint_directory}\n")

keras.backend.clear_session()
history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks)

end = time.time()
print(f"\nTraining end - {time.strftime('%x %X')}")
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
duration = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
print(f"Duration: {duration}")
print(f"  - Model name: {model.name}")
print(f"  - Checkpoints saved at: {checkpoint_directory}")
print(f"  - Final learning rate: {model.optimizer.get_config()['learning_rate']}")

train_config["duration"] = duration
history_data = history.history

for k, v in history_data.items():
    v = [float(i) for i in v]
    train_config[k] = v

with open(os.path.join(checkpoint_directory, "train_config.json"), "w") as config_file:
    json.dump(train_config, config_file)

########
########

print("\nModel evaluation")
test_dataset = data_generator(test_dataset_path, batch_size=batch_size, target_shape=(height, width))
loss, dice = model.evaluate(test_dataset)
print("Loss: %.4f" % loss)
print("Dice: %.4f" % dice)

########
########

print("\nTesting model")
test_images_path = str(Path(test_dataset_path).joinpath("images"))
input_shape = model.input_shape[1:]
height, width, channels = input_shape

supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
images = [image_path for image_path in Path(test_images_path).rglob("*.*") if image_path.suffix.lower() in supported_types and not image_path.stem.endswith("_prediction")]
images_tensor = np.empty((1, height, width, channels))

for i, image_path in enumerate(images):
    image = cv2.imread(os.path.join(test_images_path, image_path.name), cv2.IMREAD_COLOR)
    original_shape = image.shape[:2][::-1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    images_tensor[0, :, :, :] = image

    prediction = model.predict(images_tensor, batch_size=1, verbose=1)
    prediction = cv2.resize(prediction[0], original_shape)
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 255
    cv2.imwrite(os.path.join(test_images_path, f"{image_path.stem}_{model.name}_prediction.png"), prediction)
    keras.backend.clear_session()
