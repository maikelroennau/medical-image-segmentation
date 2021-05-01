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

########
########

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

########
########

seed = 1145
tf.random.set_seed(seed)
np.random.seed(seed)

model_name = "AgNOR-Nucleus"

epochs = 20
batch_size = 1
steps_per_epoch = 480
effective_batches = steps_per_epoch * epochs
effective_images = batch_size * steps_per_epoch

height = 960 # 240 480 960 1920
width = 1280 # 320 640 1280 2560
input_shape = (height, width, 3)

learning_rate = 1e-5

########
########

def load_images_and_masks(path, batch_size=16, target_size=(1920, 2560), seed=1145, augment=False, save_to_dir=None, save_prefix="augmented"):
    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    images_paths = [image_path for image_path in Path(path).joinpath("images").glob("*.*") if image_path.suffix.lower() in supported_types and not image_path.stem.endswith("_prediction")]
    masks_paths = [mask_path for mask_path in Path(path).joinpath("masks").glob("*.*") if mask_path.suffix.lower() in supported_types and not mask_path.stem.endswith("_prediction")]

    assert len(images_paths) == len(masks_paths), f"Different quantity of images ({len(images_paths)}) and masks ({len(masks_paths)})"

    images_paths.sort()
    masks_paths.sort()

    for image, mask in zip(images_paths, masks_paths):
        assert image.stem.lower() == mask.stem.lower(), f"Image and mask do not correspond: {image.name} <==> {image.name}"

    height, width = target_size
    images = np.zeros((len(images_paths), height, width, 3), dtype="float32")
    masks = np.zeros((len(masks_paths), height, width, 1), dtype="float32")

    for i, (image, mask) in enumerate(zip(images_paths, masks_paths)):
        images[i, :, :, :] = keras.preprocessing.image.load_img(image, target_size=target_size)
        masks[i, :, :, 0] = keras.preprocessing.image.load_img(mask, target_size=target_size, color_mode="grayscale")

    randomize = np.random.permutation(len(images))
    images = np.asarray(images)[randomize]
    masks = np.asarray(masks)[randomize]

    print(f"Loaded from '{path}'")
    print(f"  - Images: {len(images)}")
    print(f"  - Masks: {len(masks)}")

    if augment:
        datagen_arguments = dict(
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            rotation_range=10,
            # width_shift_range=0.05,
            # height_shift_range=0.05,
            # shear_range=0.05,
            # zoom_range=0.05,
            fill_mode="reflect",
            horizontal_flip=True,
            vertical_flip=True,
            # rescale=1./255.
        )

        images_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_arguments)
        # images_datagen.fit(images)

        # datagen_arguments.pop("featurewise_center")
        # datagen_arguments.pop("featurewise_std_normalization")
        masks_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_arguments)

        if save_to_dir:
            Path(save_to_dir).mkdir(exist_ok=True)
        train_images = images_datagen.flow(images, batch_size=batch_size, seed=seed, shuffle=True, save_prefix=f"{save_prefix}_image", save_to_dir=save_to_dir)
        train_masks = masks_datagen.flow(masks, batch_size=batch_size, seed=seed, shuffle=True, save_prefix=f"{save_prefix}_mask", save_to_dir=save_to_dir)

        return train_images, train_masks
    else:
        # images = images * 1./255.
        # masks = masks * 1./255.
        return images, masks

########
########

train_images, train_masks = load_images_and_masks("dataset/train/", target_size=(height, width), augment=True, batch_size=batch_size, save_to_dir=None)
validation_images, validation_masks = load_images_and_masks("dataset/validation/", target_size=(height, width))

# for i, (images, masks) in enumerate(zip(train_images, train_masks)):
#     print(images.shape)
#     if i + 1 == steps_per_epoch:
#         break
# assert 1 == 2

def get_generator(train_images, train_masks):
    for images, masks in zip(train_images, train_masks):
        yield (images, masks)

train_data = get_generator(train_images, train_masks)

########
########

def dice_coef(y_true, y_pred, smooth=1.):
    intersection = keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = keras.backend.sum(y_true, axis=[1,2,3]) + keras.backend.sum(y_pred, axis=[1,2,3])
    return keras.backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

########
########

data_augmentation = keras.Sequential(
    [
        # layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical", seed=seed),
        # layers.experimental.preprocessing.RandomRotation(0.2, seed=seed),
        # layers.experimental.preprocessing.RandomContrast(0.1, seed=seed),
        layers.experimental.preprocessing.Normalization(),
        # layers.experimental.preprocessing.Rescaling(1./255.),
    ]
)

########
########

def make_model(input_shape, name="AgNOR"):
    inputs = keras.Input(shape=input_shape)

    augmented = data_augmentation(inputs)

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(augmented)
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

model = make_model(input_shape=input_shape, name=model_name)
model.summary()

########
########

checkpoint_directory = os.path.join("checkpoints", f"{time.strftime('%Y%m%d%H%M%S')}")
os.makedirs(checkpoint_directory, exist_ok=True)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.25, patience=10, verbose=1,  mode="auto", cooldown=1),
    keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_directory, model_name + "_e{epoch:02d}_l{loss:.2f}_vl{val_loss:.2f}.h5"), monitor="val_dice_coef", save_best_only=False),
    # keras.callbacks.TensorBoard(log_dir=os.path.join(checkpoint_directory, "logs"), histogram_freq=1, update_freq="batch", write_images=False)
]

########
########

train_config = {
    "model_name": model.name,
    "seed": seed,
    "epochs": epochs,
    "batch_size": batch_size,
    "steps_per_epoch": steps_per_epoch,
    "effective_batches": effective_batches,
    "effective_images_per_epoch": effective_images,
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
print(f"  - Steps per epoch: {steps_per_epoch}")
print(f"  - Effective batches: {effective_batches}")
print(f"  - Effective images (per epoch): {effective_images}")
print(f"  - Input shape: {input_shape}")
print(f"  - Learning rate: {model.optimizer.get_config()['learning_rate']}")
print(f"  - Checkpoints saved at: {checkpoint_directory}\n")

keras.backend.clear_session()
history = model.fit(
    train_data,
    # train_masks,
    # batch_size=batch_size,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(validation_images, validation_masks),
    validation_batch_size=batch_size,
    validation_steps=len(validation_images),
    # initial_epoch=22,
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
test_images, test_masks = load_images_and_masks("dataset/test/", target_size=(height, width))
loss, dice = model.evaluate(test_images, test_masks, batch_size=1)
print("Loss: %.2f" % loss)
print("Dice: %.2f" % dice)

########
########

print("\nTesting model")

test_images_path = "dataset/test/images/"
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
