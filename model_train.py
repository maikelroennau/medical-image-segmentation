import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

########
########

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

########
########

model_name = "AgNOR-Nucleus"
seed = 2149

epochs = 5
batch_size = 16
steps_per_epoch = 128
effective_batches = steps_per_epoch * epochs
effective_images = batch_size * steps_per_epoch

height = 960 # 240 480 960 1920
width = 1280 # 320 640 1280 2560
input_shape = (height, width, 3)

learning_rate = 1e-5

########
########

def data_loader(batch_size=16, target_size=(1920, 2560), augmented_dir="dataset/augmentation/", seed=2149):
    datagen_arguments = dict(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rotation_range=50,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        zoom_range=0.5,
        fill_mode="reflect",
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255.
    )

    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_arguments)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_arguments)

    images = image_datagen.flow_from_directory(
        directory="dataset/train/",
        target_size=target_size,
        classes=["images"],
        class_mode=None,
        color_mode="rgb",
        batch_size=batch_size,
        # save_to_dir=f"{augmented_dir}/images",
        save_prefix="image",
        seed=seed
    )

    masks = mask_datagen.flow_from_directory(
        directory="dataset/train/",
        target_size=target_size,
        classes=["masks"],
        class_mode=None,
        color_mode="rgb",
        batch_size=batch_size,
        # save_to_dir=f"{augmented_dir}/masks",
        save_prefix="image",
        seed=seed
    )

    return images, masks

def data_generator(images, masks):
    for images, masks in zip(images, masks):
        yield (images, masks)

########
########

images, masks = data_loader(batch_size, (height, width))
generator = data_generator(images, masks)

# for i, batch in enumerate(generator):
#     print(f"Iteration {i}")
#     if i + 1 == effective_batches:
#         break
# assert 1 == 2

########
########

# images_path = "dataset/train/images/"
# masks_path = "dataset/train/masks/"

# images = os.listdir(images_path)
# masks = os.listdir(masks_path)
# images.sort()
# masks.sort()

# number_of_images = len(images)

# print(f"Total images: {number_of_images}")
# print(f"Target shape: {(number_of_images,) + input_shape}")

# images_tensor = np.empty((number_of_images,) + input_shape)
# masks_tensor = np.empty((number_of_images,) + input_shape)

# for i, (image, mask) in enumerate(zip(images, masks)):
#     assert image.split(".")[0] == mask.split(".")[0], f"Image and maks do not correspond: {image}, {mask}"
#     img = cv2.imread(os.path.join(images_path, image))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (width, height))
#     images_tensor[i, :, :, :] = img

#     msk = cv2.imread(os.path.join(masks_path, mask), cv2.IMREAD_GRAYSCALE)
#     msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
#     msk = cv2.resize(msk, (width, height))
#     masks_tensor[i, :, :, :] = msk

# print(images_tensor.shape)
# print(masks_tensor.shape)

########
########

def dice_coef(y_true, y_pred, smooth=1.):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

########
########

def make_model(input_shape, name="AgNOR"):
    inputs = keras.Input(shape=input_shape)

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
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

    conv10 = Conv2D(1, (1, 1), activation="sigmoid")(conv9)

    model = keras.Model(inputs=[inputs], outputs=[conv10], name=model_name)

    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef, "binary_accuracy"])

    return model

model = make_model(input_shape=input_shape, name=model_name)
model.summary()

########
########

checkpoint_directory = os.path.join("checkpoints", f"{time.strftime('%Y%m%d%H%M%S')}")
os.makedirs(checkpoint_directory)

with open(os.path.join(checkpoint_directory, "hyperparameters.txt"), "w") as hyperparameters:
    hyperparameters.write(f"Model name: {model.name}\n")
    hyperparameters.write(f"Seed: {seed}\n")
    hyperparameters.write(f"Epochs: {epochs}\n")
    hyperparameters.write(f"Batch size: {batch_size}\n")
    hyperparameters.write(f"Steps per epoch: {steps_per_epoch}\n")
    hyperparameters.write(f"Effective batches: {effective_batches}\n")
    hyperparameters.write(f"Effective images: {effective_images}\n")
    hyperparameters.write(f"Input shape: {input_shape}\n")
    hyperparameters.write(f"Learning rate: {model.optimizer.get_config()['learning_rate']}\n")

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10, verbose=1,  mode="auto", cooldown=1),
    keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_directory, "epoch_{epoch}.h5"), monitor="loss", save_best_only=False),
]

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
print(f"  - Effective images: {effective_images}")
print(f"  - Input shape: {input_shape}")
print(f"  - Learning rate: {model.optimizer.get_config()['learning_rate']}")
print(f"  - Checkpoints saved at: {checkpoint_directory}\n")

keras.backend.clear_session()
history = model.fit(generator, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)#, validation_data=val_ds)
# history = model.fit(images_tensor, masks_tensor, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)#, validation_data=val_ds)

end = time.time()
print(f"\nTraining end - {time.strftime('%x %X')}")
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Duration: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
print(f"  - Learning rate: {model.optimizer.get_config()['learning_rate']}")

########
########

test_images_path = "dataset/test/"

images = os.listdir(test_images_path)
images = [image for image in images if not image.endswith("_prediction.jpg")]

test_images_tensor = np.empty((len(images), height, width, 3))
original_shape = None

for i, image_path in enumerate(images):
    image = cv2.imread(os.path.join(test_images_path, image_path), cv2.IMREAD_COLOR)
    original_shape = image.shape[:2][::-1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    test_images_tensor[i, :, :, :] = image

print(test_images_tensor.shape)

########
########

keras.backend.clear_session()
loaded_model = keras.models.load_model(f"{checkpoint_directory}/epoch_{epochs}.h5", custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
predictions = loaded_model.predict(test_images_tensor, verbose=1)

for i, prediction in enumerate(predictions):
    name = os.path.basename(images[i]).split(".")[0]
    prediction = cv2.resize(prediction, original_shape)
    cv2.imwrite(os.path.join(test_images_path, f"{name}_prediction.jpg"), prediction * 255)
