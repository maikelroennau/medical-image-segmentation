import os
import time

import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, Conv2DTranspose

########
########

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

########
########

seed = 2149

epochs = 20
batch_size = 32
steps_per_epoch = 2000

image_batch_size = 16
augmentation_batch_size = 16

height = 480 # 240 480  960 1920
width = 640 # 320 640 1280 2560
input_shape = (height, width, 3)

########
########

def data_generator(batch_size=16, target_size=(1920, 2560)):
    datagen_arguments = dict(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        fill_mode="reflect",
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255.
    )

    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_arguments)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_arguments)

    seed = 2149
    augmented_dir = "dataset/augmentation/"

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

    for images, masks in zip(images, masks):
        yield (images, masks)

########
########

generator = data_generator(augmentation_batch_size, (height, width))

# for i, batch in enumerate(generator):
#     if i >= image_batch_size - 1:
#         break

########
########

# images_path = "dataset/augmentation/images/"
# masks_path = "dataset/augmentation/masks/"

# images = os.listdir(images_path)
# masks = os.listdir(masks_path)
# images.sort()
# masks.sort()

# limit = len(images)

# print(f"Total images: {limit}")
# print(f"Target shape: {(limit, height, width, 3)}")

# images_tensor = np.empty((limit, height, width, 3))
# masks_tensor = np.empty((limit, height, width, 1))

# for i, (image, mask) in enumerate(zip(images, masks)):
#     assert image.split("_")[-1] == mask.split("_")[-1], f"Image and maks do not correspond: {image}, {mask}"
#     img = cv2.imread(os.path.join(images_path, image))
#     images_tensor[i, :, :, :] = img

#     msk = cv2.imread(os.path.join(masks_path, mask), cv2.IMREAD_GRAYSCALE)
#     masks_tensor[i, :, :, 0] = msk
#     # if i + 1 == limit:
#         # break

# print(images_tensor.shape)
# print(masks_tensor.shape)

# def yield_data(x, y):
#     for xx, yy in zip(x, y):
#         yield (xx, yy)

########
########

def dice_coef(y_true, y_pred, smooth=1.):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

########
########

def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = keras.Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef, "binary_accuracy"])

    return model

model = make_model(input_shape=input_shape)
model.summary()

########
########

checkpoint_directory = os.path.join("checkpoints", f"{time.strftime('%Y%m%d%H%M%S')}")
os.makedirs(checkpoint_directory)

callbacks = [
    #keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1,  mode="auto", cooldown=1),
    keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_directory, "epoch_{epoch}.h5"), monitor="val_loss", save_best_only=False),
]

########
########

start = time.time()
print(f"Training start - {time.strftime('%x %X')}")
print(f"  - Epochs: {epochs}")
print(f"  - Batch size: {batch_size}")
print(f"  - Learning rate: {model.optimizer.get_config()['learning_rate']}\n")

keras.backend.clear_session()
history = model.fit(generator, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)#, validation_data=val_ds)
# history = model.fit(images_tensor, masks_tensor, batch_size=batch_size, epochs=20, steps_per_epoch=16, callbacks=callbacks)#, validation_data=val_ds)

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

test_images_tensor = np.empty((len(images), height, width, 3))

for i, image_path in enumerate(images):
    image = cv2.imread(os.path.join(test_images_path, image_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    test_images_tensor[i, :, :, :] = image

print(test_images_tensor.shape)

########
########

loaded_model = keras.models.load_model(f"{checkpoint_directory}/epoch_{epochs}.h5", custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
predictions = loaded_model.predict(test_images_tensor, verbose=1)

for i, prediction in enumerate(predictions):
    cv2.imwrite(os.path.join(test_images_path, f"{i}_prediction.jpg"), prediction * 255)
