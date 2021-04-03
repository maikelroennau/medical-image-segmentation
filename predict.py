import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

########
########

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

########
########

epochs = 20
batch_size = 256
steps_per_epoch = 256

image_batch_size = 256
augmentation_batch_size = 16

height = 240 # 240 480  960 1920
width = 320 # 320 640 1280 2560
input_shape = (height, width, 3)

learning_rate = 1e-5

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

def load_images(test_images_path):
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
    return test_images_tensor, original_shape, images

########
########

def predict(model, test_images_path="dataset/test/"):
    test_images_tensor, original_shape, images = load_images(test_images_path)

    loaded_model = keras.models.load_model(model, custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
    predictions = loaded_model.predict(test_images_tensor, verbose=1)

    for i, prediction in enumerate(predictions):
        name = os.path.basename(images[i]).split(".")[0]
        prediction = cv2.resize(prediction, original_shape)
        cv2.imwrite(os.path.join(test_images_path, f"{name}_prediction.jpg"), prediction * 255)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        predict(sys.argv[1], sys.argv[2])
    else:
        predict(sys.argv[1])
