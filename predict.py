import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

########
########

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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

def update_model(model, input_shape):
    model_weights = model.get_weights()
    model_json = json.loads(model.to_json())

    model_json["config"]["layers"][0]["config"]["batch_input_shape"] = [None, *input_shape]
    # model_json["config"]["layers"][1]["config"]["layers"][0]["config"]["batch_input_shape"] = [None, *input_shape]
    model_json["config"]["layers"][0]["config"]["batch_input_shape"] = [None, *input_shape]

    updated_model = keras.models.model_from_json(json.dumps(model_json))
    updated_model.set_weights(model_weights)
    return updated_model

########
########

def predict(model, images_path="dataset/test/images/"):
    loaded_model = keras.models.load_model(model, custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
    # loaded_model = update_model(loaded_model, (1920, 2560, 3))
    input_shape = loaded_model.input_shape[1:]
    height, width, channels = input_shape

    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    images = [image_path for image_path in Path(images_path).rglob("*.*") if image_path.suffix.lower() in supported_types and not image_path.stem.endswith("_prediction")]
    images_tensor = np.empty((1, height, width, channels))

    for i, image_path in enumerate(images):
        image = cv2.imread(os.path.join(images_path, image_path.name), cv2.IMREAD_COLOR)
        original_shape = image.shape[:2][::-1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        images_tensor[0, :, :, :] = image

        prediction = loaded_model.predict(images_tensor, batch_size=1, verbose=1)
        prediction = cv2.resize(prediction[0], original_shape)
        if len(prediction.shape) > 2:
            prediction[:, :, 0][prediction[:, :, 0] < 0.5] = 0
            prediction[:, :, 0][prediction[:, :, 0] >= 0.5] = 255

            prediction[:, :, 1][prediction[:, :, 1] == 0.] = 255
            prediction[:, :, 1][prediction[:, :, 1] < 255.] = 0

            output = np.zeros(prediction.shape[:2] + (3,))
            output[:, :, 0:2] = prediction.astype(np.uint8)

            cv2.imwrite(os.path.join(images_path, f"{image_path.stem}_{loaded_model.name}_{i}_prediction.png"), cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2RGB))
        else:
            prediction[prediction < 0.5] = 0
            prediction[prediction >= 0.5] = 255
            cv2.imwrite(os.path.join(images_path, f"{image_path.stem}_{loaded_model.name}_prediction.png"), prediction)
        keras.backend.clear_session()


if __name__ == "__main__":
    if len(sys.argv) > 2:
        predict(sys.argv[1], sys.argv[2])
    else:
        predict(sys.argv[1])
