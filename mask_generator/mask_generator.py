import glob
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL

import cv2
import numpy as np
import PySimpleGUI as sg
import tensorflow as tf
from tensorflow import keras

from utils import dice_coef, dice_coef_loss, filter_contours, smooth_contours


def save_annotation(nuclei_prediction, nors_prediction, annotation_directory, name, original_shape, image):
    width = original_shape[0]
    height = original_shape[1]

    nuclei_prediction = cv2.resize(nuclei_prediction, (width, height))
    nors_prediction = cv2.resize(nors_prediction, (width, height))

    annotation = {
        "version": "4.5.7",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(name),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    # Find segmentation contours
    nuclei_polygons, _ = cv2.findContours(nuclei_prediction.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nuclei_polygons = filter_contours(nuclei_polygons)
    nuclei_polygons = smooth_contours(nuclei_polygons)

    nors_polygons, _ = cv2.findContours(nors_prediction.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nors_polygons = smooth_contours(nors_polygons)

    # Filter out nuclei without nors
    filtered_nuclei = []
    for nucleus in nuclei_polygons:
        keep_nucleus = False
        for nor in nors_polygons:
            if cv2.pointPolygonTest(nucleus, tuple(nor[0][0]), True) >= 0:
                keep_nucleus = True
        if keep_nucleus:
            filtered_nuclei.append(nucleus)

    # Filter out NORs outside nucleis
    filtered_nors = []
    for nor in nors_polygons:
        keep_nor = False
        for nucleus in nuclei_polygons:
            if cv2.pointPolygonTest(nucleus, tuple(nor[0][0]), True) >= 0:
                keep_nor = True
        if keep_nor:
            filtered_nors.append(nor)

    for nucleus_points in filtered_nuclei:
        points = []
        for point in nucleus_points:
            points.append([int(value) for value in point[0]])

        shape = {
            "label": "nucleus",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        annotation["shapes"].append(shape)

    for nors_points in filtered_nors:
        points = []
        for point in nors_points:
            points.append([int(value) for value in point[0]])

        shape = {
            "label": "nor",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        annotation["shapes"].append(shape)

    # Write annotation file
    with open(os.path.join(annotation_directory, f'{os.path.splitext(os.path.basename(name))[0]}.json'), "w") as output_file:
        json.dump(annotation, output_file, indent=2)

    # Write a copy of the image to the annotation directory
    if Path(os.path.join(annotation_directory, os.path.basename(name))).is_file():
        filename = Path(os.path.join(annotation_directory, os.path.basename(name)))
        filename = filename.stem + f"_{np.random.randint(1, 1000)}" + filename.suffix
        shutil.copyfile(name, os.path.join(annotation_directory, filename))
    else:
        shutil.copyfile(name, os.path.join(annotation_directory, os.path.basename(name)))


def main():
    # Suppress Numpy warnings
    warnings.simplefilter("ignore")

    # Consturct UI
    sg.theme("DarkBlue")
    layout = [
        [
            sg.Text("Image Folder", text_color="white"),
            sg.In(size=(50, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [
            sg.Text("Status: waiting" + " " * 30, text_color="white", key="-STATUS-"),
        ]
    ]

    if "icon.ico" in glob.glob("icon.ico"):
        icon = os.path.join(".", "icon.ico")
    else:
        icon = os.path.join(sys._MEIPASS, "icon.ico")

    window = sg.Window("Mask Generator", layout, finalize=True, icon=icon)
    status = window["-STATUS-"]
    update_status = True

    # Prediction settings
    supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    # Load models
    if "epoch_20_nucleus.h5" in glob.glob("*.h5") and "epoch_50_nor.h5" in glob.glob("*.h5"):
        models_base_path = "."
    else:
        models_base_path = sys._MEIPASS

    nuclei_model = keras.models.load_model(os.path.join(models_base_path, "epoch_20_nucleus.h5"), custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
    nors_model = keras.models.load_model(os.path.join(models_base_path, "epoch_50_nor.h5"), custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})

    input_shape = nuclei_model.input_shape[1:]

    # Prepare tensor
    height, width, channels = input_shape
    image_tensor = np.empty((1, height, width, channels))

    # UI loop
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if values["-FOLDER-"] == "":
            continue

        # Folder name was filled in, make a list of files in the folder
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            images = [path for path in Path(folder).rglob("*.*") if path.suffix.lower() in supported_types]

            if len(images) == 0:
                status.update("Status: no images found!")
                continue

            annotation_directory = f"{time.strftime('%Y-%m-%d-%Hh%Mm')}-proposed-annotations"
            if not os.path.isdir(annotation_directory):
                os.mkdir(annotation_directory)

            status.update("Status: processing")
            event, values = window.read(timeout=0)

            # Load and process each image
            for i, image_path in enumerate(images):
                if not sg.OneLineProgressMeter("Progress", i + 1, len(images), "key", orientation="h"):
                    if not i + 1 == len(images):
                        status.update("Status: canceled by the user")
                        update_status = False
                        break

                image_path = str(image_path)
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                original_shape = image.shape[:2][::-1]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (width, height))
                image_tensor[0, :, :, :] = image

                # nuclei_prediction = nuclei_model.predict(image_tensor, batch_size=1, verbose=0)
                nuclei_prediction = nuclei_model.predict_on_batch(image_tensor)
                # nors_prediction = nors_model.predict(image_tensor, batch_size=1, verbose=0)
                nors_prediction = nors_model.predict_on_batch(image_tensor)
                save_annotation(nuclei_prediction[0], nors_prediction[0], annotation_directory, image_path, original_shape, image)
                keras.backend.clear_session()

        if update_status:
            status.update("Status: done!")
        else:
            update_status = True
    window.close()


if __name__ == "__main__":
    main()
