from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep


def post_process(image):
    image[image > 0] = 255
    nuclei_prediction = image[:, :, 1].astype(np.uint8)
    nors_prediction = image[:, :, 2].astype(np.uint8)

    # Find segmentation contours
    nuclei_polygons, _ = cv2.findContours(nuclei_prediction.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nuclei_polygons = filter_contours(nuclei_polygons)
    nuclei_polygons = smooth_contours(nuclei_polygons, 40)

    nors_polygons, _ = cv2.findContours(nors_prediction.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nors_polygons = smooth_contours(nors_polygons, 20)
    
    # Filter out nuclei without NORs
    filtered_nuclei = []
    for nucleus in nuclei_polygons:
        keep_nucleus = False
        for nor in nors_polygons:
            for nor_point in nor:
                if cv2.pointPolygonTest(nucleus, tuple(nor_point[0]), True) >= 0:
                    keep_nucleus = True
        if keep_nucleus:
            filtered_nuclei.append(nucleus)

    # Filter out non-convex nuclei
    # double_filtered_nuclei = []
    # for nucleus in filtered_nuclei:
    #     if cv2.contourArea(cv2.convexHull(nucleus)) - cv2.contourArea(nucleus) < 1000:
    #         double_filtered_nuclei.append(nucleus)

    # Filter out NORs outside nuclei
    filtered_nors = []
    for nor in nors_polygons:
        keep_nor = False
        for nucleus in filtered_nuclei:
            for nor_point in nor:
                if cv2.pointPolygonTest(nucleus, tuple(nor_point[0]), True) >= 0:
                    keep_nor = True
        if keep_nor:
            filtered_nors.append(nor)

    nucleus = np.zeros(image.shape[:2], dtype=np.uint8)
    nor = np.zeros(image.shape[:2], dtype=np.uint8)
    background = np.ones(image.shape[:2], dtype=np.uint8)

    cv2.drawContours(nucleus, contours=filtered_nuclei, contourIdx=-1, color=1, thickness=cv2.FILLED)
    cv2.drawContours(nor, contours=filtered_nors, contourIdx=-1, color=1, thickness=cv2.FILLED)

    nucleus = np.where(nor, 0, nucleus)
    background = np.where(np.logical_and(nucleus == 0, nor == 0), 1, 0)

    post_processed_image = np.stack([background, nucleus, nor], axis=2)
    post_processed_image[post_processed_image > 0] = 127
    
    measurements = {}
    for i, nucleus in enumerate(filtered_nuclei):
        i = str(i)
        measurements[i] = {
                "nucleus_area": cv2.contourArea(nucleus),
                "n_nors": 0,
                "nors_area": {}
            }
        for j, nor in enumerate(filtered_nors):
            for nor_point in nor:
                if cv2.pointPolygonTest(nucleus, tuple(nor_point[0]), True) >= 0:
                    measurements[i]["n_nors"] = measurements[i]["n_nors"] + 1
                    measurements[i]["nors_area"][str(j)] = cv2.contourArea(nor)
                    break

    return post_processed_image.astype(np.uint8), measurements


def plot_metrics(data, title="", output=".", figsize=(15, 15)):
    output_path = Path(output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(data)
    df.index = range(1, len(df.index) + 1)

    image = df.plot(grid=True, figsize=figsize)
    image.set(xlabel="Epoch", title=title)
    image.set_ylim(ymin=0)

    for column in df.columns:
        if "loss" in column:
            text = f"e{np.argmin(list(df[column])) + 1}"
            value = (np.argmin(list(df[column])) + 1, df[column].min())
        else:
            text = f"e{np.argmax(list(df[column])) + 1}"
            value = (np.argmax(list(df[column])) + 1, df[column].max())

        if column != "lr":
            image.annotate(text, value, arrowprops=dict(facecolor='black', shrink=0.05))

    image = image.get_figure()
    image.savefig(str(output_path))


def filter_contours(contours, min_area=5000, max_area=40000):
    filtered_contours = []
    for contour in contours:
        try:
            contour_area = cv2.contourArea(contour)
            if min_area <= contour_area and contour_area <= max_area:
                filtered_contours.append(contour)
        except:
            pass
    return filtered_contours


def smooth_contours(contours, points=30):
    smoothened_contours = []
    for contour in contours:
        try:
            x, y = contour.T

            # Convert from numpy arrays to normal arrays
            x = x.tolist()[0]
            y = y.tolist()[0]

            # Find the B-spline representation of an N-dimensional curve.
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
            tck, u = splprep([x, y], u=None, s=1.0, per=1, k=1)

            # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
            u_new = np.linspace(u.min(), u.max(), points)

            # Given the knots and coefficients of a B-spline representation, evaluate the value of the smoothing polynomial and its derivatives.
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
            x_new, y_new = splev(u_new, tck, der=0)

            # Convert it back to Numpy format for OpenCV to be able to display it
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
            smoothened_contours.append(np.asarray(res_array, dtype=np.int32))
        except:
            pass
    return smoothened_contours
