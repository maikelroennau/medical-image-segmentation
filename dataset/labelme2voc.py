import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import imgviz
import labelme
import numpy as np
import tifffile
from tqdm import tqdm


def convert_labels(input_dir, output_dir="voc", labels=None, filter_labels=None, multiple_resolutions=False, tif=False, color=False, noviz=False):
    resolutions = ((1920, 2560), (960, 1280), (480, 640), (240, 320))
    factors = (1, 0.5, 0.25, 0.125)
    class_names = []
    class_name_to_id = {}

    if filter_labels:
        lines = [line.strip() for line in open(labels).readlines() if line.strip() not in filter_labels]
    else:
        lines = open(labels).readlines()

    for i, line in enumerate(lines):
        class_id = i - 1  # starts with -1
        class_name = line.strip()

        class_name_to_id[class_name] = class_id

        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"

        class_names.append(class_name)
    class_names = tuple(class_names)

    print("class_names:", class_names)

    if not multiple_resolutions:
        resolutions = resolutions[:1]
        factors = factors[:1]

    for resolution, factor in zip(resolutions, factors):
        if multiple_resolutions:
            res_name = "x".join([str(x) for x in resolution])
            current_general_path = Path(output_dir).joinpath(res_name)
        else:
            res_name = ""
            current_general_path = Path(output_dir)

        current_general_path.mkdir(exist_ok=True, parents=True)

        classes_file = str(current_general_path.joinpath("class_names.txt"))
        with open(classes_file, "w") as f:
            f.writelines("\n".join(class_names))

        images_dir = current_general_path.joinpath("Images")
        images_dir.mkdir(exist_ok=True, parents=True)

        segmentation_dir = current_general_path.joinpath("SegmentationClassPNG")
        segmentation_dir.mkdir(exist_ok=True, parents=True)

        if not noviz:
            viz_dir = current_general_path.joinpath("SegmentationClassVisualization")
            viz_dir.mkdir(exist_ok=True, parents=True)

        for filename in tqdm(glob.glob(os.path.join(input_dir, "*.json")), desc=res_name):
            try:
                with open(filename, "r") as annotation_file:
                    annotation_file = json.load(annotation_file)
                    if "invalidated" in annotation_file.keys():
                        if annotation_file["invalidated"]:
                            continue

                label_file = labelme.LabelFile(filename=filename)
                if filter_labels:
                    label_file.shapes = [shape for shape in label_file.shapes if shape["label"] not in filter_labels]

                base = os.path.splitext(os.path.basename(filename))[0]
                image_type = ".tif" if tif else ".png"
                out_img_file = str(images_dir.joinpath(base + image_type))
                out_png_file = str(segmentation_dir.joinpath(base + ".png"))

                if not noviz:
                    out_viz_file = str(viz_dir.joinpath(base + ".png"))

                # Save image to dir
                img = labelme.utils.img_data_to_arr(label_file.imageData)
                img = cv2.resize(img, dsize=resolution[::-1], interpolation=cv2.INTER_NEAREST)
                if tif:
                    tifffile.imwrite(out_img_file, img, photometric="rgb")
                else:
                    cv2.imwrite(out_img_file, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                # Rescale points accordingly to resolution factor
                for i in range(len(label_file.shapes)):
                    label_file.shapes[i]["points"] = list(np.asarray(label_file.shapes[i]["points"]) * factor)

                lbl, _ = labelme.utils.shapes_to_label(
                    img_shape=resolution + (3,),
                    shapes=label_file.shapes,
                    label_name_to_value=class_name_to_id,
                )

                if not color:
                    cv2.imwrite(out_png_file, lbl)
                else:
                    labelme.utils.lblsave(out_png_file, lbl)

                if not noviz:
                    viz = imgviz.label2rgb(
                        label=lbl,
                        img=imgviz.rgb2gray(img),
                        font_size=15,
                        label_names=class_names,
                        loc="rb",
                    )
                    imgviz.io.imsave(out_viz_file, viz)
            except Exception as e:
                print(e)


def main():
    parser = argparse.ArgumentParser(description="Converts labelme files into multiple resolution segmentation masks in the VOC format.")

    parser.add_argument(
        "-i",
        "--input-dir",
        help="Path to images and labels file.",
        required=True,
        type=str)

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory.",
        default="voc",
        type=str)

    parser.add_argument(
        "-l",
        "--labels",
        help="Path to labels file containing the classes names. If not specified, will search for a .txt file named `labels.txt` in the input dir.",
        default=None,
        type=str)

    parser.add_argument(
        "-f",
        "--filter-labels",
        help="List of labels to remove, separated by comma. Example: `-f nucleus,nor`.",
        default=None,
        type=str)

    parser.add_argument(
        "-m",
        "--multiple-resolutions",
        help="Generate images and masks in multiple resolutions.",
        default=False,
        action="store_true")

    parser.add_argument(
        "-t",
        "--tif",
        help="Save images using `.tif` format.",
        default=False,
        action="store_true")

    parser.add_argument(
        "-c",
        "--color",
        help="Generate RGB segmentation masks.",
        default=False,
        action="store_true")

    parser.add_argument(
        "--noviz",
        help="Do not generate overlay visualization.",
        default=False,
        action="store_true")

    args = parser.parse_args()

    if not args.labels:
        args.labels = str(Path(args.input_dir).joinpath("labels.txt"))
    if args.filter_labels:
        args.filter_labels = args.filter_labels.split(",")

    convert_labels(
        args.input_dir,
        args.output_dir,
        args.labels,
        args.filter_labels,
        args.multiple_resolutions,
        args.tif,
        args.color,
        args.noviz
    )


if __name__ == "__main__":
    main()
