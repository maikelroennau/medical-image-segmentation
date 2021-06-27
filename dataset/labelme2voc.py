import argparse
import glob
import os.path as osp
from pathlib import Path

import cv2
import imgviz
import labelme
import numpy as np
from tqdm import tqdm


def convert_labels(input_dir, output_dir="voc", labels=None, filter_labels=None, color=False, noviz=False):
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

    for resolution, factor in zip(resolutions, factors):
        res_name = "x".join([str(x) for x in resolution])
        current_general_path = Path(output_dir).joinpath(res_name)
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


        for filename in tqdm(glob.glob(osp.join(input_dir, "*.json")), desc=res_name):
            label_file = labelme.LabelFile(filename=filename)
            if filter_labels:
                label_file.shapes = [shape for shape in label_file.shapes if shape["label"] not in filter_labels]

            base = osp.splitext(osp.basename(filename))[0]
            out_img_file = str(images_dir.joinpath(base + ".png"))
            out_png_file = str(segmentation_dir.joinpath(base + ".png"))

            if not noviz:
                out_viz_file = str(viz_dir.joinpath(base + ".png"))

            # Save image to dir
            img = labelme.utils.img_data_to_arr(label_file.imageData)
            img = cv2.resize(img, dsize=resolution[::-1], interpolation=cv2.INTER_NEAREST)
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
        help="List of labels to keep, separated by comma. Example: `-f nucleus,nor`.",
        default=None,
        type=str)

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

    convert_labels(args.input_dir, args.output_dir, args.labels, args.filter_labels, args.color, args.noviz)


if __name__ == "__main__":
    main()
