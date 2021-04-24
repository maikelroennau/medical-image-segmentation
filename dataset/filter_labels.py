import json
import sys
from pathlib import Path

from tqdm import tqdm


def main(path, label):
    target_dir = Path(path)
    json_files = [json_file for json_file in target_dir.glob("*.json")]

    if len(json_files) == 0:
        print("No files were found")

    for json_file in tqdm(json_files):
        annotation_data = None
        with open(str(json_file), "r") as annotation_file:
            annotation_data = json.load(annotation_file)

        new_shapes = []
        for shape in annotation_data["shapes"]:
            if shape["label"] == label:
                new_shapes.append(shape)

        annotation_data["shapes"] = new_shapes
        with open(str(json_file), "w") as annotation_file:
            json.dump(annotation_data, annotation_file)

    with open(str(target_dir.joinpath("labels.txt")), "w") as labels:
        lines = ["__ignore__\n", "_background_\n", f"{label}"]
        labels.writelines(lines)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the path to the JSON files and the class name")
