import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


experiment_data = {
    "directory": None,
    "model_name": None,
    "backbone": None,
    "decoder": None,
    "epochs": None,
    "batch_size": None,
    "steps_per_epoch": None,
    "input_shape": None,
    "loss_fuction": None,
    "initial_learning_rate": None,
    "train_dataset": None,
    "train_samples": None,
    "validation_samples": None,
    "test_samples": None,
    "duration": None,
    "loss": None,
    "min_loss_epoch": None,
    "val_loss": None,
    "min_val_loss_epoch": None,
    "val_f1-score": None,
    "max_val_f1-score_epoch": None,
    "val_iou_score": None,
    "max_val_iou_score_epoch": None,
    "description": None
}

model_metric_keys = [
    "loss",
    "f1-score",
    "iou_score",
    "val_loss",
    "val_f1-score",
    "val_iou_score",
]


def document(experiment_file, file_pattern="*train_config.json", output="."):
    experiment_file = Path(experiment_file)
    if experiment_file.is_file():
        files = [str(experiment_file)]
    elif experiment_file.is_dir():
        files = [str(experiment) for experiment in experiment_file.rglob(file_pattern)]
    else:
        print(f"The file or directory '{str(experiment_file)}' was not found.")
        return

    output = Path(output)

    files.sort()
    for json_file in tqdm(files):
        try:
            with open(json_file, "r") as experiment:
                experiment = json.load(experiment)

                if "train_metrics" in experiment.keys():
                    for key in experiment["train_metrics"].keys():
                        experiment[key] = experiment["train_metrics"][key]

                if "loss" in experiment.keys():
                    for key in model_metric_keys:
                        if "loss" in key:
                            experiment[f"min_{key}_epoch"] = np.argmin(experiment[key]) + 1
                            experiment[key] = np.min(experiment[key])
                        else:
                            experiment[f"max_{key}_epoch"] = np.argmax(experiment[key]) + 1
                            experiment[key] = np.max(experiment[key])

                experiment["directory"] = Path(experiment["directory"]).name
                experiment["input_shape"] = "x".join([str(value) for value in experiment["input_shape"]])

                for key in experiment_data.keys():
                    experiment_data[key] = None
                    if key in experiment.keys():
                        experiment_data[key] = [experiment[key]]

                df = pd.DataFrame.from_dict(experiment_data)

                if output.name.endswith(".csv"):
                    if output.is_file():
                        existing_data = pd.read_csv(str(output))
                        if int(experiment_data["directory"][0]) not in existing_data["directory"].values:
                            df.to_csv(str(output), mode="a", header=False, index=False)        
                    else:
                        df.to_csv(str(output), mode="w", header=True, index=False)
                else:
                    if not output.is_dir():
                        output.mkdir(exist_ok=True, parents=True)
                    df.to_csv(str(output.joinpath("experiments.csv")), mode="w", header=True, index=False)
        except Exception as e:
            print(f"Could not to process '{json_file}'. Reason: {e}")


def main():
    parser = argparse.ArgumentParser(description="Converts experiment JSON to CSV.")

    parser.add_argument(
        "-e",
        "--experiment-file",
        help="Path to the experiment JSON file. If path is a dir, will recursively search for experiment JSON files.",
        required=True,
        type=str)

    parser.add_argument(
            "-p",
            "--pattern",
            help="Patter of the JSON files to be searched. Only effective if '-e' is a dir.",
            default="*train_config.json",
            type=str)

    parser.add_argument(
            "-o",
            "--output",
            help="Path where to save the converted experiment data.",
            default="experiments.csv",
            type=str)

    args = parser.parse_args()
    document(args.experiment_file, args.pattern, args.output)


if __name__ == "__main__":
    main()