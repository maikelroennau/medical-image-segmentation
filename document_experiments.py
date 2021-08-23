import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


experiment_data = {
    "directory": None,
    "model_name": None,
    "decoder": None,
    "backbone": None,
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
    "val_iou_score": None,
    "max_val_iou_score_epoch": None,
    "val_dice_coef": None,
    "max_val_dice_coef_epoch": None,
    "description": None
}

model_metric_keys = [
    "loss",
    "iou_score",
    "dice_coef",
    "val_loss",
    "val_iou_score",
    "val_dice_coef"
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

    for json_file in tqdm(files):
        try:
            with open(json_file, "r") as experiment:
                experiment = json.load(experiment)

                for key in model_metric_keys:
                    if "loss" in key:
                        experiment[key] = np.min(experiment["train_metrics"][key])
                        experiment[f"min_{key}_epoch"] = np.argmin(experiment["train_metrics"][key])
                    else:
                        experiment[key] = np.max(experiment["train_metrics"][key])
                        experiment[f"max_{key}_epoch"] = np.argmax(experiment["train_metrics"][key])

                experiment["directory"] = Path(experiment["directory"]).name
                experiment["input_shape"] = "x".join([str(value) for value in experiment["input_shape"]])

                for key in experiment_data.keys():
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
            default=".",
            type=str)

    args = parser.parse_args()
    document(args.experiment_file, args.pattern, args.output)

if __name__ == "__main__":
    main()
