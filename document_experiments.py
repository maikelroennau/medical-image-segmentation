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
    "loss_function": None,
    "initial_learning_rate": None,
    "train_dataset": None,
    "train_samples": None,
    "validation_samples": None,
    "test_samples": None,
    "duration": None,
    "train_loss": None,
    "train_f1-score": None,
    "train_iou_score": None,
    "val_loss": None,
    "val_f1-score": None,
    "val_iou_score": None,
    "test_loss": None,
    "test_f1-score": None,
    "test_iou_score": None,
    "description": None
}

model_metric_keys = [
    "train_loss",
    "train_f1-score",
    "train_iou_score",
    "val_loss",
    "val_f1-score",
    "val_iou_score",
    "test_loss",
    "test_f1-score",
    "test_iou_score"
]

rename_keys = {
    "loss": "train_loss",
    "f1-score": "train_f1-score",
    "iou_score": "train_iou_score"
}


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
    pbar = tqdm(files)
    for json_file in pbar:
        pbar.set_description(json_file)
        try:
            with open(json_file, "r") as experiment:
                experiment = json.load(experiment)

                if "train_metrics" in experiment.keys():
                    for key in experiment["train_metrics"].keys():
                        experiment[key] = experiment["train_metrics"][key]

                for key, value in rename_keys.items():
                    if key in experiment.keys():
                        experiment[value] = experiment[key]

                if "test_metrics" in experiment.keys():
                    for key in experiment["test_metrics"].keys():
                        experiment[key] = experiment["test_metrics"][key]

                if "train_loss" in experiment.keys() or "test_loss" in experiment.keys():
                    for key in model_metric_keys:
                        if key not in experiment.keys():
                            continue
                        if "loss" in key:
                            if key == "train_loss" or key == "val_loss":
                                experiment[key] = np.min(experiment[key])
                            elif key == "test_loss":
                                test_loss = np.array(experiment[key])
                                test_loss = test_loss[test_loss > 0]
                                experiment[key] = np.min(test_loss)
                        else:
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
