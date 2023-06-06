import argparse
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

from utils.data import SUPPORTED_IMAGE_TYPES
from utils.utils import color_classes


def process(masks: str, output: Optional[str] = None, colormap: Optional[str] = "agnor") -> None:
    files = [str(file) for file in Path(masks).glob("*") if file.suffix in SUPPORTED_IMAGE_TYPES]

    if len(files) == 0:
        raise FileNotFoundError(f"No files found in `{masks}`.")

    if output is None:
        output = Path(masks).joinpath("colored")
    else:
        output = Path(output)
    
    output.mkdir(exist_ok=True, parents=True)

    for file in tqdm(files):
        mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        colored = color_classes(mask, colormap=colormap)
        output_file = str(output.joinpath(Path(file).name))
        cv2.imwrite(output_file, cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))


def main() -> None:
    parser = argparse.ArgumentParser(description="Color segmentation masks.")

    parser.add_argument(
        "-m",
        "--masks",
        help="The directory containing masks to color",
        type=str)

    parser.add_argument(
        "-o",
        "--output",
        help="Path where to save colored masks. Defaults to a new directory created withing the input masks' diretory.",
        required=False,
        type=str)
    
    parser.add_argument(
        "-c",
        "--colormap",
        help="What color map to use. Either `agnor` (3 classes + background), or `papanicolaou` (7 classes + background). Defaults to `agnor`.",
        choices=["agnor", "papanicolaou"],
        default="agnor",
        type=str)

    args = parser.parse_args()
    process(masks=args.masks, output=args.output, colormap=args.colormap)


if __name__ == "__main__":
    main()
