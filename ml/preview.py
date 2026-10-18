"""Create a source-labelled contact sheet for manual dataset review."""

import argparse
import csv
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageOps
from backend.config import CLASS_NAMES


def preview(data, source, output):
    with (data / "train.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    selected = []
    for label in CLASS_NAMES:
        groups = set()
        for row in records:
            if row["label"] == label and row["group"] not in groups:
                selected.append(row)
                groups.add(row["group"])
                if len(groups) == 2:
                    break
    width, height, columns = 220, 250, 4
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height + 40), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "PlantVillage RGB training samples - source labels, not independent diagnosis", fill="black")
    for index, record in enumerate(selected):
        x, y = (index % columns) * width, (index // columns) * height + 40
        with Image.open(source / record["path"]) as image:
            image = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (210, 190))
            canvas.paste(image, (x + 5, y))
        draw.text((x + 5, y + 195), "\n".join(textwrap.wrap(record["label"], 29)), fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--source", type=Path, default=Path("data/raw/plantvillage"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset_preview.jpg"))
    args = parser.parse_args()
    preview(args.data, args.source, args.output)


if __name__ == "__main__":
    main()
