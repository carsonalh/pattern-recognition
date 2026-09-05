"""Plot epoch-20 validation accuracy for the ResNet-18 hyperparameter grid."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_PATTERN = re.compile(r"resnet18_b(?P<batch>\d+)_lr(?P<learning_rate>[\d.]+)\.out$")
ACCURACY_PATTERN = re.compile(
    r"Epoch\s+20/\d+.*validation acc:\s*(?P<accuracy>\d+(?:\.\d+)?)"
)


def read_epoch_20_accuracy(path: Path) -> float:
    """Return the epoch-20 validation accuracy from a result log."""
    for line in path.read_text().splitlines():
        match = ACCURACY_PATTERN.search(line)
        if match:
            return float(match.group("accuracy"))
    raise ValueError(f"No epoch-20 validation accuracy found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=Path("experiments/resnet18-grid-rtx-5070"),
        help="directory containing ResNet-18 result logs",
    )
    parser.add_argument("--output", type=Path, help="save the plot to this file")
    args = parser.parse_args()

    values = {}
    for path in args.results_dir.glob("resnet18_b*_lr*.out"):
        match = RESULT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        batch_size = int(match.group("batch"))
        learning_rate = float(match.group("learning_rate"))
        values[(batch_size, learning_rate)] = read_epoch_20_accuracy(path)

    if not values:
        raise SystemExit(f"No matching result files found in {args.results_dir}")

    batch_sizes = sorted({batch for batch, _ in values})
    learning_rates = sorted({rate for _, rate in values})
    accuracy = np.array([
        [values[(batch, rate)] for rate in learning_rates]
        for batch in batch_sizes
    ])

    figure, axis = plt.subplots()
    image = axis.imshow(
        accuracy,
        aspect="auto",
        cmap="turbo",
        vmin=accuracy.min(),
        vmax=accuracy.max(),
    )
    axis.set_xticks(range(len(learning_rates)), labels=[str(rate) for rate in learning_rates])
    axis.set_yticks(range(len(batch_sizes)), labels=[str(batch) for batch in batch_sizes])
    axis.set_xlabel("Learning rate")
    axis.set_ylabel("Batch size")
    axis.set_title("ResNet-18 validation accuracy at epoch 20")
    for row in range(accuracy.shape[0]):
        for column in range(accuracy.shape[1]):
            axis.text(
                column,
                row,
                f"{accuracy[row, column]:.3f}",
                ha="center",
                va="center",
                color="black" if accuracy[row, column] > accuracy.mean() else "white",
            )
    figure.colorbar(image, ax=axis, label="Validation accuracy")
    figure.tight_layout()

    if args.output:
        figure.savefig(args.output, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
