"""Visualize U-Net segmentation predictions on training and test images."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from unet import DATA_SOURCE, MODEL_PATH, NUM_CLASSES, Unet, build_dataloaders, dsc


def load_model(model_path, device):
    """Load a U-Net checkpoint saved by unet.py."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    model = Unet(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_classes


@torch.no_grad()
def predict(model, images, device):
    """Return categorical predictions for a CPU image batch."""
    logits = model(images.to(device, non_blocking=True))
    return logits.argmax(dim=1).cpu(), logits.cpu()


def plot_results(
    train_images,
    train_targets,
    train_predictions,
    test_images,
    test_targets,
    test_predictions,
    num_classes,
    output_path=None,
):
    """Plot input, target, and predicted masks for both dataset splits."""
    examples_per_split = train_images.size(0)
    figure, axes = plt.subplots(
        examples_per_split * 2,
        3,
        figsize=(9, 3.2 * examples_per_split * 2),
        squeeze=False,
    )
    panels = (
        ("Train", train_images, train_targets, train_predictions),
        ("Test", test_images, test_targets, test_predictions),
    )
    for split_index, (split_name, images, targets, predictions) in enumerate(panels):
        for example_index in range(examples_per_split):
            row = split_index * examples_per_split + example_index
            axes[row, 0].imshow(images[example_index].squeeze(), cmap="gray")
            axes[row, 0].set_title(f"{split_name} image {example_index + 1}")
            axes[row, 1].imshow(
                targets[example_index], cmap="tab10", vmin=0, vmax=num_classes - 1
            )
            axes[row, 1].set_title("Ground truth")
            axes[row, 2].imshow(
                predictions[example_index],
                cmap="tab10",
                vmin=0,
                vmax=num_classes - 1,
            )
            axes[row, 2].set_title("Prediction")
            for axis in axes[row]:
                axis.axis("off")
    figure.suptitle("U-Net segmentation results")
    figure.tight_layout()
    if output_path is None:
        plt.show()
    else:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")
    plt.close(figure)


def main(
    model_path=MODEL_PATH,
    samples=4,
    num_workers=0,
    output_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, num_classes = load_model(model_path, device)
    train_loader, _, test_loader = build_dataloaders(
        data_source=DATA_SOURCE,
        batch_size=samples,
        num_workers=num_workers,
    )
    train_images, train_targets = next(iter(train_loader))
    test_images, test_targets = next(iter(test_loader))
    train_predictions, train_logits = predict(model, train_images, device)
    test_predictions, test_logits = predict(model, test_images, device)
    print(f"Train DSC: {dsc(train_logits, train_targets, num_classes):.4f}")
    print(f"Test DSC:  {dsc(test_logits, test_targets, num_classes):.4f}")
    plot_results(
        train_images,
        train_targets,
        train_predictions,
        test_images,
        test_targets,
        test_predictions,
        num_classes,
        output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="path to the checkpoint produced by unet.py",
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="number of data-loader workers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="save the figure instead of opening an interactive window",
    )
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        samples=args.samples,
        num_workers=args.num_workers,
        output_path=args.output,
    )
