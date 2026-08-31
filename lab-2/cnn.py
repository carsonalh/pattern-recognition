"""Train a convolutional neural network on the eigenfaces LFW dataset."""

from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset

from eigenfaces import load_dataset


def build_model(image_height, image_width, n_classes):
    """Construct the CNN with sequentially chained PyTorch layers."""
    feature_extractor = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
    )

    with torch.no_grad():
        flattened_features = feature_extractor(
            torch.zeros(1, 1, image_height, image_width)
        ).shape[1]

    return nn.Sequential(
        feature_extractor,
        nn.Linear(flattened_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, n_classes),
    )


def visualize_predictions(model, test_dataset, target_names, device, sample_size=12):
    """Browse predictions made only on examples from the test dataset."""
    generator = torch.Generator().manual_seed(42)
    sample_indices = torch.randperm(len(test_dataset), generator=generator)
    model.eval()

    for start in range(0, len(sample_indices), sample_size):
        page_indices = sample_indices[start : start + sample_size]
        images = torch.stack([test_dataset[index][0] for index in page_indices])
        labels = torch.tensor([test_dataset[index][1] for index in page_indices])
        with torch.no_grad():
            predictions = model(images.to(device)).argmax(dim=1).cpu()

        columns = 4
        page_size = len(page_indices)
        rows = (page_size + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(12, 3.5 * rows))
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
        for axis, image, prediction, label in zip(axes, images, predictions, labels):
            is_correct = prediction == label
            result_color = "green" if is_correct else "red"
            axis.imshow(image.squeeze(0), cmap="gray")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color(result_color)
                spine.set_linewidth(3)
            axis.text(0.5, -0.06, f"Prediction: {target_names[prediction.item()]}",
                      transform=axis.transAxes, ha="center", color=result_color)
            if not is_correct:
                axis.text(0.5, -0.12, f"Correct: {target_names[label.item()]}",
                          transform=axis.transAxes, ha="center", color="black")

        for axis in axes[page_size:]:
            axis.set_visible(False)
        figure.suptitle("Press any key for the next test-set images")
        figure.tight_layout()
        figure.canvas.draw()
        figure.canvas.flush_events()

        advanced = False

        def advance(event):
            nonlocal advanced
            key = str(event.key).lower()
            if "alt" in key or "f4" in key:
                return
            advanced = True
            plt.close(figure)

        figure.canvas.mpl_connect("key_press_event", advance)
        while plt.fignum_exists(figure.number):
            plt.pause(0.1)
        if not advanced:
            return


def main(epochs=20, batch_size=64):
    model_path = next(
        (Path(argument) for argument in sys.argv[1:] if Path(argument).suffix.lower() == ".pth"),
        None,
    )

    dataset = load_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images = torch.from_numpy(dataset.images).float().unsqueeze(1)
    labels = torch.from_numpy(dataset.target).long()

    # Split before constructing loaders or updating model parameters. Stratifying
    # keeps each person's representation proportionate in both partitions.
    full_dataset = TensorDataset(images, labels)
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)),
        test_size=0.25,
        random_state=42,
        stratify=dataset.target,
    )
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = build_model(images.shape[2], images.shape[3], len(dataset.target_names))
    model.to(device)

    if model_path is not None:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        visualize_predictions(model, test_dataset, dataset.target_names, device)
        return

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch, until = 0, epochs

    while True:
        for epoch in range(epoch, until):
            model.train()
            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(batch_images), batch_labels)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{until} loss: {loss.item():.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_images, batch_labels in test_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)
                correct += (model(batch_images).argmax(dim=1) == batch_labels).sum().item()
        print(f"Test accuracy: {correct / len(test_dataset):.4f}")

        try:
            input()
        except EOFError:
            return
        except KeyboardInterrupt:
            break

        until += epochs

    model_path = Path(f"cnn_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pth")
    torch.save(model, model_path)
    print(f"Saved model: {model_path}")
    visualize_predictions(model, test_dataset, dataset.target_names, device)

if __name__ == "__main__":
    main()
