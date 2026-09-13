"""Train a convolutional neural network on the LFW dataset."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from common import load_lfw_dataset


class CNN(nn.Module):
    def __init__(self, image_width: int, image_height: int, classes: int):
        super().__init__()
        hidden_linear_layers = 32 * (image_width // 4) * (image_height // 4)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(hidden_linear_layers, hidden_linear_layers),
            nn.ReLU(),
            nn.Linear(hidden_linear_layers, classes),
        )

    def forward(self, x):
        return self.layers(x)


def main(epochs=40, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lfw = load_lfw_dataset()
    images = torch.from_numpy(lfw.images).float().unsqueeze(1)
    labels = torch.from_numpy(lfw.target).long()
    dataset = TensorDataset(images, labels)
    train_dataset, test_dataset = random_split(
        dataset,
        [0.75, 0.25],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = CNN(images.shape[3], images.shape[2], len(lfw.target_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train loss: {total_loss / len(train_dataset):.4f}"
        )

    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            total_loss += loss_fn(predictions, labels).item() * len(images)
            correct += (predictions.argmax(dim=1) == labels).sum().item()

    print(f"Test loss: {total_loss / len(test_dataset):.4f}")
    print(f"Test accuracy: {correct / len(test_dataset):.4f}")


if __name__ == "__main__":
    main()
