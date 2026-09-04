"""Train a ResNet-18 classifier on CIFAR-10."""

import argparse
import time

import torch
from torch import nn, Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets, transforms


DATA_DIR = "data"
NUM_CLASSES = 10
START_TIME = time.monotonic()


def log(message):
    """Print a message with the elapsed time since this process started."""
    elapsed = time.monotonic() - START_TIME
    print(f"[{elapsed:10.2f}s] {message}", flush=True)


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label


def build_dataloaders(data_dir=DATA_DIR, batch_size=128, num_workers=2):
    """Download CIFAR-10 and return training, validation, and test loaders."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_data = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=None
    )
    test_data = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=None
    )
    all_data = ConcatDataset([train_data, test_data])
    train_size = int(0.8 * len(all_data))
    validation_size = int(0.15 * len(all_data))
    test_size = len(all_data) - train_size - validation_size
    train_subset, validation_subset, test_subset = random_split(
        all_data,
        [train_size, validation_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_dataset = TransformedDataset(train_subset, train_transform)
    validation_dataset = TransformedDataset(validation_subset, validation_transform)
    test_dataset = TransformedDataset(test_subset, validation_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, validation_loader, test_loader


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, downsample: bool = False) -> None:
        super().__init__()
        out_channels = 2 * in_channels if downsample else in_channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2 if downsample else 1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # Even though the batchnorm is always the same operation, we need multiple copies of it since otherwise autograd
        # will confuse them
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2,
                    bias=False, # No need to learn a bias since biases are necessarily stripped by following batch norm
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    """Scaffold for a ResNet-18; currently only performs linear classification."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # We're working with 32 x 32 images for CIFAR10
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2_1 = ResNetBlock(64)
        self.conv2_2 = ResNetBlock(64)
        self.conv3_1 = ResNetBlock(64, downsample=True)
        self.conv3_2 = ResNetBlock(128)
        self.conv4_1 = ResNetBlock(128, downsample=True)
        self.conv4_2 = ResNetBlock(256)
        self.conv5_1 = ResNetBlock(256, downsample=True)
        self.conv5_2 = ResNetBlock(512)
        self.reduce_classes = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=-3), # compress the last three dimensions into one, since the last two are 1x1 anyway
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.reduce_classes(x)
        return x


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device):
    model.train()
    amp_enabled = device.type == "cuda"
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            predictions = model(images)
            loss = loss_fn(predictions, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += (predictions.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    amp_enabled = device.type == "cuda"
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            predictions = model(images)
            loss = loss_fn(predictions, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (predictions.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main(epochs=50, batch_size=256, learning_rate=0.2):
    log(
        "Arguments: "
        f"epochs={epochs}, batch_size={batch_size}, "
        f"learning_rate={learning_rate}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    train_loader, validation_loader, _test_loader = build_dataloaders(
        batch_size=batch_size
    )
    model = ResNet18().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device
        )
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_fn, device
        )
        scheduler.step()
        log(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train loss: {train_loss:.4f}, train acc: {train_accuracy:.4f} | "
            f"validation loss: {validation_loss:.4f}, "
            f"validation acc: {validation_accuracy:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
