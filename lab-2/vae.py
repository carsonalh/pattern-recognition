"""Train a temporary linear reconstruction model on the PNG slices dataset."""

import argparse
import math
import os
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DATA_DIR = Path("data")
LOCAL_DATA_ARCHIVE = DATA_DIR / "keras_png_slices_data.zip"
CLUSTER_DATA_DIR = Path("/home/groups/comp3710/OASIS")
DATA_ROOT = "keras_png_slices_data"
WARMUP_EPOCHS = 5
DEFAULT_NUM_WORKERS = max(os.cpu_count() or 1, 8)
START_TIME = time.monotonic()


def log(message):
    """Print a message with the elapsed time since this process started."""
    elapsed = time.monotonic() - START_TIME
    print(f"[{elapsed:10.2f}s] {message}", flush=True)


def resolve_data_source():
    """Prefer the local archive, falling back to the cluster's extracted data."""
    if LOCAL_DATA_ARCHIVE.exists():
        return LOCAL_DATA_ARCHIVE
    if CLUSTER_DATA_DIR.exists():
        return CLUSTER_DATA_DIR
    raise FileNotFoundError(
        f"Could not find {LOCAL_DATA_ARCHIVE} or {CLUSTER_DATA_DIR}"
    )


DATA_SOURCE = resolve_data_source()


class ZipImageDataset(Dataset):
    """Read grayscale images lazily from an archive or extracted data directory."""

    def __init__(self, data_source, split, transform=None):
        self.data_source = Path(data_source)
        self.transform = transform or transforms.ToTensor()
        if self.data_source.is_dir():
            self.members = sorted((self.data_source / split).glob("*.png"))
        else:
            prefix = f"{DATA_ROOT}/{split}/"
            with zipfile.ZipFile(self.data_source) as archive:
                self.members = [
                    name
                    for name in archive.namelist()
                    if name.startswith(prefix) and name.endswith(".png")
                ]
        self._archive = None

        if not self.members:
            raise ValueError(f"No PNG images found for split {split!r}")

    def __len__(self):
        return len(self.members)

    def _get_archive(self):
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.data_source)
        return self._archive

    def __getitem__(self, index):
        if self.data_source.is_dir():
            with Image.open(self.members[index]) as image:
                image = image.convert("L")
        else:
            with self._get_archive().open(self.members[index]) as image_file:
                with Image.open(image_file) as image:
                    image = image.convert("L")
        return self.transform(image)


def image_shape(data_source=DATA_SOURCE, split="keras_png_slices_train"):
    """Return the tensor shape of one image without retaining an archive handle."""
    data_source = Path(data_source)
    if data_source.is_dir():
        image_path = next((data_source / split).glob("*.png"))
        with Image.open(image_path) as image:
            return (1, image.height, image.width)

    prefix = f"{DATA_ROOT}/{split}/"
    with zipfile.ZipFile(data_source) as archive:
        member = next(
            name
            for name in archive.namelist()
            if name.startswith(prefix) and name.endswith(".png")
        )
        with archive.open(member) as image_file:
            with Image.open(image_file) as image:
                return (1, image.height, image.width)


def build_dataloaders(
    data_source=DATA_SOURCE,
    batch_size=64,
    num_workers=DEFAULT_NUM_WORKERS,
):
    """Return training, validation, and test loaders from the dataset source."""
    transform = transforms.ToTensor()
    train_dataset = ZipImageDataset(
        data_source, "keras_png_slices_train", transform
    )
    validation_dataset = ZipImageDataset(
        data_source, "keras_png_slices_validate", transform
    )
    test_dataset = ZipImageDataset(data_source, "keras_png_slices_test", transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, validation_loader, test_loader


class LinearVAEPlaceholder(nn.Module):
    """Temporary trainable placeholder; replace this with the actual VAE."""

    def __init__(self, image_width):
        super().__init__()
        self.linear = nn.Linear(image_width, image_width)

    def forward(self, images):
        return self.linear(images)


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device):
    model.train()
    amp_enabled = device.type == "cuda"
    total_loss = 0.0
    total = 0
    for images in loader:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            reconstructions = model(images)
            loss = loss_fn(reconstructions, images)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    amp_enabled = device.type == "cuda"
    total_loss = 0.0
    total = 0
    for images in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            reconstructions = model(images)
            loss = loss_fn(reconstructions, images)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / total


def main(
    epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    num_workers=DEFAULT_NUM_WORKERS,
    compile_model=False,
):
    log(
        "Arguments: "
        f"epochs={epochs}, batch_size={batch_size}, "
        f"learning_rate={learning_rate}, "
        f"num_workers={num_workers}, "
        f"compile_model={compile_model}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    log(f"Loading dataset from: {DATA_SOURCE}")

    train_loader, validation_loader, _test_loader = build_dataloaders(
        data_source=DATA_SOURCE, batch_size=batch_size, num_workers=num_workers
    )
    _, _, image_width = image_shape(DATA_SOURCE)
    breakpoint()
    model = LinearVAEPlaceholder(image_width).to(device)
    if compile_model:
        model = torch.compile(model)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def learning_rate_schedule(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        cosine_epochs = max(epochs - WARMUP_EPOCHS, 1)
        progress = min((epoch - WARMUP_EPOCHS) / cosine_epochs, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_schedule
    )

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device
        )
        validation_loss = evaluate(model, validation_loader, loss_fn, device)
        scheduler.step()
        log(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.6f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
        help="number of data loader workers",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="compile_model",
        help="compile the model with torch.compile",
    )
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        compile_model=args.compile_model,
    )
