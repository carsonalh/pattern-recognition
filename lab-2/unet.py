"""Train a temporary U-Net segmentation model on the PNG slices dataset."""

import argparse
import math
import os
import signal
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
NUM_CLASSES = 4
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


def segmentation_split(split):
    """Return the segmentation-mask directory for an image split."""
    suffix = split.removeprefix("keras_png_slices_")
    return f"keras_png_slices_seg_{suffix}"


def segmentation_target(image):
    """Convert the mask's 0/85/170/255 encoding into class indices."""
    target = transforms.ToTensor()(image).mul(NUM_CLASSES - 1).round()
    return target.to(torch.long).squeeze(0)


class ZipSegmentationDataset(Dataset):
    """Read image/mask pairs lazily from an archive or extracted data directory."""

    def __init__(self, data_source, split, image_transform=None):
        self.data_source = Path(data_source)
        self.image_transform = image_transform or transforms.ToTensor()
        mask_split = segmentation_split(split)

        if self.data_source.is_dir():
            image_members = sorted((self.data_source / split).glob("*.png"))
            self.members = [
                (
                    image_path,
                    self.data_source / mask_split / image_path.name.replace(
                        "case_", "seg_", 1
                    ),
                )
                for image_path in image_members
            ]
        else:
            image_prefix = f"{DATA_ROOT}/{split}/"
            mask_prefix = f"{DATA_ROOT}/{mask_split}/"
            with zipfile.ZipFile(self.data_source) as archive:
                image_names = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith(image_prefix) and name.endswith(".png")
                )
                self.members = [
                    (
                        image_name,
                        mask_prefix
                        + Path(image_name).name.replace("case_", "seg_", 1),
                    )
                    for image_name in image_names
                ]
                available_members = set(archive.namelist())
                missing_masks = [
                    mask_name
                    for _, mask_name in self.members
                    if mask_name not in available_members
                ]
                if missing_masks:
                    raise ValueError(
                        f"Missing segmentation masks, including {missing_masks[0]!r}"
                    )
        self._archive = None

        if not self.members:
            raise ValueError(f"No PNG images found for split {split!r}")
        missing_masks = [
            mask_path
            for _, mask_path in self.members
            if self.data_source.is_dir() and not mask_path.exists()
        ]
        if missing_masks:
            raise ValueError(f"Missing segmentation mask {missing_masks[0]!r}")

    def __len__(self):
        return len(self.members)

    def _get_archive(self):
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.data_source)
        return self._archive

    def _read_image(self, member):
        if self.data_source.is_dir():
            with Image.open(member) as image:
                return image.convert("L")
        with self._get_archive().open(member) as image_file:
            with Image.open(image_file) as image:
                return image.convert("L")

    def __getitem__(self, index):
        image_member, mask_member = self.members[index]
        image = self.image_transform(self._read_image(image_member))
        target = segmentation_target(self._read_image(mask_member))
        return image, target


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


def ignore_sigint(_worker_id):
    """Keep data-loader workers alive so the main process can handle Ctrl-C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def build_dataloaders(
    data_source=DATA_SOURCE,
    batch_size=64,
    num_workers=DEFAULT_NUM_WORKERS,
):
    """Return training, validation, and test loaders from the dataset source."""
    train_dataset = ZipSegmentationDataset(
        data_source, "keras_png_slices_train"
    )
    validation_dataset = ZipSegmentationDataset(
        data_source, "keras_png_slices_validate"
    )
    test_dataset = ZipSegmentationDataset(data_source, "keras_png_slices_test")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": ignore_sigint,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, validation_loader, test_loader


class UnetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.convs(x)


class Unet(nn.Module):
    """Assumes 256x256 images as input"""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = UnetDoubleConv(1, 64)
        self.down2 = UnetDoubleConv(64, 128)
        self.down3 = UnetDoubleConv(128, 256)
        self.down4 = UnetDoubleConv(256, 512)
        self.up1 = nn.Sequential(UnetDoubleConv(512, 1024), nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2))
        self.up2 = nn.Sequential(UnetDoubleConv(1024, 512), nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        self.up3 = nn.Sequential(UnetDoubleConv(512, 256), nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))
        self.up4 = nn.Sequential(UnetDoubleConv(256, 128), nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        self.up5 = nn.Sequential(UnetDoubleConv(128, 64), nn.Conv2d(64, num_classes, kernel_size=1))
        self.num_classes = num_classes

    def forward(self, images):
        x1 = self.down1(images)
        x2 = self.down2(self.mp1(x1))
        x3 = self.down3(self.mp2(x2))
        x4 = self.down4(self.mp3(x3))
        out = torch.cat((self.up1(self.mp4(x4)), x4), dim=1)
        out = torch.cat((self.up2(out), x3), dim=1)
        out = torch.cat((self.up3(out), x2), dim=1)
        out = torch.cat((self.up4(out), x1), dim=1)
        out = self.up5(out)
        return out


@torch.no_grad()
def dsc(logits, targets, num_classes=NUM_CLASSES, smooth=1e-6):
    """Return mean foreground Dice similarity coefficient for a batch."""
    predictions = logits.argmax(dim=1)
    scores = []
    for class_index in range(1, num_classes):
        predicted_class = predictions == class_index
        target_class = targets == class_index
        intersection = (predicted_class & target_class).sum().float()
        denominator = predicted_class.sum() + target_class.sum()
        scores.append((2 * intersection + smooth) / (denominator + smooth))
    return torch.stack(scores).mean().item()


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device):
    model.train()
    amp_enabled = device.type == "cuda"
    total_loss = 0.0
    total_dsc = 0.0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            logits = model(images)
            loss = loss_fn(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dsc += dsc(logits, targets) * batch_size
        total += batch_size
    return total_loss / total, total_dsc / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    amp_enabled = device.type == "cuda"
    total_loss = 0.0
    total_dsc = 0.0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            logits = model(images)
            loss = loss_fn(logits, targets)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dsc += dsc(logits, targets) * batch_size
        total += batch_size
    return total_loss / total, total_dsc / total


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
    channels, height, width = image_shape(DATA_SOURCE)
    log(f"Input image shape: channels={channels}, height={height}, width={width}")
    model = Unet().to(device)
    if compile_model:
        model = torch.compile(model)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
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
        train_loss, train_dsc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device
        )
        validation_loss, validation_dsc = evaluate(
            model, validation_loader, loss_fn, device
        )
        scheduler.step()
        log(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train loss: {train_loss:.4f}, train DSC: {train_dsc:.4f} | "
            f"validation loss: {validation_loss:.4f}, "
            f"validation DSC: {validation_dsc:.4f} | "
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
