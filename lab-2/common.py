from pathlib import Path, PurePosixPath
import zipfile

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ZipImageDataset(Dataset):
    """Read grayscale images lazily from an archive or extracted data directory."""

    def __init__(self, data_source, split, transform=None):
        self.data_source = Path(data_source)
        split = str(split)
        self.transform = transform or transforms.ToTensor()
        if self.data_source.is_dir():
            split_dir = self.data_source / split
            if not split_dir.is_dir():
                split_dirs = [
                    path
                    for path in self.data_source.rglob(split)
                    if path.is_dir()
                ]
                if len(split_dirs) == 1:
                    split_dir = split_dirs[0]
            self.members = sorted(split_dir.glob("*.png"))
        else:
            with zipfile.ZipFile(self.data_source) as archive:
                self.members = [
                    name
                    for name in archive.namelist()
                    if PurePosixPath(name).parent.name == split
                    and name.endswith(".png")
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
        return self.transform(image), 0
