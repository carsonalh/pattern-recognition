from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import argparse
import sys
from enum import Enum, auto
from common import ZipImageDataset

class GANDataset(Enum):
    MNIST = auto()
    OASIS = auto()


class Generator(nn.Module):
    def __init__(self, latent_dims=64, dataset: GANDataset = GANDataset.OASIS):
        super().__init__()

        if dataset == GANDataset.MNIST:
            feature_dims = 32
            self.generate = nn.Sequential(
                nn.ConvTranspose2d(latent_dims, 4 * feature_dims, kernel_size=4, stride=1, bias=False),
                nn.BatchNorm2d(4 * feature_dims),
                nn.ReLU(),
                nn.ConvTranspose2d(4 * feature_dims, 2 * feature_dims, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(2 * feature_dims),
                nn.ReLU(),
                nn.ConvTranspose2d(2 * feature_dims, feature_dims, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(feature_dims),
                nn.ReLU(),
                nn.ConvTranspose2d(feature_dims, 1, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh(),
            )
        elif dataset == GANDataset.OASIS:
            feature_dims = 32
            self.generate = nn.Sequential(
                nn.ConvTranspose2d(latent_dims, 8 * feature_dims, kernel_size=4, stride=1, bias=False),
                nn.BatchNorm2d(8 * feature_dims),
                nn.ReLU(),
                nn.ConvTranspose2d(8 * feature_dims, 4 * feature_dims, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(4 * feature_dims),
                nn.ReLU(),
                nn.ConvTranspose2d(4 * feature_dims, 2 * feature_dims, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(2 * feature_dims),
                nn.ReLU(),
                nn.ConvTranspose2d(2 * feature_dims, feature_dims, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(feature_dims),
                nn.ReLU(),
                nn.ConvTranspose2d(feature_dims, 1, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh(),
                # nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                # nn.ConvTranspose2d(feature_dims, 1, kernel_size=4, stride=2, padding=1, bias=False),
            )

    def forward(self, x):
        return self.generate(x)


class Discriminator(nn.Module):
    def __init__(self, dataset: GANDataset = GANDataset.MNIST):
        super().__init__()
        if dataset == GANDataset.MNIST:
            features = 32
            self.discriminate = nn.Sequential(
                nn.Conv2d(1, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(features, 2 * features, kernel_size=3, padding=1, stride=2, bias=False),
                # nn.BatchNorm2d(2 * features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(2 * features, 4 * features, kernel_size=3, padding=1, stride=2, bias=False),
                # nn.BatchNorm2d(4 * features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(4 * features, 8 * features, kernel_size=2, padding=1, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.Linear(8 * features, 1),
                # No sigmoid: BCEWithLogitsLoss handles that for us
            )
        elif dataset == GANDataset.OASIS:
            features = 8
            self.discriminate = nn.Sequential(
                nn.Conv2d(1, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(features, 2 * features, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(2 * features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(2 * features, 4 * features, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(4 * features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(4 * features, 8 * features, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(8 * features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(8 * features, 16 * features, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(16 * features),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.Conv2d(16 * features, 32 * features, kernel_size=3, padding=1, stride=2),
                # nn.BatchNorm2d(32 * features),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Conv2d(32 * features, 64 * features, kernel_size=3, padding=1, stride=2),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Conv2d(64 * features, 1, kernel_size=4),
                nn.Conv2d(16 * features, 1, kernel_size=4),
                nn.Flatten(1),
                # No sigmoid: BCEWithLogitsLoss handles that for us
            )

    def forward(self, x):
        return self.discriminate(x)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Need CUDA to train GAN")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mnist", action='store_true')
    args = parser.parse_args()

    if args.mnist:
        training_data = torchvision.datasets.MNIST("data", train=True, download=True, transform=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,)),
        ])
        )
    else:
        paths = [
            Path("data/keras_png_slices_data.zip"),
            Path("/home/groups/comp3710/OASIS"),
        ]
        for path in paths:
            if path.exists():
                found = path
                break
        else:
            raise RuntimeError(f"Cannot find OASIS dataset locally, checked: {paths}")
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((64, 64)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,))
        ])
        training_data = ZipImageDataset(found, "keras_png_slices_train", transform)

    device = "cuda"
    batch_size = 64
    training_loader = DataLoader(training_data, batch_size=batch_size)
    latent_dims = 10

    dataset = GANDataset.MNIST if args.mnist else GANDataset.OASIS

    gen = Generator(latent_dims=latent_dims, dataset=dataset).to(device)
    dis = Discriminator(dataset=dataset).to(device)

    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=3e-4, betas=(0.5, 0.999))
    gen_loss_fn = torch.nn.BCEWithLogitsLoss()

    dis_optimizer = torch.optim.Adam(dis.parameters(), lr=3e-5, betas=(0.5, 0.999))
    dis_loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        gen_loss_total = torch.zeros((), device=device)
        dis_loss_total = torch.zeros((), device=device)

        for images, _labels in training_loader:
            images = images.to(device)

            batch_size = images.shape[0]
            real = torch.ones((batch_size,), device=device)
            fake = torch.zeros((batch_size,), device=device)

            # Train generator
            noise = torch.randn((batch_size, latent_dims, 1, 1), device=device)
            gen_loss = gen_loss_fn(dis(gen(noise)).view(-1), real)
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            # Train discriminator ...
            dis_optimizer.zero_grad()
            noise = torch.randn((batch_size, latent_dims, 1, 1), device=device)
            # ... with both real examples
            dis_real_loss = dis_loss_fn(dis(images).view(-1), real)
            # ... and fake examples
            dis_fake_loss = dis_loss_fn(dis(gen(noise)).view(-1), fake)
            # This should be fine since discriminator assumes real/fake class probabilities are equal
            dis_loss = 0.5 * (dis_real_loss + dis_fake_loss)
            dis_loss.backward()
            dis_optimizer.step()

            gen_loss_total += gen_loss
            dis_loss_total += dis_loss

        with torch.no_grad():
            sample_images_side = 5
            sample_images = sample_images_side * sample_images_side
            noise = torch.randn((sample_images, latent_dims, 1, 1), device=device)
            images = gen(noise)
            if args.mnist:
                prefix = "gan_images/MNIST"
            else:
                prefix = "gan_images/OASIS"
            shown_images = min(sample_images, images.shape[0])
            grid_image = vutils.make_grid(images[:shown_images], nrow=sample_images_side, padding=2)
            plt.imshow(grid_image[0].cpu(), cmap='gray')
            plt.savefig(f"{prefix}/image_e{epoch+1}", dpi=600)
            plt.close()

        print(f"Epoch {epoch+1}/{args.epochs}: dis. loss: {dis_loss_total.cpu().item():.4f}, gen. loss: {gen_loss_total.cpu().item():.4f}")
