import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import argparse
import sys

class Generator(nn.Module):
    def __init__(self, fc=False, latent_dims=64):
        super().__init__()

        if fc:
            self.generate = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(latent_dims, 128),
                    nn.LeakyReLU(0.1),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, 28 * 28),
                    nn.Tanh(),
                    nn.Unflatten(1, (1, 28, 28)),
            )
        else:
            features = 32
            self.generate = nn.Sequential(
                nn.ConvTranspose2d(latent_dims, 4 * features, kernel_size=4, stride=1, bias=False),
                nn.BatchNorm2d(4 * features),
                nn.ReLU(),
                nn.ConvTranspose2d(4 * features, 2 * features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(2 * features),
                nn.ReLU(),
                nn.ConvTranspose2d(2 * features, features, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(),
                nn.ConvTranspose2d(features, 1, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh(),
            )

    def forward(self, x):
        """Expects an 8x8 image noise pattern as input."""
        return self.generate(x)


class Discriminator(nn.Module):
    def __init__(self, fc=False):
        super().__init__()
        if fc:
            self.discriminate = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(28 * 28, 128),
                nn.LeakyReLU(0.1),
                nn.Linear(128, 1),
            )
        else:
            features = 32
            self.discriminate = nn.Sequential(
                nn.Conv2d(1, features, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(features, 2 * features, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(2 * features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(2 * features, 4 * features, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(4 * features, 8 * features, kernel_size=2, padding=1, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.Linear(8 * features, 1),
                # No sigmoid: BCEWithLogitsLoss handles that for us
            )


    def forward(self, x):
        return self.discriminate(x)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Need CUDA to train GAN")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fc-generator", action="store_true")
    parser.add_argument("--fc-discriminator", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    training_data = torchvision.datasets.MNIST("data", train=True, download=True, transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,), (0.5,)),
    ])
    )

    device = "cuda"
    batch_size = 64
    training_loader = DataLoader(training_data, batch_size=batch_size)
    latent_dims = 10

    gen = Generator(fc=args.fc_generator, latent_dims=latent_dims).to(device)
    dis = Discriminator(fc=args.fc_discriminator).to(device)

    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=3e-4, betas=(0.5, 0.999))
    gen_loss_fn = torch.nn.BCEWithLogitsLoss()

    dis_optimizer = torch.optim.Adam(dis.parameters(), lr=3e-4, betas=(0.5, 0.999))
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
            gen_loss = gen_loss_fn(dis(gen(noise)).squeeze(), real)
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            # Train discriminator ...
            dis_optimizer.zero_grad()
            noise = torch.randn((batch_size, latent_dims, 1, 1), device=device)
            # ... with both real examples
            dis_real_loss = dis_loss_fn(dis(images).squeeze(), real)
            # ... and fake examples
            dis_fake_loss = dis_loss_fn(dis(gen(noise)).squeeze(), fake)
            # This should be fine since discriminator assumes real/fake class probabilities are equal
            dis_loss = 0.5 * (dis_real_loss + dis_fake_loss)
            dis_loss.backward()
            dis_optimizer.step()

            gen_loss_total += gen_loss
            dis_loss_total += dis_loss

        with torch.no_grad():
            sample_images = 5
            noise = torch.randn((sample_images, latent_dims, 1, 1), device=device)
            images = gen(noise)
            if args.fc_generator and args.fc_discriminator:
                prefix = "gan_images/fc"
            else:
                prefix = "gan_images/cnn"
            for i in range(sample_images):
                plt.imshow(images[i][0].cpu())
                plt.savefig(f"{prefix}/image_e{epoch}_i{i}")
                plt.close()

        print(f"Epoch {epoch+1}/{args.epochs}: dis. loss: {dis_loss_total.cpu().item():.4f}, gen. loss: {gen_loss_total.cpu().item():.4f}")
