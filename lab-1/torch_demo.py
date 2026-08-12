import torch
import math
import numpy as np
import matplotlib.pyplot as plt

def gaussian_plot():
    x, y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
    x, y = torch.Tensor(x).to(device), torch.Tensor(y).to(device)
    z = torch.exp(-(x**2 + y**2) / 2.0)
    plt.imshow(z.cpu().numpy())
    plt.tight_layout()
    plt.show()


def sinusoid_plot():
    x, y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
    x, y = torch.Tensor(x).to(device), torch.Tensor(y).to(device)
    z = torch.sin(3 * (x + y))
    plt.imshow(z.cpu().numpy())
    plt.tight_layout()
    plt.show()


def gabor_plot():
    x, y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
    x, y = torch.Tensor(x).to(device), torch.Tensor(y).to(device)
    USE_REAL_FORMULA = False
    if USE_REAL_FORMULA:
        theta = math.pi / 4
        sigma = 1.0
        xp = x * np.cos(theta) + y * np.sin(theta)
        yp = -x * np.sin(theta) + y * np.cos(theta)
        z = torch.exp(-(xp**2 + yp**2) / (2.0 * sigma)) * torch.sin(2.0 * math.pi * xp)
    else:
        z = torch.exp(-(x**2 + y**2) / 2.0) * torch.sin(3 * (x + y))
    plt.imshow(z.cpu().numpy())
    plt.tight_layout()
    plt.show()


def sierpinski_plot():
    scale = np.array([[0.5, 0.0], [0.0, 0.5]])
    vertices = np.array([[[-0.5], [0]], [[0.5], [0]], [[0], [1]]])
    offsets = [np.array([[0.0], [0.5]]), np.array([[-0.25], [0.0]]), np.array([[0.25], [0.0]])]

    for _ in range(8):
        vertices = scale @ vertices
        vertices = np.concatenate((vertices + offsets[0], vertices + offsets[1], vertices + offsets[2]), axis=0)

    assert len(vertices) % 3 == 0

    plt.xlim([-0.5, 0.5])

    for i in range(len(vertices) // 3):
        sli = vertices[3*i:3*(i + 1)]
        tri = plt.Polygon(sli.reshape((3, 2)), color='black')
        plt.gca().add_patch(tri)

    plt.show()


def mandelbrot_plot(*, zoom):
    if zoom:
        width, steps = 1e-2, 5000
        re_begin, im_begin = -0.6549505794779913 + 5 *1e-3, 0.41727211410726495
        re_range, im_range = np.arange(re_begin, re_begin + width, width / steps), np.arange(im_begin, im_begin + width, width / steps)
    else:
        re_range, im_range = np.arange(-2.0, 1.0, 0.001), np.arange(-1.3, 1.3, 0.001)

    re, im = np.meshgrid(re_range, im_range, indexing="ij")
    re, im = torch.Tensor(re).to(device), torch.Tensor(im).to(device)
    c = re + 1j * im
    z = torch.zeros(c.shape).to(device)

    for _ in range(1000):
        z = z**2 + c

    plt.imshow(torch.abs(z).cpu().T)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("PyTorch Version:", torch.__version__)
    device_kind = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device kind:", device_kind)
    device = torch.device(device_kind)
    PRESENT = False
    if PRESENT:
        gaussian_plot()
        sinusoid_plot()
        gabor_plot()
        mandelbrot_plot(zoom=False)
        mandelbrot_plot(zoom=True)
    else:
        sierpinski_plot()
