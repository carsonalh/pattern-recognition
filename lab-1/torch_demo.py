import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("PyTorch Version:", torch.__version__)
    device_kind = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device kind:", device_kind)
    device = torch.device(device_kind)
    x, y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
    x, y = torch.Tensor(x).to(device), torch.Tensor(y).to(device)
    z = torch.exp(-(x**2 + y**2) / 2.0)
    plt.imshow(z.cpu())
    plt.tight_layout()
    plt.show()
