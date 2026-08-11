import numpy as np
import matplotlib.pyplot as plt

# Grid range and resolution
x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)

# 2D Gaussian parameters
mu_x, mu_y = 0.0, 0.0
sigma_x, sigma_y = 1.0, 1.0

# 2D Gaussian function
Z = np.exp(-(((X - mu_x) ** 2) / (2 * sigma_x ** 2) + ((Y - mu_y) ** 2) / (2 * sigma_y ** 2)))

# Plot with imshow
plt.figure(figsize=(6, 5))
plt.imshow(
    Z,
    extent=[-4, 4, -4, 4],
    origin="lower",
    cmap="viridis",
    aspect="equal",
)
plt.colorbar(label="Gaussian value")
plt.title("2D Gaussian Function")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
