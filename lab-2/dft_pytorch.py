"""Model: GPT-5.6 Sol. Adapt square-wave generation from NumPy to PyTorch."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

if torch.cuda.is_available():
    device_name = "cuda"
else:
    device_name = "cpu"

print(f"PYTORCH USING DEVICE: {device_name}")
device = torch.device(device_name)

N = 2048
T = 1.0
f0 = 1

harmonics = [1, 3, 5]

def square_wave(t):
    return torch.sign(torch.sin(2.0 * torch.pi * f0 * t))

def square_wave_fourier(t, f0, N):
    result = torch.zeros_like(t)
    for k in range(N):
        n = 2 * k + 1
        result += torch.sin(2 * torch.pi * n * f0 * t) / n
    return (4 / torch.pi) * result

def square_wave_fourier_vectorized(t, f0, N):
    n = torch.arange(1, 2 * N, 2, dtype=t.dtype, device=t.device)
    terms = torch.sin(2 * torch.pi * n[:, None] * f0 * t) / n[:, None]
    return (4 / torch.pi) * terms.sum(dim=0)

t = torch.arange(N, dtype=torch.float64, device=device) * (T / N)
square = square_wave(t)

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.plot(t.cpu(), square.cpu(), 'k', label='Square wave')
plt.title('Original Square Wave')
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.legend()

for i, Nh in enumerate(harmonics, start=2):
    plt.subplot(2, 3, i)
    y = square_wave_fourier(t, f0, Nh)
    plt.plot(t.cpu(), y.cpu(), label=f"N={Nh} harmonics")
    plt.plot(t.cpu(), square.cpu(), 'k--', alpha=0.5, label="Square wave")
    plt.title(f"Fourier Approximation with N={Nh}")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

def naive_dft(x):
    """
    Compute the Discrete Fourier Transform ( DFT ) of a 1 D signal .
    This is a "naive" implementation that directly follows the DFT formula ,
    which has a time complexity of O(N ^2) .
    Args:
        x(np.ndarray): The input signal, a 1D NumPy array .
    Returns:
        np.ndarray: The complex-valued DFT of the input signal .
    """
    N = len (x)
    # Create an empty array of complex numbers to store the DFT results
    X = np.zeros(N, dtype=np.complex128)
    # Iterate through each frequency bin ( k )
    for k in range(N):
        # For each frequency bin , sum the contributions from all input samples ( n )
        for n in range(N):
            # The core DFT formula : x [ n ] * e ^( -2 j * pi * k * n / N )
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X

def naive_dft_pytorch(x):
    """Compute the DFT with PyTorch tensor operations and no Python loops."""
    N = x.shape[0]
    indices = torch.arange(N, device=x.device, dtype=x.dtype)
    angles = -2j * torch.pi * torch.outer(indices, indices) / N
    return torch.exp(angles) @ x.to(torch.complex128)

# Construct a square wave using 50 harmonics
signal = square_wave_fourier(t, f0, 50)
signal_numpy = signal.cpu().numpy()
# Time the naive DFT implementation
start_time_naive = time.time()
dft_result = naive_dft(signal_numpy)
end_time_naive = time.time()
naive_duration = end_time_naive - start_time_naive
# Time the vectorized PyTorch DFT implementation
torch.cuda.synchronize()
start_time_pytorch = time.time()
dft_result_pytorch = naive_dft_pytorch(signal)
torch.cuda.synchronize()
end_time_pytorch = time.time()
pytorch_duration = end_time_pytorch - start_time_pytorch
# Time NumPy's FFT implementation
start_time_fft = time.time()
fft_result = np.fft.fft(signal_numpy)
end_time_fft = time.time()
fft_duration = end_time_fft - start_time_fft
# 3. Print Timings and Verification
print ("--- DFT / FFT Performance Comparison ---")
print (f"Naive DFT Execution Time: {naive_duration:.6f} seconds")
print (f"Vectorized PyTorch DFT Execution Time: {pytorch_duration:.6f} seconds")
print (f"NumPy FFT Execution Time: {fft_duration:.6f} seconds")
if pytorch_duration > 0:
    print(f"PyTorch DFT is approximately {naive_duration / pytorch_duration:.2f} times faster than the naive DFT.")
else:
    print("PyTorch DFT was too fast to measure a significant speed difference.")
# It's possible for the FFT to be so fast that the duration is 0.0, so we handle that case.
if fft_duration > 0:
    print(f"FFT is approximately {naive_duration / fft_duration:.2f} times faster.")
else:
    print("FFT was too fast to measure a significant duration difference.")
# Check if our implementation is close to NumPy's result
# np.allclose is used for comparing floating - point arrays .
print(f"\nOur DFT implementation is close to NumPy's FFT:{np.allclose(dft_result, fft_result)}")
print(
    "Vectorized PyTorch DFT is close to NumPy's FFT:"
    f"{np.allclose(dft_result_pytorch.cpu().numpy(), fft_result)}"
)
# 4. Prepare for Plotting
# Generate the frequency axis for the plot.
# np.fft.fftfreq returns the DFT sample frequencies.
# We only need the first half of the frequencies (the positive ones) due to symmetry.
xf = np.fft.fftfreq(N, d=T / N)[:N // 2]
# We normalize the magnitude by N and multiply by 2 to get the correct amplitude.
magnitude = 2.0 / N * np.abs(dft_result[0:N // 2])
# 5. Visualize the Results
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
# Plot the original time-domain signal
ax1.plot(t.cpu(), signal.cpu(), color='c')
ax1.set_title('Input Sine Wave Signal', fontsize=16)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlim(0, 1.0) # Show a few cycles of the sine wave
ax1.grid(True)
# Plot the frequency-domain signal (magnitude of the DFT)
ax2.stem(xf, magnitude, basefmt=" ")
ax2.set_title(
    'Discrete Fourier Transform (Magnitude Spectrum)',
    fontsize =16
)
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Magnitude', fontsize=12)
ax2.set_xlim(0, 50) # Focus on lower frequencies
ax2.grid(True)
# Add vertical lines for the first ten frequencies
for i in range(20):
    if i < len(xf) and i % 2 == 1: # Only plot odd harmonics
        ax2.axvline(
            xf[i], color='r', linestyle='--', alpha=0.7,
            label=f'f{i}: {i}* f0 = {xf[i]:.1f} Hz'
        )

# Only show labels for first 3 frequencies to avoid cluttering
ax2.legend()

plt.tight_layout()
plt.show()
