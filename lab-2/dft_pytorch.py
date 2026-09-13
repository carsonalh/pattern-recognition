"""Compare direct DFT implementations using a Fourier-series square wave."""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


SAMPLE_COUNTS = [256, 512, 1024, 2048]
SIGNAL_DURATION_SECONDS = 1.0
f0 = 1

APPROXIMATION_HARMONICS = [1, 3, 5, 20, 50]
DFT_SIGNAL_HARMONICS = 50


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


def select_device() -> torch.device:
    """Select the fastest available PyTorch device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PYTORCH USING DEVICE: {device.type}")
    return device


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU operations so benchmark timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def plot_fourier_approximations(
    time_samples: torch.Tensor,
    exact_square_wave: torch.Tensor,
) -> None:
    """Plot the square wave alongside several Fourier approximations."""
    column_count = 3
    plot_count = len(APPROXIMATION_HARMONICS) + 1
    row_count = math.ceil(plot_count / column_count)

    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(12, 4 * row_count),
        squeeze=False,
    )
    time_values = time_samples.cpu()
    square_wave_values = exact_square_wave.cpu()

    original_axis = axes.flat[0]
    original_axis.plot(time_values, square_wave_values, "k", label="Square wave")
    original_axis.set_title("Original Square Wave")
    original_axis.set_ylim(-1.5, 1.5)
    original_axis.grid(True)
    original_axis.legend()

    for axis, harmonic_count in zip(
        axes.flat[1:],
        APPROXIMATION_HARMONICS,
        strict=False,
    ):
        approximation = square_wave_fourier(
            time_samples,
            f0,
            harmonic_count,
        )
        axis.plot(
            time_values,
            approximation.cpu(),
            label=f"N={harmonic_count} harmonics",
        )
        axis.plot(
            time_values,
            square_wave_values,
            "k--",
            alpha=0.5,
            label="Square wave",
        )
        axis.set_title(f"Fourier Approximation with N={harmonic_count}")
        axis.set_ylim(-1.5, 1.5)
        axis.grid(True)
        axis.legend()

    for unused_axis in axes.flat[plot_count:]:
        unused_axis.set_visible(False)

    figure.tight_layout()
    plt.show()


def benchmark_transforms(
    signal: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    """Benchmark each transform and verify the direct DFT results."""
    signal_numpy = signal.cpu().numpy()

    start_time = time.perf_counter()
    naive_result = naive_dft(signal_numpy)
    naive_duration = time.perf_counter() - start_time

    synchronize(device)
    start_time = time.perf_counter()
    pytorch_result = naive_dft_pytorch(signal)
    synchronize(device)
    pytorch_duration = time.perf_counter() - start_time

    start_time = time.perf_counter()
    fft_result = np.fft.fft(signal_numpy)
    fft_duration = time.perf_counter() - start_time

    print(
        f"N={len(signal):4d} | Naive DFT matches NumPy FFT: "
        f"{np.allclose(naive_result, fft_result)} | "
        "PyTorch DFT matches NumPy FFT: "
        f"{np.allclose(pytorch_result.cpu().numpy(), fft_result)}"
    )

    durations = {
        "Naive DFT": naive_duration,
        "PyTorch DFT": pytorch_duration,
        "NumPy FFT": fft_duration,
    }
    return naive_result, durations


def plot_benchmark_results(
    sample_counts: list[int],
    durations: dict[str, list[float]],
) -> None:
    """Plot transform execution times as grouped bars for each sample count."""
    group_positions = np.arange(len(sample_counts))
    bar_width = 0.25
    method_count = len(durations)

    figure, axis = plt.subplots(figsize=(11, 6))
    for method_index, (method, method_durations) in enumerate(durations.items()):
        offset = (method_index - (method_count - 1) / 2) * bar_width
        axis.bar(
            group_positions + offset,
            method_durations,
            bar_width,
            label=method,
        )

    axis.set_title("DFT / FFT Performance by Number of Samples")
    axis.set_xlabel("Number of samples (N)")
    axis.set_ylabel("Execution time (seconds, log scale)")
    axis.set_xticks(group_positions, [str(count) for count in sample_counts])
    axis.set_yscale("log")
    axis.legend()
    axis.grid(axis="y", which="both", alpha=0.3)
    figure.tight_layout()
    plt.show()


def plot_dft_spectrum(
    time_samples: torch.Tensor,
    signal: torch.Tensor,
    dft_result: np.ndarray,
) -> None:
    """Plot the input signal and the positive-frequency DFT magnitude."""
    sample_count = len(dft_result)
    positive_frequency_count = sample_count // 2
    sample_spacing = SIGNAL_DURATION_SECONDS / sample_count
    frequencies = np.fft.fftfreq(sample_count, d=sample_spacing)[
        :positive_frequency_count
    ]
    magnitude = 2.0 / sample_count * np.abs(
        dft_result[:positive_frequency_count]
    )

    plt.style.use("seaborn-v0_8-darkgrid")
    figure, (signal_axis, spectrum_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
    )

    signal_axis.plot(time_samples.cpu(), signal.cpu(), color="c")
    signal_axis.set_title("Input Square Wave Signal", fontsize=16)
    signal_axis.set_xlabel("Time (s)", fontsize=12)
    signal_axis.set_ylabel("Amplitude", fontsize=12)
    signal_axis.set_xlim(0, SIGNAL_DURATION_SECONDS)
    signal_axis.grid(True)

    spectrum_axis.stem(frequencies, magnitude, basefmt=" ")
    spectrum_axis.set_title(
        "Discrete Fourier Transform (Magnitude Spectrum)",
        fontsize=16,
    )
    spectrum_axis.set_xlabel("Frequency (Hz)", fontsize=12)
    spectrum_axis.set_ylabel("Magnitude", fontsize=12)
    spectrum_axis.set_xlim(0, 50)
    spectrum_axis.grid(True)

    for harmonic_index in range(1, min(20, len(frequencies)), 2):
        frequency = frequencies[harmonic_index]
        spectrum_axis.axvline(
            frequency,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=(
                f"f{harmonic_index}: {harmonic_index} * f0 = "
                f"{frequency:.1f} Hz"
            ),
        )

    spectrum_axis.legend()
    figure.tight_layout()
    plt.show()


def main() -> None:
    device = select_device()
    sample_count = max(SAMPLE_COUNTS)
    time_samples = torch.arange(
        sample_count,
        dtype=torch.float64,
        device=device,
    ) * (SIGNAL_DURATION_SECONDS / sample_count)
    exact_square_wave = square_wave(time_samples)

    plot_fourier_approximations(time_samples, exact_square_wave)

    benchmark_durations = {
        "Naive DFT": [],
        "PyTorch DFT": [],
        "NumPy FFT": [],
    }
    dft_results = {}
    print("--- Transform correctness ---")
    for benchmark_sample_count in SAMPLE_COUNTS:
        benchmark_time_samples = torch.arange(
            benchmark_sample_count,
            dtype=torch.float64,
            device=device,
        ) * (SIGNAL_DURATION_SECONDS / benchmark_sample_count)
        benchmark_signal = square_wave_fourier(
            benchmark_time_samples,
            f0,
            DFT_SIGNAL_HARMONICS,
        )
        dft_result, durations = benchmark_transforms(benchmark_signal, device)
        dft_results[benchmark_sample_count] = dft_result
        for method, duration in durations.items():
            benchmark_durations[method].append(duration)

    plot_benchmark_results(SAMPLE_COUNTS, benchmark_durations)

    signal = square_wave_fourier(time_samples, f0, DFT_SIGNAL_HARMONICS)
    dft_result = dft_results[sample_count]
    plot_dft_spectrum(time_samples, signal, dft_result)


if __name__ == "__main__":
    main()
