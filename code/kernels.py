import numpy as np


def gaussian_kernel(u):
    """Gaussian kernel density evaluated elementwise."""
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def epanechnikov_kernel(u):
    """Epanechnikov kernel density evaluated elementwise."""
    out = 0.75 * (1.0 - u * u)
    return np.where(np.abs(u) <= 1.0, out, 0.0)


def get_kernel(name):
    """Return a kernel function by name."""
    if name == "gaussian":
        return gaussian_kernel
    if name == "epanechnikov":
        return epanechnikov_kernel
    raise ValueError(f"Unknown kernel: {name}")
