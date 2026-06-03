import numpy as np
import matplotlib.pyplot as plt

from config import FIGURES_DIR, N_MAIN, RANDOM_SEED
from metrics import rmse
from smoothers import nw_predict


def true_function(x):
    return np.sin(2 * np.pi * x)


def run(seed=RANDOM_SEED):
    """Create the bandwidth sensitivity figure."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, N_MAIN))
    y = true_function(x) + rng.normal(0, 0.2, size=x.size)
    grid = np.linspace(0, 1, 300)
    truth = true_function(grid)

    h_grid = np.unique(np.r_[np.logspace(-3, 0, 64), 0.024])
    errors = average_bandwidth_rmse(h_grid, seed + 1)
    h_best = h_grid[np.argmin(errors)]
    bandwidths = [0.001, 0.024, 1.0]
    titles = ["Undersmoothing", "Near-optimal smoothing", "Oversmoothing"]

    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.2), sharex=False)
    fit_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    for ax, h, title in zip(fit_axes, bandwidths, titles):
        pred = nw_predict(x, y, grid, h)
        ax.scatter(x, y, s=10, color="0.55", alpha=0.35, label="Observations")
        ax.plot(grid, truth, color="black", lw=1.7, ls="-", label="True function")
        ax.plot(grid, pred, color="#d95f02", lw=2.0, ls="--", label="Nadaraya-Watson")
        ax.set_title(title)
        ax.text(0.04, 0.90, f"h = {h:.3f}", transform=ax.transAxes, fontsize=9)
        ax.set_xlabel("x")
        ax.grid(alpha=0.18)
    fit_axes[0].set_ylabel("y")
    fit_axes[2].set_ylabel("y")

    ax_err = axes[1, 1]
    ax_err.plot(h_grid, errors, color="#377eb8", lw=2.0, marker="o", ms=3)
    ax_err.axvline(h_best, color="#d95f02", lw=1.6, ls="--")
    ax_err.scatter([h_best], [errors.min()], color="#d95f02", zorder=3)
    ax_err.text(h_best + 0.006, errors.min(), f"best h = {h_best:.3f}", fontsize=8, va="bottom")
    ax_err.set_title("Bandwidth grid error")
    ax_err.set_xlabel("Bandwidth")
    ax_err.set_ylabel("Average RMSE")
    ax_err.set_xscale("log")
    ax_err.grid(alpha=0.18)

    handles, labels = fit_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGURES_DIR / "fig_bandwidth_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def average_bandwidth_rmse(h_grid, seed, n_reps=120):
    """Estimate the average RMSE over a bandwidth grid."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 300)
    truth = true_function(grid)
    errors = np.zeros(len(h_grid))
    for _ in range(n_reps):
        x = np.sort(rng.uniform(0, 1, N_MAIN))
        y = true_function(x) + rng.normal(0, 0.2, size=x.size)
        for j, h in enumerate(h_grid):
            pred = nw_predict(x, y, grid, h)
            errors[j] += rmse(truth, pred)
    return errors / n_reps


if __name__ == "__main__":
    run()
