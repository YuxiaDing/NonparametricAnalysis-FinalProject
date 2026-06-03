import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import BANDWIDTH_GRID, FIGURES_DIR, KERNEL_NAME, N_GRID, N_MAIN, N_REPS, RANDOM_SEED, TABLES_DIR
from metrics import boundary_rmse
from smoothers import local_poly_predict, nw_predict
from table_utils import save_table


def true_function(x):
    return x


def run(seed=RANDOM_SEED + 2000):
    """Run the supplementary boundary-bias experiment."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, N_GRID)
    truth = true_function(grid)
    h = 0.12
    methods = ["Nadaraya-Watson", "Local linear"]
    preds = {m: [] for m in methods}
    rows = []

    for rep in range(N_REPS):
        x = np.sort(rng.uniform(0, 1, N_MAIN))
        y = true_function(x) + rng.normal(0, 0.1, size=N_MAIN)
        nw = nw_predict(x, y, grid, h, KERNEL_NAME)
        ll = local_poly_predict(x, y, grid, h, degree=1, kernel_name=KERNEL_NAME)
        preds["Nadaraya-Watson"].append(nw)
        preds["Local linear"].append(ll)
        rows.append({"Method": "Nadaraya-Watson", "Boundary RMSE": boundary_rmse(grid, truth, nw)})
        rows.append({"Method": "Local linear", "Boundary RMSE": boundary_rmse(grid, truth, ll)})

    results = pd.DataFrame(rows)
    summary = results.groupby("Method", as_index=False).agg({"Boundary RMSE": "mean"})
    save_table(
        summary,
        TABLES_DIR / "tab_boundary_experiment",
        "Supplementary boundary-bias experiment.",
        "tab:boundary-experiment",
        float_format="%.3g",
        placement="H",
        small=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharey=True)
    avg_nw = np.mean(np.vstack(preds["Nadaraya-Watson"]), axis=0)
    avg_ll = np.mean(np.vstack(preds["Local linear"]), axis=0)
    for ax, mask, title in [
        (axes[0], grid <= 0.2, "Left boundary"),
        (axes[1], grid >= 0.8, "Right boundary"),
    ]:
        ax.plot(grid[mask], truth[mask], color="black", lw=2, label="True function")
        ax.plot(grid[mask], avg_nw[mask], color="#d95f02", lw=2, label="Nadaraya-Watson")
        ax.plot(grid[mask], avg_ll[mask], color="#377eb8", lw=2, label="Local linear")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Average fitted value")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Boundary bias: local constant versus local linear smoothing", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_boundary_bias.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(grid, avg_nw - truth, color="#d95f02", lw=2, label="Nadaraya-Watson")
    ax.plot(grid, avg_ll - truth, color="#377eb8", lw=2, label="Local linear")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("x")
    ax.set_ylabel("Average bias")
    ax.set_title("Average bias over the design interval")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_boundary_bias_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    return results, summary


if __name__ == "__main__":
    run()
