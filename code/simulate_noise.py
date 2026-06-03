import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    BANDWIDTH_GRID,
    FIGURES_DIR,
    KERNEL_NAME,
    N_FOLDS,
    N_GRID,
    N_MAIN,
    N_REPS,
    RANDOM_SEED,
    SMOOTHING_GRID,
    SPLINE_KNOT_GRID,
    TABLES_DIR,
)
from cv_utils import default_candidates, select_by_cv
from metrics import median_absolute_error, rmse
from smoothers import predict_method
from table_utils import save_table


def true_function(x):
    return np.sin(2 * np.pi * x) + x


def sample_noise(rng, x, noise_type):
    if noise_type == "Normal":
        return rng.normal(0, 0.2, size=x.size)
    if noise_type == "Heavy-tailed":
        return 0.2 * rng.standard_t(df=3, size=x.size) / np.sqrt(3)
    if noise_type == "Heteroscedastic":
        sigma = 0.1 + 0.4 * x
        return rng.normal(0, sigma)
    if noise_type == "Contaminated":
        clean = rng.normal(0, 0.2, size=x.size)
        outlier = rng.normal(0, 1.0, size=x.size)
        mask = rng.uniform(size=x.size) < 0.05
        return np.where(mask, outlier, clean)
    raise ValueError(noise_type)


METHODS = ["Nadaraya-Watson", "Local linear", "B-spline", "Smoothing spline"]
METHOD_LABELS = ["NW", "Local\nlinear", "B-spline", "Smoothing\nspline"]
NOISE_TYPES = ["Normal", "Heavy-tailed", "Heteroscedastic", "Contaminated"]


def fit_with_cv(method, x, y, seed):
    candidates = default_candidates(method, BANDWIDTH_GRID, SPLINE_KNOT_GRID, SMOOTHING_GRID, KERNEL_NAME)
    return select_by_cv(method, x, y, candidates, n_splits=N_FOLDS, seed=seed)


def run_replicates(seed=RANDOM_SEED + 1000):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, N_GRID)
    truth = true_function(grid)
    rows = []

    for noise_type in NOISE_TYPES:
        for rep in range(N_REPS):
            x = np.sort(rng.uniform(0, 1, N_MAIN))
            y = true_function(x) + sample_noise(rng, x, noise_type)
            for method in METHODS:
                params, _ = fit_with_cv(method, x, y, seed + rep)
                pred = predict_method(method, x, y, grid, params)
                rows.append(
                    {
                        "Noise": noise_type,
                        "Method": method,
                        "RMSE": rmse(truth, pred),
                        "Median absolute error": median_absolute_error(truth, pred),
                    }
                )
    results = pd.DataFrame(rows)
    results.to_csv(TABLES_DIR / "noise_simulation_raw.csv", index=False)
    summary = (
        results.groupby(["Noise", "Method"], as_index=False)
        .agg({"RMSE": "mean", "Median absolute error": "mean"})
        .sort_values(["Noise", "RMSE"])
    )
    rmse_table = summary.pivot(index="Noise", columns="Method", values="RMSE").reset_index()
    rmse_table = rmse_table[["Noise"] + METHODS]
    save_table(
        rmse_table,
        TABLES_DIR / "tab_noise_comparison",
        "Average RMSE under different noise structures.",
        "tab:noise-comparison",
        float_format="%.3g",
        placement="H",
        small=True,
    )
    save_table(
        summary,
        TABLES_DIR / "tab_noise_comparison_full",
        "Supplementary simulation comparison under different noise structures.",
        "tab:noise-comparison-full",
        float_format="%.4g",
        placement="H",
        small=True,
    )
    return results, summary


def plot_representative(seed=RANDOM_SEED + 1200):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 250)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.9), sharex=True, sharey=True)
    colors = {
        "Nadaraya-Watson": "#d95f02",
        "Local linear": "#1b9e77",
        "B-spline": "#66a61e",
        "Smoothing spline": "#377eb8",
    }
    linestyles = {
        "Nadaraya-Watson": "--",
        "Local linear": "-.",
        "B-spline": ":",
        "Smoothing spline": "-",
    }
    for ax, noise_type in zip(axes.ravel(), NOISE_TYPES):
        x = np.sort(rng.uniform(0, 1, N_MAIN))
        y = true_function(x) + sample_noise(rng, x, noise_type)
        ax.scatter(x, y, s=8, color="0.60", alpha=0.30)
        ax.plot(grid, true_function(grid), color="black", lw=2.0, label="True function")
        for method in METHODS:
            params, _ = fit_with_cv(method, x, y, seed)
            pred = predict_method(method, x, y, grid, params)
            ax.plot(grid, pred, lw=1.6, color=colors[method], ls=linestyles[method], label=method)
        ax.set_title(noise_type)
        ax.grid(alpha=0.18)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = FIGURES_DIR / "fig_noise_fits.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_error_boxplots(results):
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.0), sharey=True)
    for ax, noise_type in zip(axes, NOISE_TYPES):
        sub = results[results["Noise"] == noise_type]
        data = [sub[sub["Method"] == method]["RMSE"].values for method in METHODS]
        parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#9ecae1")
            body.set_edgecolor("#377eb8")
            body.set_alpha(0.55)
        parts["cmedians"].set_color("#d95f02")
        ax.boxplot(
            data,
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "0.35", "linewidth": 0.8},
            medianprops={"color": "#d95f02", "linewidth": 1.0},
            whiskerprops={"color": "0.35", "linewidth": 0.8},
            capprops={"color": "0.35", "linewidth": 0.8},
        )
        ax.set_xticks(np.arange(1, len(METHODS) + 1))
        ax.set_xticklabels(METHOD_LABELS)
        ax.set_title(noise_type)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("RMSE")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_noise_boxplots.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def run(seed=RANDOM_SEED + 1000):
    results, summary = run_replicates(seed)
    plot_representative(seed + 11)
    plot_error_boxplots(results)
    return results, summary


if __name__ == "__main__":
    run()
