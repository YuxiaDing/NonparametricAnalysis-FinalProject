import time
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
from metrics import boundary_rmse, ise, rmse
from smoothers import predict_method, timed_prediction
from table_utils import save_table


def smooth_periodic(x):
    return np.sin(2 * np.pi * x)


def local_spike(x):
    return np.sin(2 * np.pi * x) + 2 * np.exp(-100 * (x - 0.5) ** 2)


def boundary_change(x):
    return x * (1 - x) + 1.5 * np.exp(-80 * (x - 0.08) ** 2)


FUNCTIONS = {
    "Smooth periodic": smooth_periodic,
    "Local spike": local_spike,
    "Boundary change": boundary_change,
}

METHODS = [
    "Linear",
    "Polynomial",
    "Nadaraya-Watson",
    "Local linear",
    "Local quadratic",
    "B-spline",
    "Smoothing spline",
]

MAIN_METHODS = ["Nadaraya-Watson", "Local linear", "B-spline", "Smoothing spline"]


def fit_with_cv(method, x, y, seed):
    candidates = default_candidates(method, BANDWIDTH_GRID, SPLINE_KNOT_GRID, SMOOTHING_GRID, KERNEL_NAME)
    if len(candidates) == 1:
        return candidates[0], np.nan
    return select_by_cv(method, x, y, candidates, n_splits=N_FOLDS, seed=seed)


def run_replicates(seed=RANDOM_SEED):
    """Run the main method-comparison simulation."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, N_GRID)
    rows = []

    for func_name, func in FUNCTIONS.items():
        truth = func(grid)
        for rep in range(N_REPS):
            x = np.sort(rng.uniform(0, 1, N_MAIN))
            y = func(x) + rng.normal(0, 0.2, size=N_MAIN)
            for method in METHODS:
                params, _ = fit_with_cv(method, x, y, seed + rep)
                start = time.perf_counter()
                pred = predict_method(method, x, y, grid, params)
                elapsed = time.perf_counter() - start
                rows.append(
                    {
                        "Function": func_name,
                        "Method": method,
                        "ISE": ise(grid, truth, pred),
                        "RMSE": rmse(truth, pred),
                        "Boundary RMSE": boundary_rmse(grid, truth, pred),
                        "Time": elapsed,
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(TABLES_DIR / "method_simulation_raw.csv", index=False)
    summary = (
        results.groupby(["Function", "Method"], as_index=False)
        .agg({"ISE": "mean", "RMSE": "mean", "Boundary RMSE": "mean", "Time": "mean"})
        .sort_values(["Function", "RMSE"])
    )
    main_summary = summary[summary["Method"].isin(MAIN_METHODS)].copy()
    save_table(
        main_summary[["Function", "Method", "RMSE", "Boundary RMSE"]],
        TABLES_DIR / "tab_method_comparison",
        "Compact simulation comparison across regression functions.",
        "tab:method-comparison",
        float_format="%.3g",
        placement="H",
        small=True,
    )
    save_table(
        summary,
        TABLES_DIR / "tab_method_comparison_full",
        "Supplementary simulation comparison across regression functions.",
        "tab:method-comparison-full",
        float_format="%.4g",
        placement="H",
        small=True,
    )
    return results, summary


def plot_representative(seed=RANDOM_SEED + 37):
    """Plot representative fitted curves for the three regression functions."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 250)
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.2), sharex=True)
    plot_methods = MAIN_METHODS
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

    for ax, (func_name, func) in zip(axes, FUNCTIONS.items()):
        x = np.sort(rng.uniform(0, 1, N_MAIN))
        y = func(x) + rng.normal(0, 0.2, size=N_MAIN)
        ax.scatter(x, y, s=8, color="0.60", alpha=0.32)
        ax.plot(grid, func(grid), color="black", lw=2.0, label="True function")
        for method in plot_methods:
            params, _ = fit_with_cv(method, x, y, seed)
            pred = predict_method(method, x, y, grid, params)
            ax.plot(grid, pred, lw=1.7, color=colors[method], ls=linestyles[method], label=method)
        ax.set_title(func_name)
        ax.set_ylabel("y")
        ax.grid(alpha=0.18)
    axes[-1].set_xlabel("x")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.015), fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = FIGURES_DIR / "fig_method_fits.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_boundary_rmse(summary):
    """Plot boundary RMSE from the method-comparison simulation."""
    functions = list(FUNCTIONS.keys())
    methods = MAIN_METHODS
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), sharey=True)
    for ax, func_name in zip(axes, functions):
        sub = summary[summary["Function"] == func_name].set_index("Method").loc[methods]
        ax.barh(np.arange(len(methods)), sub["Boundary RMSE"], color="#80b1d3", edgecolor="white")
        ax.set_title(func_name)
        ax.set_yticks(np.arange(len(methods)), methods, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlabel("Boundary RMSE")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_boundary_rmse.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def run(seed=RANDOM_SEED):
    results, summary = run_replicates(seed)
    plot_representative(seed + 11)
    plot_boundary_rmse(summary)
    return results, summary


if __name__ == "__main__":
    run()
