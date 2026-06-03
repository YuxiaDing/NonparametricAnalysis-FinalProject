import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    BANDWIDTH_GRID,
    DATA_DIR,
    FIGURES_DIR,
    KERNEL_NAME,
    N_FOLDS,
    SMOOTHING_GRID,
    SPLINE_KNOT_GRID,
    TABLES_DIR,
)
from cv_utils import default_candidates, select_by_cv
from metrics import mae, rmse
from smoothers import apply_minmax_scale, minmax_scale, predict_method
from table_utils import save_table


UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATA_SUBDIR = DATA_DIR / "Bike-Sharing-Dataset"
DAY_CSV = DATA_SUBDIR / "day.csv"


METHODS = ["Linear", "Polynomial", "Nadaraya-Watson", "Local linear", "B-spline", "Smoothing spline"]


def ensure_data():
    """Ensure that the daily Bike Sharing data are available."""
    if DAY_CSV.exists():
        return DAY_CSV
    DATA_SUBDIR.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(UCI_ZIP_URL, timeout=30) as response:
            raw = response.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for member in zf.namelist():
                if member.endswith("day.csv"):
                    target = DAY_CSV
                    with zf.open(member) as src, target.open("wb") as dst:
                        dst.write(src.read())
                    return target
    except Exception as exc:
        raise RuntimeError(
            "Bike Sharing data could not be downloaded automatically. "
            "See data/README.md for manual download instructions."
        ) from exc
    raise RuntimeError("day.csv was not found in the downloaded Bike Sharing archive.")


def load_bike_data():
    """Load the daily Bike Sharing data."""
    path = ensure_data()
    df = pd.read_csv(path)
    required = ["cnt", "temp", "atemp", "hum", "windspeed", "workingday"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df.dropna(subset=required).copy()


def fit_with_cv(method, x, y, seed=20260602):
    candidates = default_candidates(method, BANDWIDTH_GRID, SPLINE_KNOT_GRID, SMOOTHING_GRID, KERNEL_NAME)
    if len(candidates) == 1:
        params = candidates[0]
        cv_rmse = np.nan
    else:
        params, cv_rmse = select_by_cv(method, x, y, candidates, n_splits=N_FOLDS, seed=seed)
    return params, cv_rmse


def cross_validated_predictions(method, x, y, params, n_splits=5, seed=20260602):
    """Manual K-fold predictions for real-data evaluation."""
    from sklearn.model_selection import KFold

    x = np.asarray(x)
    y = np.asarray(y)
    preds = np.empty_like(y, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(x):
        preds[test_idx] = predict_method(method, x[train_idx], y[train_idx], x[test_idx], params)
    return preds


def run(seed=20260602):
    """Run the Bike Sharing real-data analysis."""
    df = load_bike_data()
    x_raw = df["temp"].to_numpy(dtype=float)
    y = df["cnt"].to_numpy(dtype=float)
    x, lo, hi = minmax_scale(x_raw)
    grid = np.linspace(0, 1, 250)
    grid_raw = grid * (hi - lo) + lo

    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    axes[0].scatter(df["temp"], df["cnt"], s=12, alpha=0.55, color="#999999")
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Total rentals")
    axes[0].set_title("Rentals versus temperature")
    for value, label, color in [(0, "Non-working day", "#d95f02"), (1, "Working day", "#377eb8")]:
        sub = df[df["workingday"] == value]
        axes[1].scatter(sub["temp"], sub["cnt"], s=10, alpha=0.45, label=label, color=color)
    axes[1].set_xlabel("Temperature")
    axes[1].set_title("Working-day groups")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].scatter(df["hum"], df["cnt"], s=12, alpha=0.55, color="#999999")
    axes[2].set_xlabel("Humidity")
    axes[2].set_title("Rentals versus humidity")
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.suptitle("Exploratory plots for the Bike Sharing data", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_bike_exploratory.pdf", bbox_inches="tight")
    plt.close(fig)

    rows = []
    predictions = {}
    for method in METHODS:
        params, selected_score = fit_with_cv(method, x, y, seed=seed)
        fit = predict_method(method, x, y, grid, params)
        cv_pred = cross_validated_predictions(method, x, y, params, n_splits=N_FOLDS, seed=seed)
        predictions[method] = fit
        rows.append(
            {
                "Method": method,
                "CV RMSE": rmse(y, cv_pred),
                "CV MAE": mae(y, cv_pred),
                "Selected parameter": format_params(params),
            }
        )

    eval_table = pd.DataFrame(rows).sort_values("CV RMSE")
    save_table(
        eval_table,
        TABLES_DIR / "tab_bike_cv",
        "Cross-validated errors for the Bike Sharing analysis.",
        "tab:bike-cv",
        float_format="%.3f",
        placement="H",
        small=True,
    )

    colors = {
        "Linear": "#7570b3",
        "Polynomial": "#1b9e77",
        "Nadaraya-Watson": "#d95f02",
        "Local linear": "#e7298a",
        "B-spline": "#66a61e",
        "Smoothing spline": "#377eb8",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(x_raw, y, s=12, color="0.70", alpha=0.55, label="Daily observations")
    for method in METHODS:
        ax.plot(grid_raw, predictions[method], lw=1.7, color=colors[method], label=method)
    ax.set_xlabel("Normalized temperature")
    ax.set_ylabel("Total rentals")
    ax.set_title("Nonparametric smoothing of total rentals by temperature")
    ax.grid(alpha=0.18)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_bike_fits.pdf", bbox_inches="tight")
    plt.close(fig)

    best_method = eval_table.iloc[0]["Method"]
    best_params, _ = fit_with_cv(best_method, x, y, seed=seed)
    fitted_train = predict_method(best_method, x, y, x, best_params)
    residuals = y - fitted_train
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    axes[0].scatter(x_raw, residuals, s=12, color="#999999", alpha=0.65)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals versus temperature ({best_method})")
    axes[1].scatter(fitted_train, residuals, s=12, color="#999999", alpha=0.65)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xlabel("Fitted value")
    axes[1].set_title("Residuals versus fitted values")
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_bike_residuals.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for value, label, color in [(0, "Non-working day", "#d95f02"), (1, "Working day", "#377eb8")]:
        sub = df[df["workingday"] == value]
        xs, sub_lo, sub_hi = minmax_scale(sub["temp"].to_numpy(dtype=float))
        ys = sub["cnt"].to_numpy(dtype=float)
        params, _ = fit_with_cv("Local linear", xs, ys, seed=seed)
        pred = predict_method("Local linear", xs, ys, grid, params)
        ax.scatter(sub["temp"], ys, s=9, alpha=0.25, color=color)
        ax.plot(grid * (sub_hi - sub_lo) + sub_lo, pred, color=color, lw=2, label=label)
    ax.set_xlabel("Normalized temperature")
    ax.set_ylabel("Total rentals")
    ax.set_title("Local linear fits by working-day status")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_bike_workingday.pdf", bbox_inches="tight")
    plt.close(fig)

    return eval_table


def format_params(params):
    if not params:
        return "None"
    parts = []
    for key, value in params.items():
        if key == "kernel":
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.3g}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "None"


if __name__ == "__main__":
    run()
