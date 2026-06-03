import numpy as np
from sklearn.model_selection import KFold

from metrics import rmse
from smoothers import predict_method


def kfold_score(method, x, y, params, n_splits=5, seed=0):
    """K-fold RMSE for a smoother."""
    x = np.asarray(x)
    y = np.asarray(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.empty_like(y, dtype=float)
    for train_idx, test_idx in kf.split(x):
        preds[test_idx] = predict_method(method, x[train_idx], y[train_idx], x[test_idx], params)
    return rmse(y, preds)


def select_by_cv(method, x, y, candidate_params, n_splits=5, seed=0):
    """Select the parameter dictionary with the smallest K-fold RMSE."""
    best_params = None
    best_score = np.inf
    for params in candidate_params:
        try:
            score = kfold_score(method, x, y, params, n_splits=n_splits, seed=seed)
        except Exception:
            score = np.inf
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, best_score


def default_candidates(method, bandwidth_grid, knot_grid, smoothing_grid, kernel_name="gaussian"):
    """Default parameter candidates for each method."""
    if method == "Nadaraya-Watson":
        return [{"bandwidth": h, "kernel": kernel_name} for h in bandwidth_grid]
    if method == "Local linear":
        return [{"bandwidth": h, "kernel": kernel_name} for h in bandwidth_grid]
    if method == "Local quadratic":
        return [{"bandwidth": h, "kernel": kernel_name} for h in bandwidth_grid]
    if method == "B-spline":
        return [{"n_knots": k} for k in knot_grid]
    if method == "Smoothing spline":
        return [{"smoothing_factor": s} for s in smoothing_grid]
    if method == "Linear":
        return [{}]
    if method == "Polynomial":
        return [{"degree": 3}]
    raise ValueError(f"Unknown method: {method}")
