import time
import warnings
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

from kernels import get_kernel


def minmax_scale(x):
    """Scale a one-dimensional array to [0, 1]."""
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if hi == lo:
        return np.zeros_like(x), lo, hi
    return (x - lo) / (hi - lo), lo, hi


def apply_minmax_scale(x, lo, hi):
    """Apply a previously fitted min-max scaling."""
    x = np.asarray(x, dtype=float)
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def linear_predict(x_train, y_train, x_eval):
    """Linear regression prediction for one predictor."""
    model = LinearRegression()
    model.fit(np.asarray(x_train).reshape(-1, 1), np.asarray(y_train))
    return model.predict(np.asarray(x_eval).reshape(-1, 1))


def polynomial_predict(x_train, y_train, x_eval, degree=3):
    """Polynomial regression prediction for one predictor."""
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=True), LinearRegression())
    model.fit(np.asarray(x_train).reshape(-1, 1), np.asarray(y_train))
    return model.predict(np.asarray(x_eval).reshape(-1, 1))


def nw_predict(x_train, y_train, x_eval, bandwidth, kernel_name="gaussian"):
    """Nadaraya-Watson kernel regression."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    kernel = get_kernel(kernel_name)
    u = (x_eval[:, None] - x_train[None, :]) / bandwidth
    weights = kernel(u)
    denom = weights.sum(axis=1)
    denom = np.where(denom <= 1e-14, 1e-14, denom)
    return (weights @ y_train) / denom


def local_poly_predict(x_train, y_train, x_eval, bandwidth, degree=1, kernel_name="gaussian", ridge=1e-10):
    """Local polynomial regression prediction."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    kernel = get_kernel(kernel_name)
    preds = np.empty_like(x_eval, dtype=float)
    eye = np.eye(degree + 1)
    eye[0, 0] = 0.0

    for k, x0 in enumerate(x_eval):
        dx = x_train - x0
        w = kernel(dx / bandwidth)
        if np.sum(w) <= 1e-12:
            preds[k] = np.mean(y_train)
            continue
        design = np.column_stack([dx ** j for j in range(degree + 1)])
        xtw = design.T * w
        lhs = xtw @ design + ridge * eye
        rhs = xtw @ y_train
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        preds[k] = beta[0]
    return preds


def bspline_predict(x_train, y_train, x_eval, n_knots=8, degree=3, alpha=1e-6):
    """B-spline regression using sklearn's SplineTransformer and ridge stabilization."""
    n_knots = max(3, int(n_knots))
    model = make_pipeline(
        SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False),
        Ridge(alpha=alpha),
    )
    model.fit(np.asarray(x_train).reshape(-1, 1), np.asarray(y_train))
    return model.predict(np.asarray(x_eval).reshape(-1, 1))


def smoothing_spline_predict(x_train, y_train, x_eval, smoothing_factor=1.0):
    """Cubic smoothing spline prediction.

    The smoothing_factor is a multiplier for n * Var(y) used as scipy's s value.
    """
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    order = np.argsort(x_train)
    xs = x_train[order]
    ys = y_train[order]
    s_value = float(smoothing_factor) * len(xs) * max(np.var(ys), 1e-8)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        spline = UnivariateSpline(xs, ys, k=3, s=s_value)
    return spline(x_eval)


def predict_method(method, x_train, y_train, x_eval, params=None):
    """Dispatch predictions for a named method."""
    params = params or {}
    if method == "Linear":
        return linear_predict(x_train, y_train, x_eval)
    if method == "Polynomial":
        return polynomial_predict(x_train, y_train, x_eval, degree=params.get("degree", 3))
    if method == "Nadaraya-Watson":
        return nw_predict(x_train, y_train, x_eval, params["bandwidth"], params.get("kernel", "gaussian"))
    if method == "Local linear":
        return local_poly_predict(x_train, y_train, x_eval, params["bandwidth"], degree=1, kernel_name=params.get("kernel", "gaussian"))
    if method == "Local quadratic":
        return local_poly_predict(x_train, y_train, x_eval, params["bandwidth"], degree=2, kernel_name=params.get("kernel", "gaussian"))
    if method == "B-spline":
        return bspline_predict(x_train, y_train, x_eval, n_knots=params.get("n_knots", 8))
    if method == "Smoothing spline":
        return smoothing_spline_predict(x_train, y_train, x_eval, smoothing_factor=params.get("smoothing_factor", 1.0))
    raise ValueError(f"Unknown method: {method}")


def timed_prediction(method, x_train, y_train, x_eval, params=None):
    """Return predictions and elapsed time."""
    start = time.perf_counter()
    pred = predict_method(method, x_train, y_train, x_eval, params)
    elapsed = time.perf_counter() - start
    return pred, elapsed
