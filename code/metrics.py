import numpy as np


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def ise(x_grid, truth, estimate):
    """Integrated squared error approximated by a grid average."""
    return float(np.mean((np.asarray(truth) - np.asarray(estimate)) ** 2))


def boundary_rmse(x_grid, truth, estimate, width=0.1):
    """RMSE on two boundary regions of [0, 1]."""
    x_grid = np.asarray(x_grid)
    mask = (x_grid <= width) | (x_grid >= 1.0 - width)
    return rmse(np.asarray(truth)[mask], np.asarray(estimate)[mask])


def median_absolute_error(y_true, y_pred):
    """Median absolute error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.median(np.abs(y_true - y_pred)))
