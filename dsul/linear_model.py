from typing import NamedTuple
from warnings import warn

import numpy as np
from scipy.optimize import basinhopping, minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_consistent_length, check_scalar, check_X_y
from sklearn.utils.validation import check_is_fitted


class LeastAbsoluteDeviation(BaseEstimator, RegressorMixin):
    def __init__(self, max_iter: int = 100) -> None:
        check_scalar(max_iter, 'max_iter', int, min_val=1)
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None) -> 'LeastAbsoluteDeviation':
        X, y = check_X_y(X, y, y_numeric=True, estimator=self)

        if sample_weight is not None:
            check_consistent_length(y, sample_weight)
            if not np.all(sample_weight > 0):
                raise ValueError(
                    'sample_weight must be an array of nonnegative numbers')

        coefs = np.zeros(X.shape[1] + 1)
        X_1 = np.c_[X, np.ones_like(y)]

        out = minimize(self._cost_function,
                       x0=coefs, args=(X_1, y, sample_weight),
                       method='L-BFGS-B',
                       options={'maxiter': self.max_iter})

        if not out.success:
            warn(f"Optimization failed with message: {out.message}")

        self.coef_ = out.x[:-1]
        self.intercept_ = out.x[-1]

        return self

    def predict(self, X):
        check_is_fitted(self, ['coef_', 'intercept_'])

        return X.dot(self.coef_) + self.intercept_

    def _cost_function(self, theta, X, y, sample_weight):
        predictions = X.dot(theta)
        abs_residuals = np.abs(predictions - y)

        if sample_weight is None:
            return np.mean(abs_residuals)
        return np.mean(abs_residuals * sample_weight)


def __least_median_1d(x):
    x = np.sort(x)
    middle = x.size // 2

    if x.size % 2 != 0:
        deltas = x[middle:] - x[:middle + 1]
    else:
        deltas = x[middle:] - x[:middle]

    idx = np.argmin(deltas)

    return deltas[idx] / 2 + x[idx]


def __lms_slope_cost_function(slope, x, y):
    residuals = y - slope * x
    intercept = __least_median_1d(residuals)

    return np.median(np.abs(residuals - intercept))


class LeastMedianSquaresResults(NamedTuple):
    slope: float
    intercept: float


def least_median_squares(x, y, max_iter: int = 100):
    check_consistent_length(x, y)

    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x.flatten()
        else:
            raise ValueError('x must be 1-dimensional')

    initial_guess = np.cov(x, y)[0, 1] / np.var(x)
    out = basinhopping(__lms_slope_cost_function,
                       x0=initial_guess, niter=max_iter,
                       minimizer_kwargs={'args': (x, y)})

    # TODO: Investigate this
    # if not out.success:
    #     warn(f"Optimization failed with message: {out.message}")

    slope = out.x[0]
    intercept = __least_median_1d(y - slope * x)

    return LeastMedianSquaresResults(slope, intercept)
