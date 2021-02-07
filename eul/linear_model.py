from warnings import warn

import numpy as np
from scipy.optimize import minimize
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
