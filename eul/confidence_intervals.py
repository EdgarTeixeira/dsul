from enum import Enum
from typing import NamedTuple

import numpy as np
from joblib import Parallel, delayed
from scipy import stats


class BootstrapCI(Enum):
    percentile = 0
    normal = 1
    bca = 2


class IntervalType(Enum):
    symmetric = 0
    left = 1
    right = 2


class ConfidenceInterval(NamedTuple):
    lower: float
    upper: float
    alpha: float


def mean_ci(sample, alpha, std_dev=None, interval_type=IntervalType.symmetric):
    if std_dev is None:
        std_error = sample.std() / np.sqrt(sample.size)

        df = sample.size - 1
        dist = stats.t(df, loc=0, scale=1)

    else:
        std_error = std_dev / np.sqrt(sample.size)
        dist = stats.norm(loc=0, scale=1)

    lower_alpha, upper_alpha = _get_alphas(alpha, interval_type)
    mean = sample.mean()
    lower, upper = mean + std_error * dist.ppf([lower_alpha, upper_alpha])

    return ConfidenceInterval(lower, upper, alpha)


def proportion_ci(sample, alpha, interval_type=IntervalType.symmetric):
    pass


def variance_ci(sample, alpha, interval_type=IntervalType.symmetric):
    pass


def stddev_ci(sample, alpha, interval_type=IntervalType.symmetric):
    pass


class SimpleBootstrap:
    def __init__(self, sample, stats_function) -> None:
        self.sample = sample
        self.stats_function = stats_function
        self.theta = stats_function(sample)

    def resample(self):
        new_sample = np.random.choice(
            self.sample, size=self.sample.size, replace=True)
        return self.stats_function(new_sample)

    def run(self, iterations: int = 100, n_jobs: int = -1) -> None:
        parallel = Parallel(n_jobs=n_jobs)
        stat_func = delayed(self.resample)

        self.bootstrap_samples_ = parallel(
            stat_func() for it in range(iterations))

    def confidence_interval(self, alpha: float,
                            bootstrap_ci: BootstrapCI = BootstrapCI.percentile,
                            interval_type: IntervalType = IntervalType.symmetric) -> ConfidenceInterval:
        lower_alpha, upper_alpha = _get_alphas(alpha, interval_type)

        if bootstrap_ci == BootstrapCI.percentile:
            ci = np.percentile(self.bootstrap_samples_, [
                               100 * lower_alpha, 100 * upper_alpha])
            if lower_alpha == 0.0:
                ci[0] = -np.inf
            elif upper_alpha == 1.0:
                ci[1] = np.inf

        elif bootstrap_ci == BootstrapCI.normal:
            z_values = stats.norm(0, 1).ppf([lower_alpha, upper_alpha])
            ci = self.theta + self.bootstrap_samples_.std() * z_values

        elif bootstrap_ci == BootstrapCI.bca:
            raise NotImplementedError()
        else:
            raise NotImplementedError("Invalid Bootstrap CI method")

        return ConfidenceInterval(ci[0], ci[1], alpha)


def _get_alphas(alpha, interval_type: IntervalType):
    if interval_type == IntervalType.symmetric:
        return alpha / 2, 1 - alpha / 2
    elif interval_type == IntervalType.left:
        return alpha, 1.0
    elif interval_type == IntervalType.right:
        return 0.0, 1 - alpha
    raise NotImplementedError("Invalid IntervalType")
