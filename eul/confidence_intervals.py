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
    sample = np.asarray(sample)

    if std_dev is None:
        std_error = sample.std(ddof=1) / np.sqrt(sample.size)

        df = sample.size - 1
        dist = stats.t(df, loc=0, scale=1)

    else:
        std_error = std_dev / np.sqrt(sample.size)
        dist = stats.norm(loc=0, scale=1)

    lower_alpha, upper_alpha = _get_alphas(alpha, interval_type)
    mean = sample.mean()
    lower, upper = mean + std_error * dist.ppf([lower_alpha, upper_alpha])

    return ConfidenceInterval(lower, upper, alpha)


def proportion_ci(sample_or_proportion, alpha, interval_type=IntervalType.symmetric):
    sample = np.asarray(sample_or_proportion)
    p = sample.mean()
    std_error = np.sqrt(p * (1 - p) / sample.size)

    lower_alpha, upper_alpha = _get_alphas(alpha, interval_type)
    dist = stats.norm(0, 1)
    lower, upper = p + std_error * dist.ppf([lower_alpha, upper_alpha])

    lower = max(lower, 0.0)
    upper = min(upper, 1.0)

    return ConfidenceInterval(lower, upper, alpha)


def variance_ci(sample, alpha, interval_type=IntervalType.symmetric):
    sample = np.asarray(sample)
    sample_var = sample.var(ddof=1)
    df = sample.size - 1

    lower_alpha, upper_alpha = _get_alphas(alpha, interval_type)
    dist = stats.chi2(df)
    lower, upper = df * sample_var / dist.ppf([lower_alpha, upper_alpha])

    return ConfidenceInterval(lower, upper, alpha)


def stddev_ci(sample, alpha, interval_type=IntervalType.symmetric):
    var_ci = variance_ci(sample, alpha, interval_type)

    return ConfidenceInterval(var_ci.lower ** 0.5,
                              var_ci.upper ** 0.5,
                              alpha)


def get_jackknife_distribution(sample, stats_function, n_jobs=-1):
    sample = np.asarray(sample)
    idx = np.arange(sample.size)

    parallel = Parallel(n_jobs=n_jobs)
    stats_func = delayed(stats_function)

    jackknife_values = parallel(
        stats_func(sample[idx != i])
        for i in range(sample.size)
    )

    return np.asarray(jackknife_values)


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
            std_norm = stats.norm(0, 1)
            z_lower, z_upper = std_norm.ppf([lower_alpha, upper_alpha])

            # Bias Correction Factor
            z0 = std_norm.ppf((self.bootstrap_samples_ < self.theta).mean())

            # Acceleration Factor
            jackknife_values = get_jackknife_distribution(self.sample, self.stats_function)
            jack_mean = jackknife_values.mean()

            num = np.power(jack_mean - jackknife_values, 3).sum()
            den = 6 * np.power(np.square(jack_mean - jackknife_values).sum(), 3/2)
            a = num / den

            # Corrected percentiles
            if lower_alpha > 0.0:
                corrected_lower = z0 + (z0 + z_lower) / (1 - a * (z0 + z_lower))
                corrected_lower = std_norm.cdf(corrected_lower)
            else:
                corrected_lower = lower_alpha

            if upper_alpha < 1.0:
                corrected_upper = z0 + (z0 + z_upper) / (1 - a * (z0 + z_upper))
                corrected_upper = std_norm.cdf(corrected_upper)
            else:
                corrected_upper = upper_alpha

            ci = np.percentile(self.bootstrap_samples_, [100 * corrected_lower, 100 * corrected_upper])
            if lower_alpha == 0.0:
                ci[0] = -np.inf
            if upper_alpha == 1.0:
                ci[1] = np.inf

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
