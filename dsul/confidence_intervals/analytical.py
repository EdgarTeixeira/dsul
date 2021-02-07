import numpy as np
from scipy import stats

from .base import ConfidenceInterval, IntervalType, _get_alphas


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
    lower, upper = df * sample_var / dist.ppf([upper_alpha, lower_alpha])

    return ConfidenceInterval(lower, upper, alpha)


def stddev_ci(sample, alpha, interval_type=IntervalType.symmetric):
    var_ci = variance_ci(sample, alpha, interval_type)

    return ConfidenceInterval(var_ci.lower ** 0.5,
                              var_ci.upper ** 0.5,
                              alpha)
