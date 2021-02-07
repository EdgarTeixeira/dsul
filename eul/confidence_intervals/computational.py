import numpy as np
from joblib import Parallel, delayed
from scipy import stats

from .base import BootstrapCI, ConfidenceInterval, IntervalType, _get_alphas


class SimpleJackknife:
    def __init__(self, sample, stats_function) -> None:
        self.sample = np.asarray(sample)
        self.stats_function = stats_function
        self.theta = stats_function(sample)

    def run(self, n_jobs: int = -1):
        parallel = Parallel(n_jobs=n_jobs)
        stat_func = delayed(self.stats_function)

        idx = np.arange(self.sample.size)

        jackknife_values = parallel(
            stat_func(self.sample[idx != i])
            for i in range(self.sample.size)
        )

        self.jackknife_samples_ = np.asarray(jackknife_values)

    def bias(self):
        delta = self.jackknife_samples_.mean() - self.theta
        return (self.sample.size - 1) * delta

    def bias_corrected(self):
        A = self.sample.size * self.theta
        B = (self.sample.size - 1) * self.jackknife_samples_.mean()

        return A - B

    def variance(self):
        var = self.jackknife_samples_.var()
        return var / (self.sample.size - 1)


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

        self.bootstrap_samples_ = np.asarray(
            parallel(stat_func() for it in range(iterations)))

    def bias(self):
        return self.bootstrap_samples_.mean() - self.theta

    def bias_corrected(self):
        return 2 * self.theta - self.bootstrap_samples_.mean()

    def variance(self):
        return self.bootstrap_samples_.var(ddof=1)

    def confidence_interval(self, alpha: float,
                            bootstrap_ci: BootstrapCI = BootstrapCI.percentile,
                            interval_type: IntervalType = IntervalType.symmetric,
                            n_jobs: int = -1) -> ConfidenceInterval:
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
            ci = self.theta + self.bootstrap_samples_.std(ddof=1) * z_values

        elif bootstrap_ci == BootstrapCI.bca:
            std_norm = stats.norm(0, 1)
            z_lower, z_upper = std_norm.ppf([lower_alpha, upper_alpha])

            # Bias Correction Factor
            z0 = std_norm.ppf((self.bootstrap_samples_ <= self.theta).mean())

            # Acceleration Factor,
            jack = SimpleJackknife(self.sample, self.stats_function)
            jack.run(n_jobs=n_jobs)
            jackknife_values = jack.jackknife_samples_
            jack_mean = jackknife_values.mean()

            num = np.power(jack_mean - jackknife_values, 3).sum()
            den = 6 * np.power(np.square(jack_mean -
                                         jackknife_values).sum(), 3/2)
            a = num / den

            # Corrected percentiles
            if lower_alpha > 0.0:
                corrected_lower = z0 + (z0 + z_lower) / \
                    (1 - a * (z0 + z_lower))
                corrected_lower = std_norm.cdf(corrected_lower)
            else:
                corrected_lower = lower_alpha

            if upper_alpha < 1.0:
                corrected_upper = z0 + (z0 + z_upper) / \
                    (1 - a * (z0 + z_upper))
                corrected_upper = std_norm.cdf(corrected_upper)
            else:
                corrected_upper = upper_alpha

            ci = np.percentile(self.bootstrap_samples_, [
                               100 * corrected_lower, 100 * corrected_upper])
            if lower_alpha == 0.0:
                ci[0] = -np.inf
            if upper_alpha == 1.0:
                ci[1] = np.inf
        else:
            raise NotImplementedError("Invalid Bootstrap CI method")

        return ConfidenceInterval(ci[0], ci[1], alpha)
