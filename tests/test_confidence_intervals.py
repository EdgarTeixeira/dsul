import eul.confidence_intervals as ci
import numpy as np
from scipy import stats


class TestAnalyticalMethods:
    def test_mean_ci_with_known_stddev(self):
        population_dist = stats.norm(0, 1)
        successes = 0
        trials = 1000

        for i in range(trials):
            sample = population_dist.rvs(30)
            results = ci.mean_ci(sample, alpha=0.2, std_dev=1)
            successes += int(results.contains(0.0))

        coverage = successes / trials
        posterior = stats.beta(successes + 1, trials - successes + 1)
        is_credible = posterior.ppf(0.05) <= coverage <= posterior.ppf(0.95)
        assert is_credible, self.error_msg(coverage, 0.8)

    def test_mean_ci_with_unknown_stddev(self):
        population_dist = stats.norm(0, 1)
        successes = 0
        trials = 1000

        for i in range(trials):
            sample = population_dist.rvs(30)
            results = ci.mean_ci(sample, alpha=0.2)
            successes += int(results.contains(0.0))

        coverage = successes / trials
        posterior = stats.beta(successes + 1, trials - successes + 1)
        is_credible = posterior.ppf(0.05) <= coverage <= posterior.ppf(0.95)
        assert is_credible, self.error_msg(coverage, 0.8)

    def test_variance_ci(self):
        population_dist = stats.norm(0, 1)
        successes = 0
        trials = 1000

        for i in range(trials):
            sample = population_dist.rvs(30)
            results = ci.variance_ci(sample, alpha=0.2)
            successes += int(results.contains(1.0))

        coverage = successes / trials
        posterior = stats.beta(successes + 1, trials - successes + 1)
        is_credible = posterior.ppf(0.05) <= coverage <= posterior.ppf(0.95)
        assert is_credible, self.error_msg(coverage, 0.8)

    def test_standard_deviation_ci(self):
        population_dist = stats.norm(0, 1)
        successes = 0
        trials = 1000

        for i in range(trials):
            sample = population_dist.rvs(30)
            results = ci.stddev_ci(sample, alpha=0.2)
            successes += int(results.contains(1.0))

        coverage = successes / trials
        posterior = stats.beta(successes + 1, trials - successes + 1)
        is_credible = posterior.ppf(0.05) <= coverage <= posterior.ppf(0.95)
        assert is_credible, self.error_msg(coverage, 0.8)

    def error_msg(self, observed, expected):
        return f"Observed Coverage: {observed}\nExpected Coverage: {expected}"


class TestSimpleBootstrap:
    def test_symmetric_percentile(self):
        sample, mean, std_error = self.generate_dataset()

        bootstrap = ci.SimpleBootstrap(sample, np.mean)
        bootstrap.run(iterations=1000)

        boots_ci = bootstrap.confidence_interval(alpha=0.1)
        theo_ci = ci.mean_ci(sample, alpha=0.1, std_dev=1)

        lower_dist = abs(boots_ci.lower - theo_ci.lower)
        assert lower_dist < 0.01, self.error_msg(boots_ci.lower, theo_ci.lower)

        upper_dist = abs(boots_ci.upper - theo_ci.upper)
        assert upper_dist < 0.01, self.error_msg(boots_ci.upper, theo_ci.upper)

    def test_left_percentile(self):
        sample, mean, std_error = self.generate_dataset()

        bootstrap = ci.SimpleBootstrap(sample, np.mean)
        bootstrap.run(iterations=1000)

        boots_ci = bootstrap.confidence_interval(
            alpha=0.1, interval_type=ci.IntervalType.left)

        theo_ci = ci.mean_ci(
            sample, alpha=0.1, std_dev=1, interval_type=ci.IntervalType.left)

        assert boots_ci.upper == theo_ci.upper, "Upper confidence should be infinite for a one-sided left CI"

        lower_dist = abs(boots_ci.lower - theo_ci.lower)
        assert lower_dist < 0.01, self.error_msg(boots_ci.lower, theo_ci.lower)

    def test_right_percentile(self):
        sample, mean, std_error = self.generate_dataset()

        bootstrap = ci.SimpleBootstrap(sample, np.mean)
        bootstrap.run(iterations=1000)

        boots_ci = bootstrap.confidence_interval(
            alpha=0.1, interval_type=ci.IntervalType.right)
        theo_ci = ci.mean_ci(
            sample, alpha=0.1, std_dev=1, interval_type=ci.IntervalType.right)

        assert boots_ci.lower == theo_ci.lower, "Lower confidence should be -infinity for a one-sided right CI"

        upper_dist = abs(boots_ci.upper - theo_ci.upper)
        assert upper_dist < 0.01, self.error_msg(boots_ci.upper, theo_ci.upper)

    def generate_dataset(self, sample_size=1000):
        sample = np.random.normal(0, 1, size=sample_size)
        mean = sample.mean()
        std_error = 1 / np.sqrt(sample.size)

        return sample, mean, std_error

    def error_msg(boots, theoretical):
        return f"Bootstrap upper: {boots}/nTheoretical upper: {theoretical}"
