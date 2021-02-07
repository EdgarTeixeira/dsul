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

    def test_proportion_ci(self):
        population_dist = stats.bernoulli(0.7)
        successes = 0
        trials = 1000

        for i in range(trials // 2):
            sample = population_dist.rvs(30)
            results = ci.proportion_ci(sample, alpha=0.2)
            successes += int(results.contains(1.0))

        for i in range(trials - trials // 2):
            sample = population_dist.rvs(30)
            results = ci.proportion_ci(sample.mean(), alpha=0.2)
            successes += int(results.contains(1.0))

        coverage = successes / trials
        posterior = stats.beta(successes + 1, trials - successes + 1)
        is_credible = posterior.ppf(0.05) <= coverage <= posterior.ppf(0.95)
        assert is_credible, self.error_msg(coverage, 0.8)

    def error_msg(self, observed, expected):
        return f"Observed Coverage: {observed}\nExpected Coverage: {expected}"


def test_jackknife_distribution():
    sample = [1, 2, 3]
    jack = ci.SimpleJackknife(sample, np.mean)
    jack.run(n_jobs=1)
    jackknife_values = jack.jackknife_samples_
    ground_truth = np.asarray([(2 + 3)/2, (1 + 3)/2, (1 + 2)/2])

    assert np.allclose(jackknife_values, ground_truth)


class TestSimpleBootstrap:
    def test_symmetric_ci(self):
        sample = np.random.normal(0, 1, size=1000)
        theo_ci = ci.mean_ci(sample, alpha=0.1, std_dev=1)

        bootstrap = ci.SimpleBootstrap(sample, np.mean)
        bootstrap.run(iterations=1000)

        # Test Percentile CI
        boots_ci = bootstrap.confidence_interval(alpha=0.1)

        lower_dist = abs(boots_ci.lower - theo_ci.lower)
        assert lower_dist < 0.01, self.error_msg(boots_ci.lower, theo_ci.lower)

        upper_dist = abs(boots_ci.upper - theo_ci.upper)
        assert upper_dist < 0.01, self.error_msg(boots_ci.upper, theo_ci.upper)

        # Test Normal CI
        boots_ci = bootstrap.confidence_interval(
            alpha=0.1, bootstrap_ci=ci.BootstrapCI.normal)

        lower_dist = abs(boots_ci.lower - theo_ci.lower)
        assert lower_dist < 0.01, self.error_msg(boots_ci.lower, theo_ci.lower)

        upper_dist = abs(boots_ci.upper - theo_ci.upper)
        assert upper_dist < 0.01, self.error_msg(boots_ci.upper, theo_ci.upper)

        # Test BCA CI
        boots_ci = bootstrap.confidence_interval(
            alpha=0.1, bootstrap_ci=ci.BootstrapCI.bca)

        lower_dist = abs(boots_ci.lower - theo_ci.lower)
        assert lower_dist < 0.01, self.error_msg(boots_ci.lower, theo_ci.lower)

        upper_dist = abs(boots_ci.upper - theo_ci.upper)
        assert upper_dist < 0.01, self.error_msg(boots_ci.upper, theo_ci.upper)

    def test_left_ci(self):
        sample = np.random.normal(0, 1, size=1000)
        theo_ci = ci.mean_ci(
            sample, alpha=0.1, std_dev=1, interval_type=ci.IntervalType.left)

        bootstrap = ci.SimpleBootstrap(sample, np.mean)
        bootstrap.run(iterations=1000)

        # Test Percentile CI
        boots_ci = bootstrap.confidence_interval(
            alpha=0.1, interval_type=ci.IntervalType.left)

        assert boots_ci.upper == theo_ci.upper, "Upper confidence should be infinite for a one-sided left CI"

        lower_dist = abs(boots_ci.lower - theo_ci.lower)
        assert lower_dist < 0.01, self.error_msg(boots_ci.lower, theo_ci.lower)

        # Test Normal CI
        boots_ci = bootstrap.confidence_interval(
            alpha=0.1, interval_type=ci.IntervalType.left, bootstrap_ci=ci.BootstrapCI.normal)

        assert boots_ci.upper == theo_ci.upper, "Upper confidence should be infinite for a one-sided left CI"

        lower_dist = abs(boots_ci.lower - theo_ci.lower)
        assert lower_dist < 0.01, self.error_msg(boots_ci.lower, theo_ci.lower)

        # Test BCA CI
        boots_ci = bootstrap.confidence_interval(
            alpha=0.1, interval_type=ci.IntervalType.left, bootstrap_ci=ci.BootstrapCI.bca)

        assert boots_ci.upper == theo_ci.upper, "Upper confidence should be infinite for a one-sided left CI"

        lower_dist = abs(boots_ci.lower - theo_ci.lower)
        assert lower_dist < 0.01, self.error_msg(boots_ci.lower, theo_ci.lower)

    def test_right_ci(self):
        sample = np.random.normal(0, 1, size=1000)
        theo_ci = ci.mean_ci(
            sample, alpha=0.1, std_dev=1, interval_type=ci.IntervalType.right)

        bootstrap = ci.SimpleBootstrap(sample, np.mean)
        bootstrap.run(iterations=1000)

        # Test Percentile CI
        boots_ci = bootstrap.confidence_interval(
            alpha=0.1, interval_type=ci.IntervalType.right)

        assert boots_ci.lower == theo_ci.lower, "Lower confidence should be -infinity for a one-sided right CI"

        upper_dist = abs(boots_ci.upper - theo_ci.upper)
        assert upper_dist < 0.01, self.error_msg(boots_ci.upper, theo_ci.upper)

        # Test Normal CI
        boots_ci = bootstrap.confidence_interval(
            alpha=0.1, interval_type=ci.IntervalType.right, bootstrap_ci=ci.BootstrapCI.normal)

        assert boots_ci.lower == theo_ci.lower, "Lower confidence should be -infinity for a one-sided right CI"

        upper_dist = abs(boots_ci.upper - theo_ci.upper)
        assert upper_dist < 0.01, self.error_msg(boots_ci.upper, theo_ci.upper)

        # Test BCA CI
        boots_ci = bootstrap.confidence_interval(
            alpha=0.1, interval_type=ci.IntervalType.right, bootstrap_ci=ci.BootstrapCI.bca)

        assert boots_ci.lower == theo_ci.lower, "Lower confidence should be -infinity for a one-sided right CI"

        upper_dist = abs(boots_ci.upper - theo_ci.upper)
        assert upper_dist < 0.01, self.error_msg(boots_ci.upper, theo_ci.upper)

    def error_msg(boots, theoretical):
        return f"Bootstrap upper: {boots}/nTheoretical upper: {theoretical}"
