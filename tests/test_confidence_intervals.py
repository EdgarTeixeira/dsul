import eul.confidence_intervals as ci
import numpy as np


class TestSimpleBootstrap:
    def test_symmetric_percentile(self):
        sample, mean, std_error = self.generate_dataset()

        bootstrap = ci.SimpleBootstrap(sample, np.mean)
        bootstrap.run(iterations=1000)

        lower, upper, _ = bootstrap.confidence_interval(alpha=0.1)
        tlower, tupper, _ = ci.mean_ci(sample, alpha=0.1, std_dev=1)

        lower_dist = abs(lower - tlower)
        assert lower_dist < 0.01, self.error_msg(lower, tlower)

        upper_dist = abs(upper - tupper)
        assert upper_dist < 0.01, self.error_msg(upper, tupper)

    def test_left_percentile(self):
        sample, mean, std_error = self.generate_dataset()

        bootstrap = ci.SimpleBootstrap(sample, np.mean)
        bootstrap.run(iterations=1000)

        lower, upper, _ = bootstrap.confidence_interval(
            alpha=0.1, interval_type=ci.IntervalType.left)
        tlower, tupper, _ = ci.mean_ci(
            sample, alpha=0.1, std_dev=1, interval_type=ci.IntervalType.left)

        assert upper == tupper, "Upper confidence should be infinite for a one-sided left CI"

        lower_dist = abs(lower - tlower)
        assert lower_dist < 0.01, self.error_msg(lower, tlower)

    def test_right_percentile(self):
        sample, mean, std_error = self.generate_dataset()

        bootstrap = ci.SimpleBootstrap(sample, np.mean)
        bootstrap.run(iterations=1000)

        lower, upper, _ = bootstrap.confidence_interval(
            alpha=0.1, interval_type=ci.IntervalType.right)
        tlower, tupper, _ = ci.mean_ci(
            sample, alpha=0.1, std_dev=1, interval_type=ci.IntervalType.right)

        assert lower == tlower, "Lower confidence should be -infinity for a one-sided right CI"

        upper_dist = abs(upper - tupper)
        assert upper_dist < 0.01, self.error_msg(upper, tupper)

    def generate_dataset(self, sample_size=1000):
        sample = np.random.normal(0, 1, size=sample_size)
        mean = sample.mean()
        std_error = 1 / np.sqrt(sample.size)

        return sample, mean, std_error

    def error_msg(boots, theoretical):
        return f"Bootstrap upper: {boots}/nTheoretical upper: {theoretical}"
