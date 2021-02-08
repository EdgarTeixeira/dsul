import dsul.linear_model as lm
import numpy as np


def test_least_absolute_deviations():
    X = np.zeros(3).reshape(-1, 1)
    y = np.asarray([1, 2, 100_000])

    reg = lm.LeastAbsoluteDeviation()
    reg.fit(X, y)

    assert np.allclose(reg.intercept_, 2.0)

    X = np.random.normal(size=[5000, 2])
    y = X[:, 0] - 3 * X[:, 1] + np.pi

    reg = lm.LeastAbsoluteDeviation(max_iter=1000)
    reg.fit(X, y)

    assert np.allclose(reg.coef_, [1, -3])
    assert np.allclose(reg.intercept_, np.pi)


def test_least_median_squares():
    X = np.random.normal(size=99)
    y = 2 * X - 3

    lms = lm.least_median_squares(X, y)

    assert np.allclose(lms.slope, 2)
    assert np.allclose(lms.intercept, -3)

    X = np.random.normal(size=100)
    y = 2 * X - 3

    lms = lm.least_median_squares(X, y)

    assert np.allclose(lms.slope, 2)
    assert np.allclose(lms.intercept, -3)
