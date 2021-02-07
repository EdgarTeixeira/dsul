from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import numpy as np
from scipy import special
from scipy.special import beta, digamma, erf, erfinv, hyp2f1
from scipy.stats import uniform


def check_is_probability(x):
    raise NotImplementedError()


class Interval:
    pass


class ContinuousInterval(NamedTuple):
    lower: float
    upper: float


class DiscreteInterval(NamedTuple):
    lower: int
    upper: int


class Distribution(ABC):
    @abstractmethod
    def mean(self) -> float:
        pass

    @abstractmethod
    def var(self) -> float:
        pass

    @abstractmethod
    def std(self) -> float:
        pass

    @abstractmethod
    def median(self) -> float:
        pass

    @abstractmethod
    def entropy(self) -> float:
        pass

    @abstractmethod
    def support(self) -> Interval:
        pass

    @abstractmethod
    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    @abstractmethod
    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    @staticmethod
    def fit(self, dataset: np.ndarray) -> 'Distribution':
        pass


class ContinuousDistribution(Distribution):
    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass


class DiscreteDistribution(Distribution):
    @abstractmethod
    def pmf(self, x: np.ndarray) -> np.ndarray:
        pass


class Uniform(ContinuousDistribution):
    def __init__(self, lower, upper) -> None:
        self.lower = lower
        self.upper = upper

    def mean(self) -> float:
        return (self.lower + self.upper) / 2

    def var(self) -> float:
        return pow(self.upper - self.lower, 2) / 12

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        return self.mean()

    def entropy(self) -> float:
        return np.log(self.upper - self.lower)

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(self.lower, self.upper)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        dist = uniform(self.lower, (self.upper - self.lower))
        return dist.rvs(size, random_state=random_state)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        constant = 1 / (self.upper - self.lower)
        density = np.where((x < self.lower) | (x > self.upper), 0.0, constant)

        return density

    def cdf(self, x: np.ndarray) -> np.ndarray:
        value = (x - self.lower) / (self.upper - self.lower)
        return np.clip(value, 0.0, 1.0)

    def sf(self, x: np.ndarray) -> np.ndarray:
        return 1 - self.cdf(x)

    def ppf(self, x: np.ndarray) -> np.ndarray:
        check_is_probability(x)

        return self.lower + x * (self.upper - self.lower)

    @staticmethod
    def fit(self, dataset: np.ndarray) -> 'Uniform':
        return Uniform(lower=dataset.min(), upper=dataset.max())


class Normal(ContinuousDistribution):
    def __init__(self, mean, std) -> None:
        self.mean_ = mean
        self.std_ = std

    def mean(self) -> float:
        return self.mean_

    def var(self) -> float:
        return self.std_ ** 2

    def std(self) -> float:
        return self.std_

    def median(self) -> float:
        return self.mean_

    def entropy(self) -> float:
        return 0.5 * np.log(2 * np.pi * np.e * pow(self.std_, 2))

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(-np.inf, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        p = uniform(0, 1).rvs(size, random_state=random_state)
        return self.ppf(p)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        const = 1 / (self.std_ * pow(2 * np.pi, 0.5))
        func = np.exp(-0.5 * np.square((x - self.mean_) / self.std_))

        return const * func

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * (1 + erf((x - self.mean_) / (self.std_ * pow(2, 0.5))))

    def sf(self, x: np.ndarray) -> np.ndarray:
        return 1 - self.cdf(x)

    def ppf(self, x: np.ndarray) -> np.ndarray:
        check_is_probability(x)

        A = self.std_ * pow(2, 0.5) * erfinv(2 * x - 1)
        return self.mean_ + A

    @staticmethod
    def fit(self, dataset: np.ndarray) -> 'Normal':
        mu = dataset.mean()
        sigma = dataset.std(ddof=1)

        return Normal(mean=mu, std=sigma)


class StudentT(ContinuousDistribution):
    def __init__(self, df, loc, scale) -> None:
        self.df = df
        self.loc = loc
        self.scale = scale

    def mean(self) -> float:
        if self.df > 1:
            return self.loc
        return np.nan

    def var(self) -> float:
        if self.df > 2:
            return pow(self.scale, 2) * self.df / (self.df - 2)
        elif self.df > 1:
            return np.inf
        return np.nan

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        return self.loc

    def entropy(self) -> float:
        raise NotImplementedError()

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(-np.inf, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        p = uniform(0, 1).rvs(size, random_state=random_state)
        return self.ppf(p)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        return 1 - self.cdf(x)

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> 'StudentT':
        pass


class Laplace(ContinuousDistribution):
    def __init__(self, mu, b) -> None:
        self.mu = mu
        self.b = b

    def mean(self) -> float:
        return self.mu

    def var(self) -> float:
        return 2 * pow(self.b, 2)

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        return self.mu

    def entropy(self) -> float:
        return np.log(2 * self.b * np.e)

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(-np.inf, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> 'Distribution':
        pass


class Logistic(ContinuousDistribution):
    def __init__(self, loc, scale) -> None:
        self.loc = loc
        self.scale = scale

    def mean(self) -> float:
        return self.loc

    def var(self) -> float:
        return pow(self.scale, 2) * pow(np.pi, 2) / 3

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        return self.loc

    def entropy(self) -> float:
        return np.log(self.scale) + 2

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(-np.inf, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> 'Distribution':
        pass


class Cauchy(ContinuousDistribution):
    def __init__(self, loc, scale) -> None:
        self.loc = loc
        self.scale = scale

    def mean(self) -> float:
        return np.nan

    def var(self) -> float:
        return np.nan

    def std(self) -> float:
        return np.nan

    def median(self) -> float:
        return self.loc

    def entropy(self) -> float:
        return np.log(4 * np.pi * self.scale)

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(-np.inf, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> 'Distribution':
        pass


class Exponential(ContinuousDistribution):
    def __init__(self, rate) -> None:
        self.rate = rate

    def mean(self) -> float:
        return 1 / self.rate

    def var(self) -> float:
        return 1 / pow(self.rate, 2)

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        return np.log(2) / self.rate

    def entropy(self) -> float:
        return 1 - np.log(self.rate)

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(0, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> 'Distribution':
        pass


class Pareto(ContinuousDistribution):
    def __init__(self, xmin, shape) -> None:
        self.xmin = xmin
        self.shape = shape

    def mean(self) -> float:
        if self.shape <= 1:
            return np.inf
        return self.shape * self.xmin / (self.shape - 1)

    def var(self) -> float:
        if self.shape <= 2:
            return np.inf

        num = pow(self.xmin, 2) * self.shape
        den = pow(self.shape - 1, 2) * (self.shape - 2)
        return num / den

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        return self.xmin * pow(2, self.shape)

    def entropy(self) -> float:
        A = self.xmin / self.shape
        B = np.exp(1 + 1 / self.shape)

        return np.log(A * B)

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(self.xmin, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> 'Distribution':
        pass


class Lomax(ContinuousDistribution):
    def __init__(self, shape, scale) -> None:
        self.shape = shape
        self.scale = scale

    def mean(self) -> float:
        if self.shape > 1:
            return self.scale / (self.shape - 1)
        return np.nan

    def var(self) -> float:
        if self.shape <= 1:
            return np.nan
        elif self.shape <= 2:
            return np.inf

        A = pow(self.scale, 2) * self.shape
        B = pow(self.shape - 1, 2) * (self.shape - 2)

        return A / B

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        return self.scale * (pow(2, 1 / self.shape) - 1)

    def entropy(self) -> float:
        raise NotImplementedError()

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(0, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> Distribution:
        pass


class LogNormal(ContinuousDistribution):
    def __init__(self, mean, std) -> None:
        self.mean_ = mean
        self.std_ = std

    def mean(self) -> float:
        return np.exp(self.mean_ + pow(self.std_, 2) / 2)

    def var(self) -> float:
        A = np.exp(pow(self.std_, 2)) - 1
        B = np.exp(2 * self.mean_ + pow(self.std_, 2))
        return A * B

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        return np.exp(self.mean_)

    def entropy(self) -> float:
        A = self.std_ * pow(2 * np.pi, 0.5) * np.exp(self.mean_ + 0.5)
        return np.log(A) / np.log(2)

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(0, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> 'Distribution':
        pass


class Weibull(ContinuousDistribution):
    def __init__(self, scale, shape) -> None:
        self.scale = scale
        self.shape = shape

    def mean(self) -> float:
        return self.scale * special.gamma(1 + 1 / self.shape)

    def var(self) -> float:
        A = special.gamma(1 + 2 / self.shape)
        B = special.gamma(1 + 1 / self.shape)

        return pow(self.scale, 2) * (A - pow(B, 2))

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        return self.scale * pow(np.log(2), 1 / self.shape)

    def entropy(self) -> float:
        A = np.euler_gamma * (1 - 1 / self.shape)
        B = np.log(self.scale / self.shape)

        return A + B + 1

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(0, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> 'Distribution':
        pass


class Gamma(ContinuousDistribution):
    def __init__(self, alpha, beta) -> None:
        self.alpha = alpha
        self.beta = beta

    def mean(self) -> float:
        return self.alpha / self.beta

    def var(self) -> float:
        return self.alpha / pow(self.beta, 2)

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        raise NotImplementedError()

    def entropy(self) -> float:
        A = self.alpha - np.log(self.beta) + special.gammaln(self.alpha)
        B = (1 - self.alpha) * special.digamma(self.alpha)

        return A + B

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(0, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> 'Distribution':
        pass


class ChiSquare(ContinuousDistribution):
    def __init__(self, k) -> None:
        self.k = k

    def mean(self) -> float:
        return self.k

    def var(self) -> float:
        return 2 * self.k

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        raise NotImplementedError()

    def entropy(self) -> float:
        A = self.k / 2 + np.log(2 * special.gamma(self.k / 2))
        B = (1 - self.k / 2) * special.digamma(self.k / 2)

        return A + B

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(0, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> Distribution:
        pass


class Beta(ContinuousDistribution):
    def __init__(self, alpha, beta) -> None:
        self.alpha = alpha
        self.beta = beta

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def var(self) -> float:
        A = self.alpha * self.beta
        B = pow(self.alpha + self.beta, 2) * (self.alpha + self.beta + 1)
        return A / B

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        raise NotImplementedError()

    def entropy(self) -> float:
        A = special.betaln(self.alpha, self.beta)
        B = (self.alpha - 1) * special.digamma(self.alpha)
        C = (self.beta - 1) * special.digamma(self.beta)
        D = (self.alpha + self.beta - 2) * \
            special.digamma(self.alpha + self.beta)

        return A - B - C + D

    def support(self) -> ContinuousInterval:
        return ContinuousInterval(0, 1)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> Distribution:
        pass


class Bernoulli(DiscreteDistribution):
    def __init__(self, p) -> None:
        self.p = p

    def mean(self) -> float:
        return self.p

    def var(self) -> float:
        return self.p * (1 - self.p)

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        if self.p < 0.5:
            return 0
        elif self.p > 0.5:
            return 1
        return self.p

    def entropy(self) -> float:
        return -((1 - self.p) * np.log(1 - self.p) + self.p * np.log(self.p))

    def support(self) -> DiscreteInterval:
        return DiscreteInterval(0, 1)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pmf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> Distribution:
        pass


class Binomial(DiscreteDistribution):
    def __init__(self, p, n) -> None:
        self.p = p
        self.n = n

    def mean(self) -> float:
        return self.p * self.n

    def var(self) -> float:
        return self.p * (1 - self.p) * self.n

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        return self.mean()

    def entropy(self) -> float:
        raise NotImplementedError()

    def support(self) -> DiscreteInterval:
        return DiscreteInterval(0, self.n)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pmf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> Distribution:
        pass


class Hypergeometric(DiscreteDistribution):
    def __init__(self, n, N, K) -> None:
        self.n = n
        self.N = N
        self.K = K

    def mean(self) -> float:
        return self.n * self.K / self.N

    def var(self) -> float:
        A = self.n * self.K / self.N
        B = (self.N - self.K) / self.N
        C = (self.N - self.n) / (self.N - 1)

        return A * B * C

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        raise NotImplementedError()

    def entropy(self) -> float:
        raise NotImplementedError()

    def support(self) -> DiscreteInterval:
        return DiscreteInterval(max(0, self.n + self.K - self.N), min(self.n, self.K))

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pmf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> Distribution:
        pass


class Geometric(DiscreteDistribution):
    def __init__(self, p) -> None:
        self.p = p

    def mean(self) -> float:
        return (1 - self.p) / self.p

    def var(self) -> float:
        return (1 - self.p) / pow(self.p, 2)

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        A = -1 / np.log2(1 - self.p)
        return A - 1

    def entropy(self) -> float:
        return -((1 - self.p) * np.log(1 - self.p) + self.p * np.log(self.p)) / self.p

    def support(self) -> DiscreteInterval:
        return DiscreteInterval(0, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pmf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> Distribution:
        pass


class Poisson(DiscreteDistribution):
    def __init__(self, rate) -> None:
        self.rate = rate

    def mean(self) -> float:
        return self.rate

    def var(self) -> float:
        return self.rate

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        raise NotImplementedError()

    def entropy(self) -> float:
        raise NotImplementedError()

    def support(self) -> DiscreteInterval:
        return DiscreteInterval(0, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pmf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> Distribution:
        pass


class ZeroInflatedPoisson(DiscreteDistribution):
    def __init__(self, rate, p) -> None:
        self.rate = rate
        self.p = p

    def mean(self) -> float:
        return (1 - self.p) * self.rate

    def var(self) -> float:
        return self.rate * (1 - self.p) * (1 + self.p * self.rate)

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        raise NotImplementedError()

    def entropy(self) -> float:
        raise NotImplementedError()

    def support(self) -> DiscreteInterval:
        return DiscreteInterval(0, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pmf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> Distribution:
        pass


class NegativeBinomial(DiscreteDistribution):
    def __init__(self, p, r) -> None:
        self.p = p
        self.r = r

    def mean(self) -> float:
        return self.p * self.r / (1 - self.p)

    def var(self) -> float:
        return self.p * self.r / pow(1 - self.p, 2)

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        raise NotImplementedError()

    def entropy(self) -> float:
        raise NotImplementedError()

    def support(self) -> DiscreteInterval:
        return DiscreteInterval(0, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pmf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> Distribution:
        pass


class NegativeHypergeometric(DiscreteDistribution):
    def __init__(self, r, N, K) -> None:
        self.r = r
        self.N = N
        self.K = K

    def mean(self) -> float:
        return self.r * self.K / (self.N - self.K + 1)

    def var(self) -> float:
        num = self.r * (self.N + 1) * self.K
        den = (self.N - self.K + 1) * (self.N - self.K + 2)
        A = num / den
        B = 1 - self.r / (self.N - self.K + 1)

        return A * B

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        raise NotImplementedError()

    def entropy(self) -> float:
        raise NotImplementedError()

    def support(self) -> DiscreteInterval:
        return DiscreteInterval(0, self.K)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pmf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> Distribution:
        pass


class Zeta(DiscreteDistribution):
    def __init__(self, shape) -> None:
        self.shape = shape

    def mean(self) -> float:
        if self.shape > 2:
            return special.zeta(self.shape - 1) / special.zeros(self.shape)
        raise NotImplementedError()

    def var(self) -> float:
        if self.shape > 3:
            A = special.zeta(self.shape)
            B = special.zeta(self.shape - 2)
            C = pow(special.zeta(self.shape - 1), 2)

            return (A * B - C) / pow(A, 2)
        raise NotImplementedError()

    def std(self) -> float:
        return np.sqrt(self.var())

    def median(self) -> float:
        raise NotImplementedError()

    def entropy(self) -> float:
        raise NotImplementedError()

    def support(self) -> DiscreteInterval:
        return DiscreteInterval(0, np.inf)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        pass

    def pmf(self, x: np.ndarray) -> np.ndarray:
        pass

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sf(self, x: np.ndarray) -> np.ndarray:
        pass

    def ppf(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def fit(self, dataset: np.ndarray) -> Distribution:
        pass
