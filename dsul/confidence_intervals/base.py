from enum import Enum
from dataclasses import dataclass


class BootstrapCI(Enum):
    percentile = 0
    normal = 1
    bca = 2


class IntervalType(Enum):
    symmetric = 0
    left = 1
    right = 2


@dataclass
class ConfidenceInterval:
    lower: float
    upper: float
    alpha: float

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


def _get_alphas(alpha, interval_type: IntervalType):
    if interval_type == IntervalType.symmetric:
        return alpha / 2, 1 - alpha / 2
    elif interval_type == IntervalType.left:
        return alpha, 1.0
    elif interval_type == IntervalType.right:
        return 0.0, 1 - alpha
    raise NotImplementedError("Invalid IntervalType")
