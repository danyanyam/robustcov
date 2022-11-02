import numpy as np
from typing import Tuple as T
from abc import ABC, abstractmethod
from sklearn.covariance import LedoitWolf


class AbstractSampler(ABC):
    @abstractmethod
    def simulate(self):
        ...


class MultivariateNormalMuCovSampler(AbstractSampler):

    def __init__(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        shrink: bool,
        time_periods: int
    ) -> None:

        self.mu = mu.flatten()
        self.cov = cov
        self.shrink = shrink
        self.time_periods = time_periods
        assert len(mu) == len(cov)
        assert time_periods > 0

    def simulate(self) -> T[np.ndarray]:

        sample = np.random.multivariate_normal(
            self.mu,
            self.cov,
            size=self.time_periods
        )

        expected_return = sample.mean(axis=0).reshape(-1, 1)

        if self.shrink:
            covariance = LedoitWolf().fit(sample).covariance_
        else:
            covariance = np.cov(sample, rowvar=0)
        return expected_return, covariance
