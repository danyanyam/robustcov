import unittest
import numpy as np
from robvariance.sampler import MultivariateNormalMuCovSampler


class SamplerTesting(unittest.TestCase):
    """ test sampler doesnt mess with shapes """

    def test_sampler_sizes(self):
        mu = np.array([1, 2, 3])
        cov = np.identity(3)

        sampler = MultivariateNormalMuCovSampler(
            mu,
            cov,
            False,
            100
        )
        mu_sim, cov_sim = sampler.simulate()

        assert len(mu_sim) == len(mu)
        assert cov_sim.shape == cov.shape

    def test_ledoitsampler_sizes(self):
        mu = np.array([1, 2, 3])
        cov = np.identity(3)

        sampler = MultivariateNormalMuCovSampler(
            mu,
            cov,
            True,
            100
        )
        mu_sim, cov_sim = sampler.simulate()

        assert len(mu_sim) == len(mu)
        assert cov_sim.shape == cov.shape
