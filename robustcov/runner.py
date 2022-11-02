import numpy as np
from typing import List as L

from robustcov.optimizers import AbstractOptimizer
from robustcov.sampler import MultivariateNormalMuCovSampler
from robustcov.denoiser import CovDenoiser


class PortfolioCreator:

    def __init__(
        self,
        optimizers: L[AbstractOptimizer],
        time_periods: int = 100,
        trials: int = 100,
        bandwidth: float = 0.2,
        min_var_portf: bool = True,
        shrink: bool = False,
        pts: int = 10000,
        kernel: str = 'gaussian',
        print_every: int = 10

    ):
        self.optimizers = optimizers
        assert len(self.optimizers) > 0

        self.time_periods = time_periods
        self.trials = trials
        self.bandwidth = bandwidth
        self.min_var_portf = min_var_portf
        self.shrink = shrink
        self.pts = pts
        self.kernel = kernel
        self.print_every = print_every

    def estimate(self, mu: np.ndarray, cov: np.ndarray):

        results = {}

        sampler = MultivariateNormalMuCovSampler(
            mu=mu,
            cov=cov,
            shrink=self.shrink,
            time_periods=self.time_periods
        )
        denoiser = CovDenoiser(
            q=self.time_periods / len(cov),
            bandwidth=self.bandwidth,
            pts=self.pts,
            kernel=self.kernel
        )

        for trial in range(self.trials):

            if bool(self.print_every):
                if (trial % self.print_every == 0):
                    print(f'[{trial}/{self.trials}] done')

            mu_simulated, cov_simulated = sampler.simulate()
            cov_denoised = denoiser.transform(cov_simulated)

            for optimizer in self.optimizers:

                if self.min_var_portf and str(optimizer) == 'ConvexOptimizer':
                    mu_simulated = None

                optimizer.fit(mu=mu_simulated, cov=cov_denoised)
                results[(str(optimizer), trial)] = optimizer.get_params()

        return results
