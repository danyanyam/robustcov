from abc import ABC, abstractmethod
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize, Bounds
import pandas as pd
from typing import List as L, Tuple as T

from robvariance.utils import corr2cov, cov2corr


class AbstractTransformer(ABC):
    @abstractmethod
    def transform(self):
        ...


class DenoiseCovTransformer(AbstractTransformer):
    def __init__(
        self,
        q: float,
        bandwidth: float,
        pts: int = 1000,
        kernel: str = 'gaussian'
    ) -> None:

        self.q = q
        self.bandwidth = bandwidth
        self.pts = pts
        self.kernel = kernel

        assert self.q > 0
        assert self.bandwidth > 0

    def transform(self, cov: np.ndarray) -> np.ndarray:

        corr = cov2corr(cov)

        # eigen_values - cols a = 0 with a[i] = \lambda
        eigen_values, eigen_vectors = self._transform_pca(corr)
        eigen_max = self._derive_max_eigen_value(eigen_values)

        # choosing how many top eigen values must be saved, since they are
        # higher than the estimated expected maximum eigen value
        save_top: int = len(eigen_values) - \
            eigen_values[::-1].searchsorted(eigen_max)

        # pad noisy eigen values with average evalue and return denoised
        # correlation
        corr_denoised = self._calculate_shrinked_corr(
            eigen_values,
            eigen_vectors,
            save_top
        )
        return corr2cov(corr_denoised, np.diag(cov)**.5)

    def _transform_pca(self, corr: np.ndarray) -> T[np.ndarray]:
        # get eigen values, eigen vectors from a Hermitian matrix
        eigen_values, eigen_vectors = np.linalg.eigh(corr)

        # sorting eigen values in descending order
        indices = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[indices]
        eigen_vectors = eigen_vectors[:, indices]

        return eigen_values, eigen_vectors

    def _derive_max_eigen_value(self, eigen_values):
        # Find max random eVal by fitting Marcenko's dist to the empirical one
        # this is done to define variance in order to use formula for maximum
        # eigen value

        out = minimize(
            lambda *x: self._sse_PDFs(*x),
            np.array([0.5]),
            args=(eigen_values),
            bounds=Bounds(1E-5, 1-1E-5)
        )

        var = out['x'][0] if out['success'] else 1
        eigen_max = var * (1 + (1 / self.q)**.5)**2

        return eigen_max

    def _sse_PDFs(self, var: float, eigen_values: np.ndarray) -> float:
        # theoretical pdf
        theoretical_pdf = self._mp_PDF(var)
        # empirical pdf
        empirical_pdf = self._fit_kde(eigen_values=eigen_values,
                                      x=theoretical_pdf.index.values)

        return np.sum((empirical_pdf - theoretical_pdf)**2)

    def _mp_PDF(self, var: L[float]) -> pd.Series:
        """Marcenko-Pastur pdf"""
        # q=T/N
        var = var[0]
        eigen_min = var * (1 - (1 / self.q)**.5)**2
        eigen_max = var * (1 + (1 / self.q)**.5)**2
        eigen_values = np.linspace(eigen_min, eigen_max, self.pts)

        pdf = self.q / (2 * np.pi * var * eigen_values) * \
            ((eigen_max - eigen_values) * (eigen_values-eigen_min))**.5
        return pd.Series(pdf, index=eigen_values.flatten())

    def _fit_kde(self, eigen_values: np.ndarray, x: np.ndarray) -> pd.Series:
        # observations to column vector

        if len(eigen_values.shape) == 1:
            eigen_values = eigen_values.reshape(-1, 1)

        kde = KernelDensity(kernel=self.kernel,
                            bandwidth=self.bandwidth)
        kde = kde.fit(eigen_values)

        # if x were not passed, then we use unique observations
        x = np.unique(eigen_values).reshape(-1, 1) if x is None else x
        x = x.reshape(-1, 1) if len(x.shape) == 1 else x

        logprob = kde.score_samples(x)  # log(density)
        return pd.Series(np.exp(logprob), index=x.flatten())

    def _calculate_shrinked_corr(
        self,
        sorted_ev: np.ndarray,
        eigen_vectors: np.ndarray,
        save_top: int
    ) -> np.ndarray:
        """shrinks the eigenvalues associated with noise, and returns a de-noised
        correlation matrix"""
        # Remove noise from corr by fixing random eigenvalues

        # sorted by decreasing order
        sorted_ev = sorted_ev.copy()
        assert len(sorted_ev.shape) == 1

        # replacing each eigen value, less than the maximum expected with
        # average eigen value
        pad = sorted_ev[save_top:].sum() / (len(sorted_ev) - save_top)
        sorted_ev[save_top:] = pad
        sorted_ev = np.diag(sorted_ev)

        # svd decomposition
        cov1 = np.dot(eigen_vectors, sorted_ev).dot(eigen_vectors.T)
        return cov2corr(cov1)
