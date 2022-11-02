import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
from robvariance.utils import cov2corr
from typing import Optional as O, Dict as D
from sklearn.metrics import silhouette_samples


class AbstractOptimizer(ABC):

    @abstractmethod
    def fit(self, cov: np.ndarray, mu: np.ndarray) -> None:
        """ represents what to optimize """

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """ represents how optimization results are retrieved """


class ConvexOptimizer(AbstractOptimizer):
    """ Applies convex optimization algorithm """

    def fit(self, cov: np.ndarray, mu: O[np.ndarray] = None) -> None:
        """

        Args:
            cov (np.ndarray): covariance matrix
            mu (O[np.ndarray], optional): represents convex optimization
                constraint. If none, then variance is minimised.
        """
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(cov)

        ones = np.ones(shape=(len(inv), 1))
        mu = ones if mu is None else mu

        w = np.dot(inv, mu)
        w /= np.dot(ones.T, w)
        self.w = w

    def get_params(self):
        return self.w.flatten()

    def __str__(self):
        return "ConvexOptimizer"


class NCOOptimizer(AbstractOptimizer):
    """ Applies Nested Clustered Optimization Algorithm """

    def __init__(self, max_clusters: O[int] = None, n_init: int = 10):
        self.max_clusters = max_clusters
        self.n_init = n_init
        self._convex_opt = ConvexOptimizer()

        assert self.n_init > 0

    def _define_clusters(self, corr: np.ndarray) -> D[int, int]:
        """Returns optimal clusters mapping

        Args:
            corr (np.ndarray): correlation matrix across assets

        Returns:
            D[int, int]: mapping from asset to cluster
        """

        distance_matrix = ((1 - corr.fillna(0)) / 2.)**.5
        silhouettes = pd.Series(dtype=object)

        if self.max_clusters is None:
            self.max_clusters = len(corr) // 2

        assert self.max_clusters <= len(corr) // 2, "clusters are maximum half"

        # create random knn clustering
        for _ in range(self.n_init):
            for n_clusters in range(2, self.max_clusters + 1):

                # clustering correlation matrix on specified number of blocks
                kmeans_ = KMeans(n_clusters=n_clusters, n_init=1)
                kmeans_ = kmeans_.fit(distance_matrix)

                # evaluating metrics on each knn fit
                silhouettes_ = silhouette_samples(distance_matrix,
                                                  kmeans_.labels_)
                running = silhouettes_.mean() / silhouettes_.std()
                existing = silhouettes.mean() / silhouettes.std()

                # update if metric has improved
                if np.isnan(existing) or running > existing:
                    existing = silhouettes_
                    kmeans = kmeans_

        clusters = {
            i: corr.columns[np.where(kmeans.labels_ == i)[0]].tolist()
            for i in np.unique(kmeans.labels_)
        }

        return clusters

    def fit(
        self,
        cov: np.ndarray,
        mu: np.ndarray,
    ) -> None:
        """Calculates optimal weights, using NCO algorithm

        Args:
            cov (np.ndarray): denoised sampled covariance matrix
            mu (np.ndarray): sampled expected return array

        """
        cov = pd.DataFrame(cov)

        if mu is not None:
            mu = pd.Series(mu.flatten())
            assert len(mu) == len(cov)

        # get correlation matrix based on sampled covariance matrix
        corr1 = cov2corr(cov)

        # define which asset to which cluster is assigned
        clusters: D[int, int] = self._define_clusters(corr1)

        # init optimal weights within each cluster
        intra_weights = pd.DataFrame(0,
                                     index=cov.index,
                                     columns=clusters.keys())

        # calculating optimal allocations within each cluster of covariances
        for cluster_num, cluster in clusters.items():
            cov_ = cov.loc[cluster, cluster].values

            if mu is None:
                mu_ = None
            else:
                mu_ = mu.loc[cluster].values.reshape(-1, 1)

            self._convex_opt.fit(cov_, mu_)
            params = self._convex_opt.get_params()
            intra_weights.loc[cluster, cluster_num] = params

        # calculate optimal allocations across different clusters
        cov_ = intra_weights.T.dot(np.dot(cov, intra_weights))
        mu_ = (None if mu is None else intra_weights.T.dot(mu))

        self._convex_opt.fit(cov_, mu_)
        params = self._convex_opt.get_params()
        inter_weights = pd.Series(params, index=cov_.index)

        nco = intra_weights.mul(inter_weights, axis=1).sum(axis=1)
        nco = nco.values.reshape(-1, 1)
        self.w = nco

    def get_params(self):
        return self.w.flatten()

    def __str__(self):
        return "NCO Optimizer"
