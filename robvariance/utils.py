import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from typing import Tuple as T, Optional as O


def corr2cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    """recovers the covariance matrix from the de-noise correlation matrix"""
    # creating matrix of stds and multiply it by correlations
    # elementwise
    return corr * np.outer(std, std)


def cov2corr(cov: np.ndarray) -> np.ndarray:
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr


def init_mu_cov(
    blocks_num: int,
    blocks_size: int,
    blocks_corr: float,
    std: O[np.ndarray] = None
) -> T[np.ndarray, pd.DataFrame]:
    """Code snippet 7 creates a random vector of means and a random covariance
    matrix that represent a stylized version of a 50 securities portfolio,
    grouped in 10 blocks with intra-cluster correlations of 0.5. This vector
    and matrix characterize the “true” process that generates observations,
    {𝜇, 𝑉}.  We set a seed for the purpose of reproducing results across runs
    with different parameters. In practice, the pair {𝜇, 𝑉} does not need to be
    simulated, and MCOS receives {𝜇, 𝑉} as an input."""

    # create block-diagonal matrix
    corr0 = generate_correlation_block_matrix(
        blocks_num,
        blocks_size,
        blocks_corr
    )

    # shuffling columns and make rows correspond to columns
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0.loc[cols, cols].copy(deep=True)

    if std is None:
        std = np.random.uniform(.05, .2, len(corr0))
    else:
        std = np.array([std]*corr0.shape[1])

    # transforming correlation matrix to covariance matrix
    cov0 = corr2cov(corr0, std)
    mu0 = np.random.normal(std, std, cov0.shape[0]).reshape(-1, 1)
    return mu0, cov0


def generate_correlation_block_matrix(
    block_num: int,
    block_size: int,
    block_corr: float
) -> pd.DataFrame:
    """Creates correlation block matrix from specified settings

    Args:
        block_num (int): amount of correlation clusters
        block_size (int): amount of stocks in each correlation cluster
        block_corr (float): correlation among stocks in cluster

    Returns:
        pd.DataFrame: matrix of
            (block_num * block_size,  block_num * block_size)
        size, representing pairwise stock correlations
    """
    # creating 1 block
    block = np.ones((block_size, block_size)) * block_corr
    np.fill_diagonal(block, 1)
    # creating nBlocks from 1 block
    corr = block_diag(*([block] * block_num))
    return pd.DataFrame(corr)
