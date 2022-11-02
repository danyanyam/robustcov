from robvariance.runner import PortfolioCreater
from robvariance.optimizers import ConvexOptimizer
from robvariance.optimizers import NCOOptimizer
from robvariance.utils import init_mu_cov


def main():

    blocks_num, blocks_size, blocks_corr = 10, 5, 0.5

    mu, cov = init_mu_cov(
        blocks_num=blocks_num,
        blocks_size=blocks_size,
        blocks_corr=blocks_corr,
    )

    optimizers = [
        ConvexOptimizer(),
        NCOOptimizer()
    ]

    estimator = PortfolioCreater(optimizers=optimizers)
    results = estimator.estimate(mu, cov)

    print(results)


if __name__ == "__main__":
    main()
