# A Robust Estimator of the Efficient Frontier

The repository contains a replication of the code from the article, as well as
a module, named `robustcov` for simulating portfolio weights using the MCO
method, as well as convex optimization on a denoised covariance matrix.

In the folder `notebooks` you can find application of NCO algorithm on the numeric
example from article (`notebooks/block_simulated_optimization.ipynb`) and
real-world application on provided data as well as application for 5 years
investments and some analysis
(`notebooks/portfolio_analysis.ipynb`)

## 🔖 Contents

- [🗂 Folder structure](https://github.com/danyanyam/robustcov#-folder-structure)
- [🛠 Installation and Dependencies](https://github.com/danyanyam/robustcov#-installation-and-dependencies)
- [💹 Usage](https://github.com/danyanyam/robustcov#-usage)

---

### 🗂 Folder structure

```bash

├── examples
│   └── block_simulated_optimization.ipynb
├── reference
│	└──...
├── replication
│   └── raw.ipynb
├── robustcov
│	├── denoiser.py
│	├── optimizers.py
│	├── runner.py
│	├── sampler.py
│	└── utils.py
├── tests
│	└── test_sampler.py
├── requirements.txt
├── setup.py
└── README.MD
```

1. In the folder `replication`, you can find raw code from the
   [article](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961).

2. In the folder `robustcov` you can find a bunch of scripts, that are used in examples folder.

---

### 🛠 Installation and Dependencies

`git clone` and `cd` to the folder with project and then use the following commands:

```bash
pip install -r requirements.txt;
pip install -e .
```

---

### 💹 Usage

Basic usage is shown in the `notebooks` folder. The most straightforward is
block simulated optimization, described in the article:

```python
def numeric_example():
    np.random.seed(0)

    blocks_num, blocks_size, blocks_corr = 10, 5, 0.5

    # generate synthetic data
    mu, cov = init_mu_cov(
        blocks_num=blocks_num,
        blocks_size=blocks_size,
        blocks_corr=blocks_corr,
    )

    # choose NCO and convex optimizers on denoised data
    optimizers = [
        ConvexOptimizer(),
        NCOOptimizer()
    ]

    # init estimator
    estimator = PortfolioCreater(optimizers=optimizers, print_every=10)

    # obtain matrix of weights, where rows - trials and columns -
    # simulated weights for corresponding assets
    return estimator.estimate(mu, cov)



results = numeric_example()

```
