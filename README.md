# A Robust Estimator of the Efficient Frontier

## 🔖 Contents
- [🛠 Installation and Dependencies](https://github.com/danyanyam/robust_variance#-installation-and-dependencies)
- [🗂 Folder structure](https://github.com/danyanyam/robust_variance#-folder-structure)

### 🛠 Installation and Dependencies
`git clone` and `cd` to the folder with project and then use the following commands:

```bash
pip install -r requirements.txt;
pip install -e .
```

### 🗂 Folder structure

```bash

├── examples
│   └── block_simulated_optimization.ipynb
│
├── reference
│	└──...
│
├── replication
│   └── raw.ipynb
│
├── robustcov
│	├── denoiser.py
│	├── optimizers.py
│	├── runner.py
│	├── sampler.py
│	└── utils.py
│
├── tests
│	└──test_sampler.py
│
├── requirements.txt
├── setup.py
└── README.MD
```

1. In the folder `replication`, you can find raw code from the
   [article](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961).

2. In the folder `robustcov` you can find a bunch of scripts, that are used in examples folder.
