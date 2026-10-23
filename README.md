# Matched Gibs Posterior
Official implementation of the ICLR 2026 paper "A Fair Bayesian Inference through Matched Gibbs Posterior".

Paper: https://proceedings.iclr.cc/paper_files/paper/2026/hash/61c2975281d60d3b1ce4cefc157d99df-Abstract-Conference.html


## Usage
* run files in `experiments`
  * `crime_pretrain_mdp.sh`: run pretraining for matched Gibbs constraint
  * `crime_pretrain_mmd.sh`: run pretraining for Gibbs with MMD constraint
  * `crime_matched_gibbs_mdp.sh`: run matched Gibbs
    * run `crime_matched_gibbs_mdp.sh` after running `crime_pretrain_mdp.sh`
  * `crime_gibbs_mmd.sh`: run Gibbs with MMD constraint
    * run `crime_gibbs_mmd.sh` after running `crime_pretrain_mmd.sh`
  * `crime_mfg_mmd.sh`: run mean-field Gaussian variational inference with MMD constraint

## Structure
```
.
├── main.py                       # main file
├── src/                          # source code
│   ├── runner_bayesian.py        # runner for Gibbs with MMD constraint and matched Gibbs
│   ├── runner_variational.py     # runner for mean-field Gaussian variational inference
│   ├── train.py                  # training script
│   ├── eval.py                   # evaluation script
│   ├── loss.py                   # loss functions
│   ├── models.py                 # models
│   ├── dataset.py                # torch dataset with sensitive attributes, balanced sampler
│   └── utils.py                  # utility functions
├── hamiltorch/                   # Hamiltonian Monte Carlo
├── data/                         # data preprocessing code
│   ├── preprocessor.py           # data preprocessor for all datasets
│   ├── registry.py               # registry for each data preprocessor
│   ├── datasets/                 # datasets
│   │   ├── adult.py              # Adult
│   │   ├── crime.py              # Crime
│   │   ├── dutch.py              # Dutch
│   │   ├── celeba_hq.py          # CelebAMask-HQ
│   │   └── civil.py              # CivilComments with RoBERTa embedding
│   └── celeba_HQ/                # CelebA-HQ
│       └── embedding.py          # embedding code
└── experiments/                  # example scripts
```

* For `hamiltorch`, we slightly modified the original code from https://github.com/hamiltorch/hamiltorch