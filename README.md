# Attention Mixtures for Time-Aware Sequential Recommendation

This repository provides Python code for our paper:

V-A. Tran, G. Salha-Galvan, B. Sguerra, R. Hennequin. Attention Mixtures for Time-Aware Sequential Recommendation. In: *Proceedings of 46th International ACM SIGIR Conference on Research and Development in Information Retrieval*, July 2023.

## Environment
- python 3.9.13
- tensorflow 2.11.0
- tqdm 4.65.0
- numpy 1.24.2
- scipy 1.10.1
- pandas 1.5.3
- toolz 0.12.0

## Datasets
The following datasets are considered in our work that could be easily downloaded from Internet and put in `exp/data` directory 
- Movielens 1M (https://grouplens.org/datasets/movielens/1m/)
- Amazon Book (https://jmcauley.ucsd.edu/data/amazon/)
- LFM 1B (http://www.cp.jku.at/datasets/LFM-1b/)

## Hyperparameters
Hyperparameters are in the corresponding configuration file in `configs` directory.

## Experiments
All experiment scripts for train / evaluation of our models and other baselines described in the paper could be found in `scripts` directory.

You could do the following steps to run experiment:
1. Download data and put it into `exp/data` directory. For example `exp/data/ml1m` for Movielens 1M.
2. Change data path and interaction file name in configuration file (for example `configs/ml1m.json`).
3. Run experiment script (that contains both train and evaluation commands) in `scripts` directory