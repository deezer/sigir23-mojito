# MOJITO


This repository provides our Python code to reproduce experiments from the paper **Attention Mixtures for Time-Aware Sequential Recommendation**, accepted for publication in the proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval ([SIGIR 2023](https://sigir.org/sigir2023/)). The paper is available online [on arXiv](https://arxiv.org/pdf/2304.08158.pdf).


## Attention Mixtures for Time-Aware Sequential Recommendation**

**Transformers** emerged as powerful methods for **sequential recommendation**. However, existing architectures often overlook the complex dependencies between user preferences and the temporal context. 

In our SIGIR 2023 paper, we introduce **MOJITO**, an improved Transformer sequential recommender system that addresses this limitation. MOJITO leverages **Gaussian mixtures** of attention-based **temporal contex**t and **item embedding representations** for sequential modeling. Such an approach permits to accurately predict which items should be recommended next to users depending on past actions and the temporal context.

We demonstrate the relevance of our approach, by empirically outperforming existing Transformers for sequential recommendation on three real-world datasets covering various application domains: movie, book, and music recommendation. 

<p align="center">
  <img height="325" src="figures/mojito.pdf">
</p>


## Environment
- python 3.9.13
- tensorflow 2.11.0
- tqdm 4.65.0
- numpy 1.24.2
- scipy 1.10.1
- pandas 1.5.3
- toolz 0.12.0

## Datasets

Please download the datasets used in experiments in the links provided below, and put them in the `exp/data` directory. 
- [Movielens 1M](https://grouplens.org/datasets/movielens/1m/)
- [Amazon Book](https://jmcauley.ucsd.edu/data/amazon/)
- [LFM 1B](http://www.cp.jku.at/datasets/LFM-1b/)

## Hyperparameters

Optimal model hyperparameters are reported in the in the `configs` directory.

## Experiments

All experiment scripts for train / evaluation of our models and other baselines described in the paper can be found in the `scripts` directory.

To run experiment:
1. Download datasets and put them in the `exp/data` directory. For example `exp/data/ml1m` for Movielens 1M.
2. Change data path and interaction file name in configuration file (for example `configs/ml1m.json`).
3. Run experiment script (that contains both train and evaluation commands) in `scripts` directory

   
## Cite

Please cite our paper if you use this code in your own work:

```BibTeX
@inproceedings{tran2023attention,
  title={Attention Mixtures for Time-Aware Sequential Recommendation},
  author={Tran, Viet-Anh and Salha-Galvan, Guillaume and Sguerra, Bruno and Hennequin, Romain},
  booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2023}
}
```
