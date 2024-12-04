# SAPID

This project is a PyTorch implementation of SAPID: Sequentially Diversified and Accurate Recommendations in Chronological Order for a Series of Users (WSDM 2025).


## Prerequisites
- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Scipy](https://scipy.org)
- [Click](https://click.palletsprojects.com/en/7.x/)
- [tqdm](https://tqdm.github.io/)


## Datasets
We provide 4 datasets in this project: Gowalla, Amazon-electronics, Amazon-home, and Ml-1m.
We include the preprocessed datasets in the repository: `data/{data_name}`.
Use git lfs with the following scripts after cloning this repository to download datasets.
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs
git lfs pull
```

We also provide a [direct link](https://drive.google.com/drive/folders/1_-zf7tEb0WwHFSB_u7JneElz899ag0n0?usp=sharing) to download all datasets. 


## Base model
We provide 2 most widely used sequential recommendation models: SASRec and GRU4Rec.
We adopt the implementation of [pmixer](https://github.com/pmixer/SASRec.pytorch) for SASRec,
and the implementation of [PatrickSVM](https://github.com/PatrickSVM/Session-Based-Recommender-Models/) for GRU4Rec.
The implemented models are given in `model.py` and pretrained parameters are given in `model/`.

## Usage

### Training phase
You can run the training code by `python train.py` with arguments `--base`, `--data`, `--epochs`, and `--alpha`.
The role of each argument is as follows:

* `--data`: choose a dataset
* `--base`: choose a base model
* `--epochs`: choose the number of epochs
* `--alpha`: choose a debiasing hyperparameter $\alpha$

### Recommendation phase
You can run the evaluation code by `python eval.py` with arguments `--base`, `--data`, `--epochs`, `--alpha`, `--candn`, `--path`, and `--k`.
The role of each argument is as follows:

* `--data`: choose a dataset
* `--base`: choose a base model
* `--epochs`: choose the number of epochs
* `--alpha`: choose a debiasing hyperparameter $\alpha$
* `--candn`: choose a size $c$ of an initial candidate pool
* `--path`: choose a path of pre-trained model (optional) 
* `--k`: choose the length of a recommendation list (default: 10)

## Reference
Please cite the following paper if you use this code.
~~~
@inproceedings{SAPID,
  author    = {Jongjin Kim and
               U Kang},
  title     = {Sequentially Diversified and Accurate Recommendations in Chronological Order for a Series of Users},
  booktitle = {The 18th ACM International Conference on Web Search and Data Mining (WSDM)},
  year      = {2025}
}
~~~