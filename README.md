This code framework comes from an open source code repo: [facebookresearch/ssl-relation-prediction: Simple yet SoTA Knowledge Graph Embeddings. (github.com)](https://github.com/facebookresearch/ssl-relation-prediction). Thanks for their sharing.

We have added our method UniBi to `models.py` for testing UniBi's effectiveness on the ogb-biokg datasets.

## Load the data

Preprocess biokg dataset from LinkPropPredDataset.

```python
python preprocess_datasets.py
```

Please place the ogbl dataset in the data folder.

```bash
data
	--ogbl-biokg
		--meta_info.pickle
		--test.pickle
		--train.pickle
		--valid.pickle
```

## Run the code

```bash
bash unibi_3.sh
```

or

```bash
nohup bash unibi_3.sh > ./unibi3_biokg.log &
```



## Citation
If you make use of this code and UniBi model, please kindly cite the following paper:

```bib
@inproceedings{
chen2021relation,
title={Relation Prediction as an Auxiliary Training Objective for Improving Multi-Relational Graph Representations},
author={Yihong Chen and Pasquale Minervini and Sebastian Riedel and Pontus Stenetorp},
booktitle={3rd Conference on Automated Knowledge Base Construction},
year={2021},
url={https://openreview.net/forum?id=Qa3uS3H7-Le}
}
@article{DBLP:journals/corr/abs-2309-13834,
  author       = {Jiayi Li and
                  Ruilin Luo and
                  Jiaqi Sun and
                  Jing Xiao and
                  Yujiu Yang},
  title        = {Prior Bilinear Based Models for Knowledge Graph Completion},
  journal      = {CoRR},
  volume       = {abs/2309.13834},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2309.13834},
  doi          = {10.48550/ARXIV.2309.13834},
  eprinttype    = {arXiv},
  eprint       = {2309.13834},
  timestamp    = {Wed, 27 Sep 2023 16:51:35 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2309-13834.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
