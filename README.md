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



