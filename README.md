This code framework comes from an open source code repo: [facebookresearch/ssl-relation-prediction: Simple yet SoTA Knowledge Graph Embeddings. (github.com)](https://github.com/facebookresearch/ssl-relation-prediction). Thanks for their sharing.

We have added our method UniBi to it for testing its effectiveness on the ogb-biokg and ogb-wikikg2 datasets.

Running the code:

```bash
bash unibi_2.sh
nohup bash unibi_2.sh > unibi_2_rank3000_lr1e-1_regDURA_l5e-3.log &
```



