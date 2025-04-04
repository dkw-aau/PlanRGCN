# Baseline feature generation
## FIrst step
If file with all queries have not been created, then do this first by:

```
cat train_sampled.tsv >> all.tsv
cat val_sampled.tsv >> all.tsv
cat test_sampled.tsv >> all.tsv
```
THen remove the duplicate headers using a text editor of choic

## Algebra feature generations:
```
python3 -m qpp_features.feature_generator all.tsv /data/wikidata_0_1_10_v2_weight_loss -t alg_feat
```
