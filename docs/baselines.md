# Query Performance Predictions
Existing (baseline) Query Performance Prediction for SPARQL queries

## Dataset Queries
To extract queries, run the extract_queries.sh script in the scripts folder.
Prerequisites for this are maven and java 17 and a running endpoint with the LSQ dump files loaded in. (see link for used LSQ dump: https://github.com/Abiram98/LSQDBpediaLogs)

** The final query log for DBpedia is contained in dataset/queries2015_2016_clean.tsv**


## Feature generation
Be aware that the distance matrix computation can be time consuming, even with the our highly parallelized code.
```
DATASET=/data/dataset_debug
DISTANCE=/data/dataset_debug/distances
bash /PlanRGCN/qpp/scripts/baseline_feat_const.sh $DATASET $DISTANCE > feature.log
```

## Model Training
For SVM
````
python3 -m qpp_new.trainer svm  --data-dir $DATASET --results-dir $DATASET
python3 /PlanRGCN/qpp/scripts/rt_to_cls_converter.py $DATASET/svm/k25/test_pred.csv $DATASET/svm/test_pred.csv svm_prediction
```
For NN
````
python3 -m qpp_new.trainer nn  --data-dir $DATASET --results-dir $DATASET
python3 /PlanRGCN/qpp/scripts/rt_to_cls_converter.py $DATASET/nn/k25/test_pred.csv $DATASET/nn/test_pred.csv nn_prediction
```