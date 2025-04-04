#!bin/bash
# first parameter path to folder with operator and graph pattern features.
# $1 = /qpp/dataset/
if ! [[ -f "$1/extra/extra.csv" ]]; then
    
    echo "Creating extra features for CLEI NN work"
    echo "creating together array"
    mvn -f /PlanRGCN/qpe/pom.xml install
    python3 -c """
import pandas as pd
df1 = pd.read_csv('$1/train_sampled.tsv',sep='\t')
df2 = pd.read_csv('$1/test_sampled.tsv',sep='\t')
df3 = pd.read_csv('$1/val_sampled.tsv',sep='\t')
df = pd.concat([df1,df3,df2])
df.to_csv('$1/all_queries.tsv', sep='\t', index=False)
"""
    mkdir -p "$1/extra"
    (cd ../qpp_features/sparql-query2vec/target && java -jar sparql-query2vec-0.0.1.jar extra "$1/all_queries.tsv" "$1/extra/extra.csv")
    (cd "$1/extra" && ls | grep -P ".*\.preduri\..*\.txt" | xargs -d"\n" rm)
fi


exit


(cd ../ && python3 '/qpp/qpp_models/models4LSQ.py' nn -d $1)
mkdir -p "$1/CLEI_NN"
(cd ../ && mv "$1/nn_train_pred.csv" "$1/CLEI_NN/train_pred.csv")
(cd ../ && mv "$1/nn_val_pred.csv" "$1/CLEI_NN/val_pred.csv")
(cd ../ && mv "$1/nn_test_pred.csv" "$1/CLEI_NN/test_pred.csv")

#(cd ../ && python3 '/qpp/qpp_models/models4LSQ.py' svm -d $1)

