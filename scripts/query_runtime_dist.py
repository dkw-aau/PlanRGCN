import os, pandas as pd

os.environ['QG_JAR'] = '/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'
os.environ['QPP_JAR'] = '/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'



# We generated new queries for DBpedia queries
test_sampled = '/data/DBpedia_3_class_full/test_sampled.tsv'
unseen_queries = '/data/DBpedia_3_class_full/newUnseenQs4/queries.tsv'
newPPqueries = '/data/DBpedia_3_class_full/newPPs/queries.tsv'

test_df = pd.read_csv(test_sampled, sep='\t')
unseen_df = pd.read_csv(unseen_queries, sep='\t')
pp_df = pd.read_csv(newPPqueries, sep='\t')

# Output folder for new test set
output = '/data/DBpedia_3_class_full/newtestset'


#Create a new test sampled file
complete_test_df = pd.concat([test_df, unseen_df, pp_df])
complete_test_df.drop_duplicates(subset='queryID', keep='first', inplace=True) #inplace=True
test_save_path = os.path.join(output, 'test_sampled.tsv')
complete_test_df.to_csv(test_save_path, sep='\t')


#objective functions for different time intervals
import numpy as np
def five_int(lat):
    if lat < 0.004:
        return 0
    elif (0.004 < lat) and (lat <= 1):
        return 1
    elif (1 < lat) and (lat <= 10):
        return 2
    elif (10 < lat) and (lat <= 899):
        return 3
    elif 899 < lat:
        return 4

#percentile based prediction snapper for DBpedia
def d_quantileInt(lat):
    if lat <= 0.0352054542551437:
        return 0
    elif (0.0352054542551437 < lat) and (lat <= 50.830727839345734):
        return 1
    elif (50.830727839345734 < lat):
        return 2

from collections import Counter
dict(Counter(list(complete_test_df['mean_latency'].apply(d_quantileInt))))
d = dict(Counter(list(complete_test_df['mean_latency'].apply(five_int))))
for i in range(5):
    print(f"{i+1}: {d[i]}")


from query_class_analyzer import QueryClassAnalyzer, SemiFineGrainedQueryClassAnalyzer
base = '/data/DBpedia_3_class_full/'
SemiFineGrainedQueryClassAnalyzer(f"{base}/train_sampled.tsv",
                   f"{base}/val_sampled.tsv",
                   test_save_path,
                   objective_file='/data/wikidata_3_class_full/plan01_n_gq/objective.py')

# for quantile /data/DBpediaV2/objective.py

#for 5 intervals