import pandas as pd
from analysis.util import analyse_workload
import os, sys
import numpy as np
#replacedDistinct = '/SPARQLBench/wikidata_queries/distinctRemovedWikidata/bench'
replacedDistinct = '/SPARQLBench/dbpedia_queries/distinctRemovedDBpedia/bench/'
distinctRemovedFiles =[f'{replacedDistinct}/{x}' for x in os.listdir(replacedDistinct)]
#print(distinctRemovedFiles)
analyse_workload(paths=distinctRemovedFiles, path_to_analysis='/tmp')
bench = pd.read_csv('/tmp/latency_log.tsv', sep='\t')
bench =bench.set_index('id')

split_path= '/data/DBpedia_3_class_full'
#split_path= '/data/wikidata_3_class_full'
first = True
for ori, new in zip(['ori_test_sampled.tsv','ori_val_sampled.tsv','ori_train_sampled.tsv'],['test_sampled.tsv','val_sampled.tsv','train_sampled.tsv']):
    if not os.path.exists(os.path.join(split_path,ori)):
        continue
    df = pd.read_csv(os.path.join(split_path,ori), sep='\t')
    df = df.set_index('id')
    if first:
        for c in df.columns:
            if c not in bench.columns:
                bench[c] = np.nan
        first = False
        bench = bench[df.columns]
    c = 0
    for idx, row in bench.iterrows():
        if idx in df.index:
            c +=1
            df.loc[idx] = row
    print(c)
    df.reset_index(inplace=True)
    df['queryString'] = df['query_string_2']
    df['queryID'] = df['id']
    
    df.to_csv(os.path.join(split_path,new), sep='\t', index=False)

