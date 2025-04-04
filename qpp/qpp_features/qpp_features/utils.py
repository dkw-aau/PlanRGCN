from qpp_features.database import DatabaseConnector
import os,json, json5
import pandas as pd

def get_ids(base_dir):
    ids = []
    df = pd.read_csv(f"{base_dir}/train_sampled.tsv", sep='\t')
    ids.extend(list(df['queryID']))
    df = pd.read_csv(f"{base_dir}/val_sampled.tsv", sep='\t')
    ids.extend(list(df['queryID']))
    df = pd.read_csv(f"{base_dir}/test_sampled.tsv", sep='\t')
    ids.extend(list(df['queryID']))
    return ids

def load_csv(path):
    df = pd.read_csv(path, names=['queryID1', 'queryID2','dist','time'])
    df['queryID1']= df['queryID1'].apply(lambda x: x[1:].replace("\"",""))
    df['queryID2']= df['queryID2'].apply(lambda x: x[1:].replace("\"",""))
    df['dist'] = df['dist'].apply(lambda x: x.replace("\"","").replace("\\",""))
    df['time'] = df['time'].apply(lambda x: x.replace("\"","").replace("\\",""))
    return df

def get_dists_files(dist_path):
	return [f"{dist_path}/{x}" for x in os.listdir(dist_path)]
def load_df(dist_files):
	dfs = list()
	for x in dists_files:
    	try:
        	dfs.append(load_csv(x))
    	except Exception as e:
        	print(e.with_traceback())
	return pd.concat(dfs)

