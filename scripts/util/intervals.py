import sys
import pandas as pd
import os

path = sys.argv[1]
def interval(x):
    if x <=1:
        return 0
    if x <= 10:
        return 1
    return 2

if not os.path.exists(path):
    print("path does not exist! "+path)
for d in ["train_sampled.tsv", "val_sampled.tsv", "test_sampled.tsv"]:
    df = pd.read_csv(os.path.join(path, d), sep='\t')
    df['cls'] = df['mean_latency'].apply(interval)
    print('-'*10)
    print(d)
    print(df['cls'].value_counts())
    print('-'*10)
