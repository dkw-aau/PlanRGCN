import os
import pandas as pd
import numpy as np

algebra_feats = "data/test/algebra_features.txt"
dist_folder = "data/dists"

def create_sing_dist_file(path):
    files = os.listdir(path)
    files = [x for x in files if x.endswith('.csv') and x.startswith('disthungarian_distance')]
    files = sorted(files)
    idx, distances, execution_times = [], [], []
    for i in range(len(files)):
        with open(path + '/' + files[i], 'r') as f:
            for line in f.readlines():
                n_l = line[:-1]
                spl = n_l.split(',')
                if len(spl) < 1:
                    continue
                idx.append(spl[0])
                execution_times.append(float(spl[1]))
                distances.append(spl[2:])
    return idx, np.array(distances, dtype=np.double), np.array(execution_times, dtype=np.double)

ids, distances, execution_times = create_sing_dist_file(dist_folder)
df = pd.read_csv(algebra_feats)
