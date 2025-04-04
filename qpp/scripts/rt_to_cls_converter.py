import pandas as pd
import sys

def snap_lat2onehotv2(lat):
    if lat < 1:
        return 0
    elif (1 < lat) and (lat < 10):
        return 1
    elif 10 < lat:
        return 2

ip = sys.argv[1]
op = sys.argv[2]
pred_field = sys.argv[3]

df = pd.read_csv(ip)
df['time_cls'] = df['time'].apply(snap_lat2onehotv2)
df[pred_field] = df[pred_field].apply(snap_lat2onehotv2)
col = ['id','time_cls',pred_field]
for c in df.columns:
    if c not in col:
        col.append(c)
df = df[col]
df.to_csv(op,index=False)
