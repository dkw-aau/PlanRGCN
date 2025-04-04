import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
from load_balance import const
from  load_balance.workload.workload import Workload
from load_balance.workload.arrival_time import ArrivalRateDecider

def get_workload(query_file, predicted_file, add_lsq_url,cls_field, mu = const.MU, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    w = Workload(true_field_name=cls_field)
    w.load_queries(query_file)
    w.set_time_cls(predicted_file,add_lsq_url=add_lsq_url)
    a = ArrivalRateDecider(seed=seed)
    w.shuffle_queries()
    w.shuffle_queries()
    w.set_arrival_times(a.assign_arrival_rate(w, mu= mu))
    return w



if __name__ == "__main__":
    # timeout -s 2 7200 python3 -m load_balance.main_balancer qpp -d wikidata_0_1_10_v3_path_weight_loss -b planrgcn_binner_litplan -t planrgcn_prediction -o /tmp/test -r 44 -u http://172.21.233.14:8891/sparql -f 4 -m 4 -s 2 -i 10 --seed 42

    parser = argparse.ArgumentParser(
        prog='Workload Generator',
        description='Workload generator for test dataset workload',
        epilog='')

    parser.add_argument('-f', '--query_file')
    parser.add_argument('-p', '--prediction_file')
    parser.add_argument('-t', '--true_field_name')
    parser.add_argument('-o', '--save_dir')
    parser.add_argument('-l', '--add_lsq_url', default='yes')
    parser.add_argument('-r', '--MU', default=44, type=int)
    parser.add_argument('--interval', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    query_file = args.query_file
    predicted_file = args.prediction_file
    save_dir = args.save_dir
    cls_field = args.true_field_name
    add_lsq_url = True if args.add_lsq_url.lower() == 'yes' else False
    MU = int(args.MU)

    os.makedirs(Path(save_dir), exist_ok=True)
    Path(save_dir)

    w = get_workload(query_file, predicted_file, add_lsq_url, cls_field, mu=MU, seed=args.seed)
    with open(os.path.join(save_dir, "workload.pck"), 'wb') as wf:
        w.pickle(wf)