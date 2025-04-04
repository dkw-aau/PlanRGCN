import os

from pathlib import Path

if 'QG_JAR' not in os.environ.keys():
    os.environ['QG_JAR']='/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'

if 'QPP_JAR' not in os.environ.keys():
    os.environ['QPP_JAR']='/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'

import pickle
import sys
from graph_construction.feats.featurizer_path import FeaturizerPath
from trainer.train_ray import main
from graph_construction.query_graph import (
    snap_lat2onehotv2,
)
from graph_construction.qp.query_plan_path import QueryPlanPath,QueryGraph
from ray import tune
import os

sample_name = sys.argv[1]
path_to_save = sys.argv[2]
train_path = f"/data/{sample_name}/train_sampled.tsv"
val_path = f"/data/{sample_name}/val_sampled.tsv"
test_path = f"/data/{sample_name}/test_sampled.tsv"
qp_path = f"/data/{sample_name}/queryplans/"
save_prep_path=f'{path_to_save}/prepper.pcl'
feat_base_path='/data/planrgcn_feat/extracted_features_dbpedia2016'
feat_base_path=sys.argv[3]

# KG statistics feature paths
pred_stat_path = (
    f"{feat_base_path}/predicate/pred_stat/batches_response_stats"
)


pred_com_path = f"{feat_base_path}/predicate/pred_co"

ent_path = (
    f"{feat_base_path}/entity/ent_stat/batches_response_stats"
)

lit_path= (
    f"{feat_base_path}/literals/literals_stat/batches_response_stats"
)

# Training Configurations
num_samples = 22  # 4
num_cpus= 22
max_num_epochs = 100
query_plan_dir = qp_path
time_col = "mean_latency"
is_lsq = True
cls_func = snap_lat2onehotv2
#featurizer_class = FeaturizerBinning
featurizer_class = FeaturizerPath
scaling = "binner"
n_classes = 3
query_plan = QueryPlanPath

if not os.path.exists(save_prep_path):
    print("Please create datasets first!")
    exit()
with open(save_prep_path, 'rb') as prepf:
    prepper = pickle.load(prepf)

resume=False


os.makedirs(path_to_save, exist_ok=True)

config = {
    "l1": tune.choice([ 1024, 2048, 4096]),
    "l2": tune.choice([ 512, 1024, 2048, 4096]),
    "dropout": tune.choice([0.0, 0.6]),
    "wd": 0.01,
    "lr": tune.grid_search([1e-5]),
    "epochs": 100,
    "batch_size": tune.choice([ 256]),
    "loss_type": tune.choice(["cross-entropy"]),
    "pred_com_path": tune.choice(
        [ "pred2index_louvain.pickle"]
    ),
}


import numpy as np

def earlystop(trial_id: str, result: dict) -> bool:
    """This function should return true when the trial should be stopped and false for continued training.

    Args:
        trial_id (str): _description_
        result (dict): _description_

    Returns:
        bool: _description_
    """
    if result["val_f1"] < 0.7 and result["training_iteration"] >= 50:
        return True
    if result["val_f1"] < 0.5 and result["training_iteration"] >= 10:
        return True
    #l_n = len(result["val_f1_lst"])
    #l = np.sum(np.diff(result["val_f1_lst"]))/l_n
    l = result["val_f1_lst"][:-1]
    #if improvement in last patience epochs is less than 1% in validation loss then terminate trial.
    if result["training_iteration"] >= 10 and np.max(l) >= result["val_f1"]:
        return True
    return False

main(
    num_samples=num_samples,
    max_num_epochs=max_num_epochs,
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    query_plan_dir=qp_path,
    pred_stat_path=pred_stat_path,
    pred_com_path=pred_com_path,
    ent_path=ent_path,
    lit_path=lit_path,
    time_col=time_col,
    is_lsq=is_lsq,
    cls_func=cls_func,
    featurizer_class=featurizer_class,
    scaling=scaling,
    n_classes=n_classes,
    query_plan=query_plan,
    path_to_save=path_to_save,
    config=config,
    resume=resume,
    num_cpus=num_cpus,
    earlystop=earlystop,
    save_prep_path=save_prep_path,
    patience=5,
    prepper=None,
)
