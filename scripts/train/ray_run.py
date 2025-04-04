import os

from pathlib import Path

if 'QG_JAR' not in os.environ.keys():
    os.environ['QG_JAR']='/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'

if 'QPP_JAR' not in os.environ.keys():
    os.environ['QPP_JAR']='/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'

from graph_construction.jar_utils import get_query_graph
import json
import pickle
import sys
from graph_construction.feats.featurizer_path import FeaturizerPath
from trainer.train_ray import main
from graph_construction.query_graph import (
    snap_lat2onehotv2,
)
from graph_construction.qp.query_plan_path import QueryPlanPath, QueryGraph
from ray import tune
import os
import argparse
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with a specific configuration.")

    # Required arguments
    parser.add_argument('sample_name', type=str, help="Name of the dataset to use.")
    parser.add_argument('path_to_save', type=str, help="Path to save all artifacts.")

    # Optional arguments
    parser.add_argument('--feat_path', type=str, default='/data/planrgcn_feat/extracted_features_dbpedia2016', help="Path to metaKG statistics used for the model.")
    parser.add_argument('--layer1_size', type=int, default=4096, help="Size of the first layer.")
    parser.add_argument('--layer2_size', type=int, default=4096, help="Size of the second layer.")
    parser.add_argument('--class_path', type=str, default=None, help="path that defined 'n_classes' and 'cls_func' for prediction objective")
    parser.add_argument('--use_pred_co', type=str, default="yes", help="whether to use predicat co-occurence features.")
    parser.add_argument('--conv_type', type=str, default='RGCN', help="the graph convolution operation to use")

    return parser.parse_args()

args = parse_args()
print("Sample Name:", args.sample_name)
print("Path to Save:", args.path_to_save)
if args.feat_path:
    print("Feature Path:", args.feat_path)
if args.layer1_size:
    print("Layer 1 Size:", args.layer1_size)
if args.layer2_size:
    print("Layer 2 Size:", args.layer2_size)
    print("Layer 1 Size:", args.layer1_size)
if args.conv_type:
    print("conv_type :", args.conv_type)

sample_name = args.sample_name
path_to_save = args.path_to_save
train_path = f"/data/{sample_name}/train_sampled.tsv"
val_path = f"/data/{sample_name}/val_sampled.tsv"
test_path = f"/data/{sample_name}/test_sampled.tsv"
qp_path = f"/data/{sample_name}/queryplans/"
save_prep_path=f'{path_to_save}/prepper.pcl'
feat_base_path=args.feat_path

if not (args.class_path == None or args.class_path == 'None'):
    exec(open(args.class_path).read(), globals())
if not 'cls_func' in globals():
    from graph_construction.query_graph import snap_lat2onehotv2
    cls_func = snap_lat2onehotv2
    n_classes = 3

### weight recalculation
def loss_weight_cal(train_path, output_path):
    import pandas as pd
    all = pd.read_csv(train_path, sep='\t')
    all['cls'] = all['mean_latency'].apply(lambda x: np.argmax(cls_func(x)))
    value_counts = all.cls.value_counts()
    classes = len(value_counts.keys())
    weights = []
    for k in value_counts.keys():
        weights.append(len(all)/(value_counts[k]*classes))
    with open(output_path, "w") as f:
        json.dump(weights, f)

loss_weight_cal(train_path, f"/data/{sample_name}/loss_weight.json")
os.system(f"python3 '/PlanRGCN/scripts/train/dataset_creator.py' {args.sample_name} {args.path_to_save} --feat_path {feat_base_path} --class_path {args.class_path} --use_pred_co {args.use_pred_co}")


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
num_samples = 1
num_cpus= 8
num_gpus = 1
max_num_epochs = 80
query_plan_dir = qp_path
time_col = "mean_latency"
is_lsq = True
#featurizer_class = FeaturizerBinning
featurizer_class = FeaturizerPath
scaling = "binner"
query_plan = QueryPlanPath
query_plan = QueryGraph

if not os.path.exists(save_prep_path):
    print("Please create datasets first!")
    exit()
with open(save_prep_path, 'rb') as prepf:
    prepper = pickle.load(prepf)

resume=False


os.makedirs(path_to_save, exist_ok=True)

config = {
    "l1": tune.grid_search([ args.layer1_size]),
    "l2": tune.grid_search([ args.layer2_size]),
    "dropout": tune.choice([0.0, 0.6]),
    "wd": 0.01,
    "lr": tune.grid_search([1e-5]),
    "epochs": 100,
    "batch_size": tune.grid_search([ 256]),
    "loss_type": tune.grid_search(["cross-entropy"]),
    "pred_com_path": tune.grid_search(
        [ "pred2index_louvain.pickle"]
    ),
    "conv_type" : args.conv_type,
}



def earlystop(trial_id: str, result: dict) -> bool:
    """This function should return true when the trial should be stopped and false for continued training.

    Args:
        trial_id (str): _description_
        result (dict): _description_

    Returns:
        bool: _description_
    """
    if result["training_iteration"] >= 10 and result["val_f1"] < 0.7 and result["training_iteration"] >= 50:
        return True
    #l_n = len(result["val_f1_lst"])
    #l = np.sum(np.diff(result["val_f1_lst"]))/l_n
    l = result["val_f1_lst"][:-1]
    #if improvement in last patience epochs is less than 1% in validation loss then terminate trial.
    if result["training_iteration"] >= 30 and np.min(l) >= result["val_f1"]:
        return True
    return False

def earlystop(trial_id: str, result: dict) -> bool:
    #l_n = len(result["val_f1_lst"])
    #l = np.sum(np.diff(result["val_f1_lst"]))/l_n
    l = result["val_f1_lst"][:-1]
    #if improvement in last patience epochs is less than 1% in validation loss then terminate trial.
    if result["training_iteration"] >= 30 and np.mean(l) >= result["val_f1"]:
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
    resources_per_trial={"cpu": num_cpus, "gpu" : num_gpus}
)
