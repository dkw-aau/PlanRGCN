from functools import partial
import os
import tempfile
import dgl
from graph_construction.feats.featurizer import FeaturizerPredCo, FeaturizerPredCoEnt
from graph_construction.feats.featurizer_path import FeaturizerPath
from graph_construction.feats.feature_binner import FeaturizerBinning
from graph_construction.query_graph import (
        QueryPlan,
        QueryPlanCommonBi,
        snap_lat2onehot,
        snap_lat2onehotv2,
)
from trainer.train_ray import get_dataloaders, predict
from graph_construction.qp.query_plan_path import QueryPlanPath
import ray
from trainer.data_util import DatasetPrep
from trainer.model import Classifier2RGCN
import torch as th
import numpy as np
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from pathlib import Path
from ray import tune, train

# from ray.air import Checkpoint, session
from ray.train import Checkpoint

path_check = "/PlanRGCN/wikidata_0_1_10_v2_path_hybrid/ray_save/train_function_2023-11-25_09-21-11/train_function_f9110_00004_4_batch_size=128,dropout=0.0000,l1=4096,l2=2048,loss_type=cross-entropy,lr=0.0000,pred_com_path=pred2in_2023-11-25_09-21-12/checkpoint_000049/checkpoint.pt"
data = {
        "checkpoint":path_check,
}
sample_name = "wikidata_0_1_10_v2_path_hybrid"
data["train_path"]=f"/qpp/dataset/{sample_name}/train_sampled.tsv"
data["val_path"]=f"/qpp/dataset/{sample_name}/val_sampled.tsv"
data["test_path"]=f"/qpp/dataset/{sample_name}/test_sampled.tsv"
data["batch_size"]=128
data["query_plan_dir"]=f"/qpp/dataset/{sample_name}/queryplans/"
data["pred_stat_path"]="/PlanRGCN/extracted_features_wd/predicate/pred_stat/batches_response_stats"
data["pred_com_path"]="/PlanRGCN/extracted_features_wd/predicate/pred_co"
data["ent_path"]="/PlanRGCN/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats"
data["time_col"]="mean_latency"
data["is_lsq"]=True
data["cls_func"]=snap_lat2onehotv2
data["featurizer_class"]=FeaturizerPath
data["scaling"]="robust"
data["query_plan"] = QueryPlanPath
data["inputd"] = 432
data['l1'] = 4096
data['l2']= 1024
data['dropout'] = 0.0
data['n_classes'] = 3
data["path_to_save"]= f"/PlanRGCN/{sample_name}" 
def run_trained_model(data):
    model_state = th.load(data["checkpoint"])
    train, val, test , input_d, _ = get_dataloaders(train_path = data["train_path"],
            val_path = data["val_path"],
            test_path= data["test_path"],
            batch_size=data["batch_size"],
            query_plan_dir=data["query_plan_dir"],
            pred_stat_path= data["pred_stat_path"],
            pred_com_path = os.path.join(data["pred_com_path"],"pred2index_louvain.pickle"),
            ent_path=data["ent_path"],
            time_col=data["time_col"],
            is_lsq=data["is_lsq"],
            cls_func=data["cls_func"],
            featurizer_class=data["featurizer_class"],
            scaling=data["scaling"],
            query_plan=data["query_plan"],
    )
    model = Classifier2RGCN(data["inputd"], data['l1'], data['l2'], data['dropout'], data['n_classes'])
    print(QueryPlan.max_relations)
    predict(model, train, val, test, True, path_to_save=data["path_to_save"])
run_trained_model(data)
