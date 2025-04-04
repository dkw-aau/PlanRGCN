#from graph_construction.feats.feature_binner import FeaturizerBinning
from graph_construction.feats.featurizer_path import FeaturizerPath
from trainer.train_ray import main
#from graph_construction.feats.featurizer import FeaturizerPredCoEnt
from graph_construction.query_graph import (
    QueryPlan,
    QueryPlanCommonBi,
    snap_lat2onehot,
    snap_lat2onehotv2,
)
from graph_construction.qp.query_plan_path import QueryPlanPath
#from graph_construction.qp.query_plan_lit import QueryPlanLit
from ray import tune
import os

# Results save path
#path_to_save = "/PlanRGCN/temp_results"
# Dataset split paths
#train_path = "/qpp/dataset/DBpedia2016_sample_0_1_10/train_sampled.tsv"
#val_path = "/qpp/dataset/DBpedia2016_sample_0_1_10/val_sampled.tsv"
#test_path = "/qpp/dataset/DBpedia2016_sample_0_1_10/test_sampled.tsv"
#qp_path = "/qpp/dataset/DBpedia2016_sample_0_1_10/queryplans/"


sample_name = "DBpedia2016_v2"  # balanced dataset
sample_name = "DBpedia2016_v2_aug"  # Previously best

sample_name = "DBpedia2016_0_1_10_weight_loss"
sample_name = "DBpedia2016_0_1_10_path_v3_weight_loss"
sample_name = "DBpedia2016_0_1_10_weight_loss_retrain"

train_path = f"/data/{sample_name}/train_sampled.tsv"
val_path = f"/data/{sample_name}/val_sampled.tsv"
test_path = f"/data/{sample_name}/test_sampled.tsv"
qp_path = f"/data/{sample_name}/queryplans/"


# KG statistics feature paths
pred_stat_path = (
    "/data/planrgcn_feat/extracted_features_dbpedia2016/predicate/pred_stat/batches_response_stats"
)
pred_com_path = "/PlanRGCN/data/dbpedia2016/predicate/pred_co"

ent_path = (
    "/data/planrgcn_feat/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats"
)

lit_path= (
    "/data/planrgcn_feat/extracted_features_dbpedia2016/literals_stat/batches_response_stats"
)
# Training Configurations
num_samples = 22  # 4  # use this
num_cpus= 22
max_num_epochs = 100
query_plan_dir = qp_path
time_col = "mean_latency"
is_lsq = True
cls_func = snap_lat2onehotv2
#featurizer_class = FeaturizerBinning
featurizer_class = FeaturizerPath
#scaling = "robust"
# scaling = "std"
scaling = "binner"
n_classes = 3
#query_plan = QueryPlanLit
query_plan = QueryPlanPath
prepper = None
resume=False

# Results save path
#path_to_save = f"/data/{sample_name}/planrgcn_{scaling}"
#if lit_path is not None:
#    path_to_save += "_litplan"
path_to_save = f"/data/DBpedia2016_0_1_10_weight_loss_retrain/plargcn_results_17_3"

os.makedirs(path_to_save, exist_ok=True)


config = {
    "l1": tune.choice([512, 256, 1024, 2048, 4096]),
    "l2": tune.choice([512, 256, 1024, 4096]),
    "dropout": tune.choice([0.0, 0.6]),
    "wd": 0.01,
    "lr": tune.grid_search([1e-5]),
    "epochs": 100,
    "batch_size": tune.choice([128, 256, 512]),
    "loss_type": tune.choice(["cross-entropy","mse"]),
    "pred_com_path": tune.choice(["pred2index_louvain.pickle"]),
}
config = {
    "l1": tune.choice([ 1024, 2048, 4096]),
    "l2": tune.choice([ 512, 1024, 2048, 4096,8192]),
    "dropout": tune.choice([0.0, 0.6]),
    "wd": 0.01,
    "lr": tune.grid_search([1e-5]),
    "epochs": 100,
    "batch_size": tune.choice([ 256]),
    "loss_type": tune.choice(["cross-entropy","mse"]),
    "pred_com_path": tune.choice(
        [ "pred2index_louvain.pickle"]
    ),
}

def earlystopDBpedia(trial_id: str, result: dict) -> bool:
    """This function should return true when the trial should be stopped and false for continued training.

    Args:
        trial_id (str): _description_
        result (dict): _description_

    Returns:
        bool: whether to stop trial or not
    """
    if result["val f1"] < 0.7 and result["training_iteration"] >= 50:
        return True
    if result["val f1"] < 0.5 and result["training_iteration"] >= 10:
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
    earlystop=earlystopDBpedia,
)
