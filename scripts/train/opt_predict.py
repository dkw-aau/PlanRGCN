
from graph_construction.query_graph import snap_lat2onehotv2
from trainer.train_ray import get_dataloaders
from graph_construction.feats.featurizer_path import FeaturizerPath
from graph_construction.qp.query_plan_path import QueryPlanPath
from ray import tune
import os

#sample_name = sys.argv[1]
#path_to_save = sys.argv[2]
#feat_base_path=sys.argv[3]
#use_pred_co = sys.argv[4]
sample_name = "wikidata_3_class_full"
path_to_save = "/tmp/test2"
feat_base_path="/data/metaKGStat/wikidata"
use_pred_co = "no"
train_path = f"/data/{sample_name}/train_sampled.tsv"
val_path = f"/data/{sample_name}/val_sampled.tsv"
test_path = f"/data/{sample_name}/test_sampled.tsv"
qp_path = f"/data/{sample_name}/queryplans/"
save_prep_path=f'{path_to_save}/prepper.pcl'


# KG statistics feature paths
pred_stat_path = (
    f"{feat_base_path}/predicate/pred_stat/batches_response_stats"
)

if 'no' in use_pred_co:
    print('no predcom path')
    pred_com_path = None
else:
    pred_com_path = f"{feat_base_path}/predicate/pred_co"

ent_path = (
    f"{feat_base_path}/entity/ent_stat/batches_response_stats"
)

lit_path= (
    f"{feat_base_path}/literals/literals_stat/batches_response_stats"
)

# Training Configurations
query_plan_dir = qp_path
time_col = "mean_latency"
is_lsq = True
cls_func = snap_lat2onehotv2
#featurizer_class = FeaturizerBinning
featurizer_class = FeaturizerPath
scaling = "binner"
n_classes = 3
query_plan = QueryPlanPath
prepper = None
resume=False

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

if pred_com_path != None:
    pred_com_path = os.path.join(pred_com_path,"pred2index_louvain.pickle") 

train_loader, val_loader, test_loader, input_d, val_pp_loader = get_dataloaders(
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    val_pp_path= None,
    batch_size=256,
    query_plan_dir=query_plan_dir,
    pred_stat_path=pred_stat_path,
    pred_com_path=pred_com_path,
    ent_path=ent_path,
    lit_path=lit_path,
    time_col=time_col,
    is_lsq=is_lsq,
    cls_func=cls_func,
    featurizer_class=featurizer_class,
    scaling=scaling,
    query_plan=query_plan,
    save_prep_path=save_prep_path,
    save_path= None,
    config=config
)
print(input_d)
