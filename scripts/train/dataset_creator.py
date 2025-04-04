


from trainer.train_ray import get_dataloaders
from graph_construction.feats.featurizer_path import FeaturizerPath
from graph_construction.qp.query_plan_path import QueryPlanPath, QueryGraph
from ray import tune
import os,sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with a specific configuration.")

    # Required arguments
    parser.add_argument('sample_name', type=str, help="Name of the dataset to use.")
    parser.add_argument('path_to_save', type=str, help="Path to save all artifacts.")

    # Optional arguments
    parser.add_argument('--feat_path', type=str, default='/data/planrgcn_feat/extracted_features_dbpedia2016', help="Path to metaKG statistics used for the model.")
    parser.add_argument('--use_pred_co', type=str, default="yes", help="whether to use predicat co-occurence features.")
    parser.add_argument('--class_path', type=str, default=None, help="path that defined 'n_classes' and 'cls_func' for prediction objective")
    #parser.add_argument('--bin', type=int, default=50, help="number of bins to create for each feature, bin size")


    return parser.parse_args()

args = parse_args()
sample_name = args.sample_name
path_to_save =  args.path_to_save
feat_base_path=args.feat_path
use_pred_co = args.use_pred_co
print(args)

if not (args.class_path == None or args.class_path == "None"):
    exec(open(args.class_path).read(), globals())
if not 'cls_func' in globals():
    from graph_construction.query_graph import snap_lat2onehotv2
    cls_func = snap_lat2onehotv2
    n_classes = 3
# sample_name = "wikidata_3_class_full"
# path_to_save = "/tmp/test2"
# feat_base_path="/data/metaKGStat/wikidata"
# use_pred_co = "no"
batch_size=256
train_path = f"/data/{sample_name}/train_sampled.tsv"
val_path = f"/data/{sample_name}/val_sampled.tsv"
test_path = f"/data/{sample_name}/test_sampled.tsv"
qp_path = f"/data/{sample_name}/queryplans/"
save_prep_path=os.path.join(path_to_save,'prepper.pcl')
"""if os.path.exists(save_prep_path):
    print('prepeper already exists. Do you want to continue recreating prepper? (y/n)')
    x = input()
    if x == 'n':
        exit()
"""
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
#featurizer_class = FeaturizerBinning
featurizer_class = FeaturizerPath
scaling = "binner"
query_plan = QueryGraph
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
    batch_size=batch_size,
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
