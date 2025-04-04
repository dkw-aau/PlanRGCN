"""This is a module for creating embeddings of query plans based on a trained model.
"""
import pickle
from graph_construction.feats.featurizer_path import FeaturizerPath
from graph_construction.qp.query_plan import QueryPlan
from graph_construction.qp.query_plan_path import QueryPlanPath
from graph_construction.query_graph import snap_lat2onehotv2, snap_reg
import torch as th
from trainer.model import Classifier2RGCN
from trainer.train_ray import get_dataloaders
import os
from pathlib import Path

sample_name = "wikidata_0_1_10_v2_path_hybrid"
config = {
    "train_path" : f"/data/{sample_name}/train_sampled.tsv",
"val_path" : f"/data/{sample_name}/val_sampled.tsv",
"test_path" : f"/data/{sample_name}/test_sampled.tsv",
    "reg_train_path" : f"/data/{sample_name}/reg_train_sampled.pickle",
"reg_val_path" : f"/data/{sample_name}/reg_val_sampled.pickle",
"reg_test_path" : f"/data/{sample_name}/reg_test_sampled.pickle",
"qp_path" : f"/data/{sample_name}/queryplans/",
"pred_stat_path" : (
    "/data/extracted_features_wd/predicate/pred_stat/batches_response_stats"
),
"pred_com_path" : "/data/extracted_features_wd/predicate/pred_co/pred2index_louvain.pickle",
"ent_path" : (
    "/data/extracted_features_wd/entities/ent_stat/batches_response_stats"
),
"time_col" : "mean_latency",
#"cls_func" : snap_lat2onehotv2,
"cls_func" : snap_reg,
"featurizer_class" : FeaturizerPath,
"scaling" : "binner",
"inputd" : 314,
"l1" : 4096,
"l2" : 1024,
"dropout": 0.0,
"batch_size": 128,
"nclasses":3,
"best_checkpoint": "/PlanRGCN/wikidata_0_1_10_v2_path_hybrid/ray_save/train_function_2023-11-27_15-08-31/train_function_d3a12_00007_7_batch_size=128,dropout=0.0000,l1=4096,l2=1024,loss_type=cross-entropy,lr=0.0000,pred_com_path=pred2in_2023-11-27_15-08-31/checkpoint_000049/checkpoint.pt",
"class_name": Classifier2RGCN,
"query_plan" : QueryPlanPath,
"max_relations": 13
}

def script(config:dict):
    QueryPlan.max_relations = config["max_relations"]
    model = config["class_name"](config["inputd"],config['l1'],config['l2'],config['dropout'],config["nclasses"])
    model_state = th.load(config["best_checkpoint"])
    model.load_state_dict(model_state["model_state"])
    
    train_loader, val_loader, test_loader, input_d = get_dataloaders(
        train_path=config["train_path"],
        val_path=config["val_path"],
        test_path=config["test_path"],
        batch_size=config["batch_size"],
        query_plan_dir=config["qp_path"],
        pred_stat_path=config["pred_stat_path"],
        pred_com_path=config["pred_com_path"],
        ent_path=config["ent_path"],
        time_col=config["time_col"],
        is_lsq=True,
        cls_func=config["cls_func"],
        featurizer_class=config["featurizer_class"],
        scaling=config["scaling"],
        query_plan=config["query_plan"],
        debug=False
    )
    train_data = get_embed(model, train_loader)
    save_embedding(config['reg_train_path'], train_data)
    
    val_data = get_embed(model, val_loader)
    save_embedding(config['reg_val_path'], val_data)
    
    test_data = get_embed(model, test_loader)
    save_embedding(config['reg_test_path'], test_data)
    
def get_embed(trained_model, dataloader):
    embeddings = {'queryID': [], 'embeddings': [], "labels":[]}
    with th.no_grad():
        for _, (graphs, labels,ids) in enumerate(dataloader):
            feats = graphs.ndata["node_features"]
            edge_types = graphs.edata["rel_type"]
            logits = trained_model.get_last_hidden_layer(graphs, feats, edge_types)
            n_logits = logits.numpy()
            embeddings["embeddings"].extend(n_logits)
            embeddings["queryID"].extend(ids)
            embeddings["labels"].extend(labels.numpy())
            assert len(embeddings["queryID"]) == len(embeddings["labels"]) and len(embeddings["queryID"]) == len(embeddings["embeddings"])
    return embeddings

def save_embedding(path, data):
    os.makedirs(Path(path).parent.absolute(), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data,f)
def load_embedding_data(path):
    """return data for downstream latency prediction task

    Args:
        path (str): path to pickled data

    Raises:
        Exception: _description_

    Returns:
        list(str): query ids
        list(np.array): embeddings
        list(np.array[int]): labels/latencies for queries
    """
    if not os.path.exists(path):
        raise Exception(f"path {path} does not exist!")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data['queryID'],data['embeddings'],data['labels']
if __name__ == "__main__":
    script(config)
    exit()
    _ ,embd, _ = load_embedding_data(config['reg_train_path'])
    print(embd)