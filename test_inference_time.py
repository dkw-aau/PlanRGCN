
from graph_construction.feats.featurizer_path import FeaturizerPath
from graph_construction.qp.query_plan_path import QueryPlanPath
import time, json

from trainer.model import Classifier2RGCN
import pandas as pd
import os
from graph_construction.nodes.PathComplexException import PathComplexException
def get_qp_files(path):
    df = pd.read_csv(f'{path}/test_sampled.tsv', sep='\t')
    test_ids = list(df['id'].apply(lambda x : x[20:]))
    qp_path = f'{path}/queryplans'
    test_files = [(f"{qp_path}/{x}",x) for x in os.listdir(qp_path) if not '.' in x and x in test_ids]
    return test_files

def wikidata(path ='/data/wikidata_0_1_10_v3_path_weight_loss'):
    test_f = get_qp_files(path)
    inputd = 415
    l1 = 2048
    l2 = 4096
    dropout = 0.0
    n = 3
    
    model = Classifier2RGCN(inputd, l1, l2, dropout, n)
    pred_stat_path = (
        "/data/planrgcn_features/extracted_features_wd/predicate/pred_stat/batches_response_stats"
    )
    pred_com_path = "/data/planrgcn_features/extracted_features_wd/predicate/pred_co/pred2index_louvain.pickle"

    ent_path = (
        "/data/planrgcn_features/extracted_features_wd/entities/ent_stat/batches_response_stats"
    )
    lit_path =(
            "/data/planrgcn_features/extracted_features_wd/literals_stat/batches_response_stats"
    )
    feat_start = time.time()
    featurizer = FeaturizerPath(pred_stat_path=pred_stat_path,
            pred_com_path=pred_com_path,
            ent_path=ent_path,
            lit_path = lit_path,
            scaling="binner")
    feat_duration = time.time()-feat_start
    print(f'feature loading time {feat_duration}')
    
    write_inference_time(test_f, featurizer, path, model)

def dbpedia(path ='/data/DBpedia2016_0_1_10_path_v3_weight_loss'):
    test_f = get_qp_files(path)
    inputd = 520
    l1 = 4096
    l2 = 1024
    dropout = 0.0
    n = 3
    
    model = Classifier2RGCN(inputd, l1, l2, dropout, n)
    
    pred_stat_path = (
        "/data/planrgcn_features/extracted_features_dbpedia2016/predicate/pred_stat/batches_response_stats"
    )
    pred_com_path = "/data/planrgcn_features/extracted_features_dbpedia2016/predicate/pred_co/pred2index_louvain.pickle"

    ent_path = (
        "/data/planrgcn_features/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats"
    )
    lit_path =(
            "/data/planrgcn_features/extracted_features_dbpedia2016/literals_stat/batches_response_stats"
    )
    feat_start = time.time()
    featurizer = FeaturizerPath(pred_stat_path=pred_stat_path,
            pred_com_path=pred_com_path,
            ent_path=ent_path,
            lit_path = lit_path,
            scaling="binner")
    feat_duration = time.time()-feat_start
    print(f'feature loading time {feat_duration}')
    write_inference_time(test_f, featurizer, path, model)
    
    
def write_inference_time(test_f, featurizer, path, model):
    ids = []
    times = []
    failed_paths = []
    overall_start = time.time()
    for item_no,(qp_path, id) in enumerate(test_f):
        if (item_no % 50 == 0):
            print(f'Currently at {item_no} of {len(test_f)}: time {time.time()-overall_start}')
        try:
            data = json.load(open(qp_path,'r'))
            qp=QueryPlanPath(data)
            starttime = time.time()
            qp.feature(featurizer)
            dgl_graph = qp.to_dgl()
            feats = dgl_graph.ndata["node_features"]
            edge_types = dgl_graph.edata["rel_type"]
            pred = model(dgl_graph, feats, edge_types)
            endtime= time.time()
            times.append(endtime-starttime)
            ids.append(id)
        except PathComplexException:
            failed_paths.append(qp_path)
    df = pd.DataFrame({'id':ids,'time':times})
    df.to_csv(f"{path}/test_model_inference_times.csv" ,index=False)
    print(f"Finished after {time.time()-overall_start}")
    print(failed_paths)

exp_type = 'wikidata'
exp_type = 'dbpedia'
match exp_type:
    case 'wikidata':
        wikidata()
    case 'dbpedia':
        dbpedia()