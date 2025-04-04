


import os

from pathlib import Path

if 'QG_JAR' not in os.environ.keys():
    os.environ['QG_JAR']='/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'


if 'QPP_JAR' not in os.environ.keys():
    os.environ['QPP_JAR']='/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'


import sys

import torch
import json
from trainer.model import Classifier2RGCN

import pandas as pd

from graph_construction.jar_utils import get_query_graph
from graph_construction.query_graph import create_query_graph

from graph_construction.qp.query_plan_path import QueryGraph
import pickle as p
import time

import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for model inference")

    parser.add_argument('--prep_path', type=str, default='/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/prepper.pcl',
                        help='Path to the preparation file')
    parser.add_argument('--model_path', type=str, default='/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/best_model.pt',
                        help='Path to the model file')
    parser.add_argument('--config_path', type=str, default='/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/model_config.json',
                        help='Path to the model configuration file')
    parser.add_argument('--gpu', type=str, default='yes',
                        help='yes if use gpu otherwise will not use gpu. Default uses gpu')

    return parser.parse_args()

def cls_func(lat):
    vec = np.zeros(3)
    if lat < 1:
        vec[0] = 1
    elif (1 < lat) and (lat < 10):
        vec[1] = 1
    elif 10 < lat:
        vec[2] = 1
    return vec

def snap_pred(pred):
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(cls_func(pred), dtype=torch.float32)
    return torch.argmax(pred)

if __name__ == "__main__":
    args = get_args()

    # Save the parsed arguments into the same variable names
    prep_path = args.prep_path
    model_path = args.model_path
    config_path = args.config_path



    config = json.load(open(config_path, 'r'))

    inputd = config['input_d']
    l1 = config['l1']
    l2 = config['l2']
    n_classes = 3

    prepper = p.load(open(prep_path, 'rb'))
    featurizer = prepper.feat
    query_plan = QueryGraph

    def single_qg_con(query_text, featurizer, query_plan):
        try:
            query = query_text
            start = time.time()
            qg = create_query_graph(query, query_plan=query_plan)
            qg.feature(featurizer=featurizer)
            dgl_graph = qg.to_dgl()
            end = time.time()
            qg_time = end - start
            return dgl_graph, qg_time
        except Exception as e:
            return None


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_path != None:
        dct = torch.load(model_path)
        model = Classifier2RGCN(inputd, l1, l2, 0.0, n_classes)
        model.load_state_dict(dct['model_state'])
    else:
        model = Classifier2RGCN(inputd, l1, l2, 0.0, n_classes)

    model = model.to(device)
    model.eval()

    entries = []
    with torch.no_grad():
        try:
            while True:
                query_text = input("Please input your SPARQL query:\n")
                #query_text = row['queryString']
                data = single_qg_con(query_text, featurizer, query_plan)
                if data is None:
                    continue
                qg, dur = data

                feats = qg.ndata['node_features']
                edge_types = qg.edata['rel_type']
                feats = feats.to(device)
                edge_types = edge_types.to(device)
                qg = qg.to(device)
                start = time.time()
                pred = model(qg, feats, edge_types)
                inference_time = time.time() - start
                pred = snap_pred(pred)
                pred = pred.detach().to('cpu').tolist()
                print(f"Prediction: {pred}")
        except KeyboardInterrupt:
            print("\nProgram terminated by user.")