import os
if 'QG_JAR' not in os.environ.keys():
    os.environ['QG_JAR']='/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'


if 'QPP_JAR' not in os.environ.keys():
    os.environ['QPP_JAR']='/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'

#Setup Python Path
import sys
sys.path.extend( ['/PlanRGCN/qpp/qpp_new/qpp_new',
 '/PlanRGCN',
 '/PlanRGCN/PlanRGCN/feature_extraction',
 '/PlanRGCN/PlanRGCN/feature_representation',
 '/PlanRGCN/PlanRGCN/graph_construction',
 '/PlanRGCN/PlanRGCN/trainer',
 '/PlanRGCN/PlanRGCN/query_log_splitting',
 '/PlanRGCN/inductive_query',
 '/PlanRGCN/qpp/qpp_features',
 '/PlanRGCN/load_balance',
 '/PlanRGCN/load_balance/load_balance',
 '/PlanRGCN/load_balance/load_balance/schedulers',
 '/PlanRGCN/load_balance/load_balance/workload',
 '/PlanRGCN/load_balance/load_balance/result_analysis',
 '/PlanRGCN/qpp/qpp_new',
 '/PlanRGCN/qpp/qpp_new/qpp_new',
 '/usr/lib/python310.zip',
 '/usr/lib/python3.10',
 '/usr/lib/python3.10/lib-dynload',
 '/usr/local/lib/python3.10/dist-packages',
 '/PlanRGCN/feat_con_time',
 '/usr/lib/python3/dist-packages'])


import pandas as pd
import graph_construction.jar_utils
import time

import numpy as np

import keras
import tensorflow.keras.backend as K
import argparse

#Defining these function are necessary to import NN models from tensorflow
@keras.utils.register_keras_serializable(package="functions")
def coeff_determination(y_true, y_pred):
    """Coefficient of determination to use in metrics calculated by epochs"""
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

@keras.utils.register_keras_serializable(package="functions")
def rmse(y_true, y_pred):
    """RMSE to use in metrics calculated by epochs"""
    return K.exp(K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)))


class SPARQLDistCalcular:
    def __init__(self):
        import jpype
        import jpype.imports
        from new_distance import GEDCalculator
        self.calc = GEDCalculator()


    def distance_ged(self, query1:str, query2:str):
        try:
            return self.calc.calculateDistance(query1,query2)
        except:
            return np.inf

import sys
#sys.path.append('/PlanRGCN/qpp/qpp_new')
from qpp_new.data_prep import NNDataPrepper, OriginalDataPrepper, SVMDataPrepper
from argparse import ArgumentParser
import os
from qpp_new.models.svm import SVMTrainer
from qpp_new.models.NN import NNTrainer
import pickle as pcl



if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run SVM and NN training with specified parameters.")

    # Add arguments
    parser.add_argument('--svm_trainer_path', type=str, required=True,
                        help="Path to the SVM trainer pickle file.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Directory containing experiment data.")
    parser.add_argument('--new_query_folder', type=str, required=True,
                        help="Folder containing new queries for predictions.")
    parser.add_argument('--jarfile', type=str, required=True,
                        help="Path to the JAR file for extracting baseline features.")
    parser.add_argument('--GED_query_file', type=str, required=True,
                        help="File with K representative queries.")
    parser.add_argument('--query_log', type=str, required=True,
                        help="Path to the full query log (all queries across train, val, test).")
    parser.add_argument('--K', type=int, required=True,
                        help="The K value used in the K-medoids algorithm.")
    parser.add_argument('--nn_trainer_path', type=str, required=True,
                        help="Path to the NN trainer pickle file.")

    # Parse the arguments
    args = parser.parse_args()

    # Store args into your specified variables
    svm_trainer_path = args.svm_trainer_path
    data_dir = args.data_dir
    new_query_folder = args.new_query_folder
    new_query_path = os.path.join(new_query_folder, 'queries.tsv')
    jarfile = args.jarfile
    GED_query_file = args.GED_query_file
    query_log = args.query_log
    K = args.K
    nn_trainer_path = args.nn_trainer_path

    # Output the stored variables
    print(f"SVM Trainer Path: {svm_trainer_path}")
    print(f"Data Directory: {data_dir}")
    print(f"New Query Folder: {new_query_folder}")
    print(f"New Query Path: {new_query_path}")
    print(f"JAR File: {jarfile}")
    print(f"GED Query File: {GED_query_file}")
    print(f"Query Log: {query_log}")
    print(f"K Value: {K}")
    print(f"NN Trainer Path: {nn_trainer_path}")

    """"#Required input data
    svm_trainer_path = '/data/DBpedia_3_class_full/svm/svmtrainer.pickle' #SVM object for svm model
    data_dir = '/data/DBpedia_3_class_full' # Directory that include all information about experiments
    new_query_folder = '/data/DBpedia_3_class_full/newQueryTest' # The folder that where the new queries that require predictions are stored. It is assumed that these new queires are stored in a file named 'queries.txt'
    new_query_path = os.path.join(new_query_folder, 'queries.tsv')
    #jarfile = "/PlanRGCN/qpp/jars/sparql-query2vec-0.0.1.jar"
    jarfile = '/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar' # Jar file that contains implementation for extracting baseline features.
    GED_query_file = '/data/DBpedia_3_class_full/baseline/knn25/ged_queries.txt' # File with K represenatitve queries
    query_log = '/data/DBpedia_3_class_full/all.tsv' # path to the full query log (all queries across train, val, and test
    K = 25 # the k used in k mediod algorithm
    nn_trainer_path = '/data/DBpedia_3_class_full/nn/k25/nntrainer.pickle' # NN object for NN model
    """

    # loading query data
    qs_df = pd.read_csv(new_query_path, sep='\t')

    #alg feature generation for new queries
    start = time.time()
    os.system(f"java -jar {jarfile} algebra-features {new_query_path} {os.path.join(new_query_folder, 'alg.tsv')}")
    alg_feat_time = time.time()-start
    with open(os.path.join(new_query_folder, 'alg_inf.txt'),'w') as e_f:
        e_f.write(f"Total duration for extracting AlgebraInference features: {alg_feat_time}")

    #GED feature generation
    ged_queries = []
    with open(GED_query_file, 'r') as g_f:
        for line in g_f.readlines():
            ged_queries.append(line)
    ged_calculator = SPARQLDistCalcular()
    ged_entries = []
    ged_dur_entries = []
    for idx, row in qs_df.iterrows():
        query_text = row['queryString']
        query_id = row['id']
        vec = []
        start = time.time()
        for ged_q in ged_queries:
            vec.append(ged_calculator.distance_ged(query_text, ged_q))
        dur = time.time()-start
        entry = (query_id, vec)
        ged_entries.append(entry)
        ged_dur_entries.append((query_id, dur))
    with open(os.path.join(new_query_folder, 'ged.csv'),'w') as ged_writer:
        ged_writer.write('id,vec\n')
        for ged in ged_entries:
            ged_writer.write(ged[0]+","+str(ged[1])+'\n')
    pd.DataFrame(ged_dur_entries, columns=['id', 'time']).to_csv(os.path.join(new_query_folder, 'ged_dur.csv'), index=False)

    # Extra features for NN
    ## first we create new "full" query log.
    all_df = pd.read_csv(query_log, sep='\t')
    all_df = pd.concat([all_df[['id','queryString', 'mean_latency', 'resultset_0']], qs_df[['id','queryString','mean_latency', 'resultset_0']] ])
    all_df.to_csv(os.path.join(new_query_folder,"all.tsv"), sep='\t', index=False)

    os.system(f"java -jar {jarfile} extra {os.path.join(new_query_folder,'all.tsv')} {os.path.join(new_query_folder,'extra')}")
    start = time.time()
    os.system(
        f"java -jar {jarfile} extra {new_query_path} /tmp/extra")
    extra_nn_feat_time = time.time()-start
    with open(os.path.join(new_query_folder, 'extra_inference.txt'),'w') as e_f:
        e_f.write(f"Total duration for extracting extra features: {extra_nn_feat_time}")

    with open(svm_trainer_path, 'rb') as f:
        svm_trainer = pcl.load(f)


    prepper = SVMDataPrepper(
        train_algebra_path=os.path.join(data_dir, "baseline", "train_alg.tsv"),
        val_algebra_path=os.path.join(data_dir, "baseline", "val_alg.tsv"),
        test_algebra_path=os.path.join(new_query_folder, 'alg.tsv'),
        train_ged_path=os.path.join(data_dir, "baseline", f"knn{K}/train_ged.csv"),
        val_ged_path=os.path.join(data_dir, "baseline", f"knn{K}/val_ged.csv"),
        test_ged_path=os.path.join(new_query_folder, 'ged.csv'),
    )
    train, val, test = prepper.prepare()
    svm_trainer.predict_trained(test=test,output_path=os.path.join(new_query_folder, 'svm_pred.csv'))


    #NN prediction
    prepper = NNDataPrepper(
        train_algebra_path=os.path.join(data_dir, "baseline", "train_alg.tsv"),
        val_algebra_path=os.path.join(data_dir, "baseline", "val_alg.tsv"),
        test_algebra_path=os.path.join(new_query_folder, 'alg.tsv'),
        train_ged_path=os.path.join(data_dir, "baseline", f"knn{K}/train_ged.csv"),
        val_ged_path=os.path.join(data_dir, "baseline", f"knn{K}/val_ged.csv"),
        test_ged_path=os.path.join(new_query_folder, 'ged.csv'),
        filter_join_data_path=os.path.join(new_query_folder, 'extra'),
    )
    train, val, test = prepper.prepare()
    with open(nn_trainer_path, 'rb') as nn_path:
        nn_trainer = pcl.load(nn_path)

        nn_trainer.predict_trained(test, os.path.join(new_query_folder, 'nn_prediction.csv'))

        #### Manual Tests
        """nn_n = pd.read_csv('/data/DBpedia_3_class_full/newQueryTest/nn_prediction.csv')
        nn_o = pd.read_csv('/data/DBpedia_3_class_full/nn/k25/nn_test_pred.csv')
        test_id = 'http://lsq.aksw.org/lsqQuery-dRhGqLYbZTiBglBi0UVHzVcXFZeyQyC27yUV_uGkBKMLIMIT'
        test_id = nn_o['id'].iloc[10]
        nn_o[nn_o['id'] == test_id]['nn_prediction']- nn_n[nn_n['id'] == test_id]['nn_prediction']"""
