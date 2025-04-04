import os
from clustering import (
    cluster_random_split,
    cluster_specific_fit,
    cluster_lsq_datasets,
    find_k_queries_lsq_datasets,
)
from utils import separate_alg_splits, prepare_svm_dat
import sys
import pandas as pd


# Name of expriment
def generate_data(experiment):
    datapath = os.environ["datapath"]
    path = datapath + "/" + experiment
    generate_data_path = datapath + "/" + experiment
    dists = generate_data_path + "/dists/"
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(generate_data_path):
        os.makedirs(generate_data_path)
    if not os.path.exists(generate_data_path + "/data"):
        os.makedirs(generate_data_path + "/data")
    if not os.path.exists(dists):
        os.makedirs(dists)
    if not os.path.exists("/test"):
        os.makedirs("/test")

    os.system(
        "cd sparql-query2vec/target && java -jar sparql-query2vec-0.0.1.jar edit-distance "
        + datapath
        + "/datasetlsq_30000.csv "
        + dists
    )
    os.system(
        "cd sparql-query2vec/target && java -jar sparql-query2vec-0.0.1.jar algebra-features "
        + datapath
        + "/datasetlsq_30000.csv "
        + generate_data_path
        + "/data/alg_feats.csv"
    )
    os.system(
        "cd sparql-query2vec/target && java -jar sparql-query2vec-0.0.1.jar rlearning  "
        + datapath
        + "/datasetlsq_30000.csv "
        + generate_data_path
        + "/data/extra.csv"
    )

    if os.path.exists(path + "/config.prop"):
        config_file = path + "/config.prop"
    else:
        config_file = os.environ["config"]
    cluster_specific_fit(
        datapath + "/" + experiment + "/clust_log.txt",
        datapath + "/datasetlsq_30000.csv",
        datapath + "/" + experiment + "/data/dists",
        config_file,
        datapath + "/" + experiment + "/clust.pickle",
        datapath + "/dataclei/x_train_data.csv",
        datapath + "/dataclei/x_val_data.csv",
        datapath + "/dataclei/x_test_data.csv",
    )
    if not os.path.exists(generate_data_path + "/data/svm"):
        os.makedirs(generate_data_path + "/data/svm")
    separate_alg_splits(
        generate_data_path + "/data/alg_feats.csv",
        generate_data_path + "/data/training_queries.csv",
        generate_data_path + "/data/train_algebra.csv",
    )
    separate_alg_splits(
        generate_data_path + "/data/alg_feats.csv",
        generate_data_path + "/data/val_queries.csv",
        generate_data_path + "/data/val_algebra.csv",
    )
    separate_alg_splits(
        generate_data_path + "/data/alg_feats.csv",
        generate_data_path + "/data/test_queries.csv",
        generate_data_path + "/data/test_algebra.csv",
    )
    prepare_svm_dat(generate_data_path + "/data", datapath + "/datasetlsq_30000.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "old":
        path = os.environ["experiment_name"]
        generate_data(path)
    elif sys.argv[1] == "center-queries":
        # This is the new functionality that computes the center queries for k clustered files
        if len(sys.argv) != 5:
            print(
                "Incorrect number of arguments. \n\tpath to folder with train splits\n\tconfig file\n\tpath to save center queries"
            )
            exit()
        path = sys.argv[2]
        config_file = sys.argv[3]
        path_to_save = sys.argv[4]
        # config_file='/Users/abirammohanaraj/Documents/GitHub/dkw/qpp/dataset/config.prop'
        files = os.listdir(path)
        files = sorted(files)
        dirs = [x for x in files if os.path.isdir(f"{path}/{x}")]
        data_files = [f"{x}.tsv" for x in dirs]
        for train_data_file, dist_dir in zip(data_files, dirs):
            train_df = pd.read_csv(f"{path}/{train_data_file}", sep="\t")
            find_k_queries_lsq_datasets(
                train_df,
                f"{path}/{dist_dir}",
                config_file,
                save_train_k_queries=path_to_save,
            )
    elif len(sys.argv) > 2 and sys.argv[1] == "lsq-data":
        # TODO need to change with new format.
        path = sys.argv[2]
        train_df = pd.read_csv(path + "train.tsv", sep="\t")
        val_df = pd.read_csv(path + "val.tsv", sep="\t")
        test_df = pd.read_csv(path + "test.tsv", sep="\t")
        print(f"Train, val, test loaded")
        cluster_lsq_datasets(
            train_df,
            val_df,
            test_df,
            path + "train_graph.txt",
            path + "val_graph.txt",
            path + "test_graph.txt",
            path + "dists",
            path + "config.prop",
            save_clust_obj=path + "cluster.pickle",
        )
    elif len(sys.argv) > 2 and sys.argv[1] == "DBpediaSample":
        # TODO need to change with new format.
        path = sys.argv[2]
        train_df = pd.read_csv(path + "train_sampled.tsv", sep="\t")
        val_df = pd.read_csv(path + "val_sampled.tsv", sep="\t")
        test_df = pd.read_csv(path + "test_sampled.tsv", sep="\t")
        print(f"Train, val, test loaded")
        cluster_lsq_datasets(
            train_df,
            val_df,
            test_df,
            path + "train_graph.txt",
            path + "val_graph.txt",
            path + "test_graph.txt",
            path + "dists",
            path + "config.prop",
            save_clust_obj=path + "cluster.pickle",
        )
