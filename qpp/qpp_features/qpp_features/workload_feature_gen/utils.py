import os
from subprocess import PIPE, Popen, STDOUT, check_output
import pathlib

import numpy as np
import pandas as pd
from threading import Timer
import pandas as pd
from sklearn.model_selection import train_test_split


def java_similarity(path_to_jar, query1, query2):
    p = Popen(
        ["java", "-cp", path_to_jar, "Main", "ged", query1, query2],
        stdout=PIPE,
        stderr=STDOUT,
    )
    return float([x for x in p.stdout][4][:-1])


def java_similarity_jar(query1, query2):
    path = pathlib.Path(__file__).parent.resolve()
    path = path + "/distanceJars"
    # path = "/Users/abirammohanaraj/Documents/GitHub/qpp/SPARQLAlgebra/target/"
    jarstring = "FeatureExtraction-1.0-jar-with-dependencies.jar"
    dependency = "GMT.jar"
    path_to_jar = path + jarstring + ":" + path + dependency
    kill = lambda process: process.kill()
    p = Popen(
        ["java", "-cp", path_to_jar, "Main", "ged", query1, query2],
        stdout=PIPE,
        stderr=STDOUT,
    )
    my_timer = Timer(10, kill, [p])
    # cmd ='java -cp '+ path_to_jar+ " Main "+ "ged " +"\""+ query1 +"\""+ "\""+query2+"\""
    # output = check_output(cmd,stderr=STDOUT,timeout=10)
    # return output
    try:
        my_timer.start()
        stdout, stderr = p.communicate()
        return stdout.decode("utf-8").split("\n")[-1]
    finally:
        my_timer.cancel()
    return np.inf
    # return np.inf
    # return float([x for x in p.stdout][4][:-1])


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df["s"], df["query"], df["runTimeMs"]


def create_sing_dist_file(path):
    files = os.listdir(path)
    # files = [x for x in files if x.endswith('.csv') and x.startswith('hungarian_distance')]
    files = [x for x in files if x.endswith("csv")]
    files = sorted(files)
    idx, distances, execution_times = [], [], []
    id_to_dist_mapper, index = {}, 0
    for i in range(len(files)):
        with open(path + "/" + files[i], "r") as f:
            print("File {} of {}".format(i, len(files)))
            for line in f.readlines():
                n_l = line[:-1]
                spl = n_l.split(",")
                if len(spl) < 1:
                    continue
                idx.append(spl[0])
                id_to_dist_mapper[spl[0]] = index
                index += 1
                execution_times.append(float(spl[1]))
                distances.append(spl[2:])

    return (
        idx,
        np.array(distances, dtype=np.double),
        np.array(execution_times, dtype=np.double),
        id_to_dist_mapper,
    )


def sort_queries_dist_matrx(query_ids, queries, actual_ids):
    actual_queries = {}
    for idx, query in zip(query_ids, queries):
        for i in actual_ids:
            if i == idx:
                actual_queries[i] = query
    return [actual_queries[x] for x in actual_ids]


def recompute_distmatrix(train_ids, val_ids, test_ids, id_to_dist, old_dist_matrix):
    dist = []
    for i in train_ids:
        temp = []
        for x in train_ids:
            temp.append(old_dist_matrix[id_to_dist[i]][id_to_dist[x]])
        for x in val_ids:
            temp.append(old_dist_matrix[id_to_dist[i]][id_to_dist[x]])
        for x in test_ids:
            temp.append(old_dist_matrix[id_to_dist[i]][id_to_dist[x]])
        dist.append(temp)
    for i in val_ids:
        temp = []
        for x in train_ids:
            temp.append(old_dist_matrix[id_to_dist[i]][id_to_dist[x]])
        for x in val_ids:
            temp.append(old_dist_matrix[id_to_dist[i]][id_to_dist[x]])
        for x in test_ids:
            temp.append(old_dist_matrix[id_to_dist[i]][id_to_dist[x]])
        dist.append(temp)
    for i in test_ids:
        temp = []
        for x in train_ids:
            temp.append(old_dist_matrix[id_to_dist[i]][id_to_dist[x]])
        for x in val_ids:
            temp.append(old_dist_matrix[id_to_dist[i]][id_to_dist[x]])
        for x in test_ids:
            temp.append(old_dist_matrix[id_to_dist[i]][id_to_dist[x]])
        dist.append(temp)
    return np.array(dist)


def split_dataset(ids, queries, rand_state=42):
    spl_60 = int(len(queries) * 0.6)
    spl_20 = int((len(queries) - spl_60) / 2)
    remain_20 = len(queries) - spl_60 - spl_20
    assert len(ids) == len(queries)
    train, testval, train_ids, testval_ids = train_test_split(
        queries, ids, train_size=0.6, random_state=rand_state
    )
    val, test, val_ids, test_ids = train_test_split(
        testval, testval_ids, train_size=0.5, random_state=rand_state
    )
    assert len(train) + len(val) + len(test) == len(queries)
    assert len(train_ids) + len(test_ids) + len(val_ids) == len(ids)
    return train, train_ids, val, val_ids, test, test_ids


def predefined_dataset(ids, queries, train, val, test, id_column="id"):
    train_ids = list(pd.read_csv(train)[id_column])
    test_ids = list(pd.read_csv(test)[id_column])
    val_ids = list(pd.read_csv(val)[id_column])
    train_q, val_q, test_q = [], [], []
    for i, q in zip(ids, queries):
        if i in train_ids:
            train_q.append(q)
        if i in test_ids:
            test_q.append(q)
        if i in val_ids:
            val_q.append(q)
    assert len(train_q) == len(train_ids)
    assert len(test_q) == len(test_ids)
    assert len(val_q) == len(val_ids)
    return train_q, train_ids, val_q, val_ids, test_q, test_ids


def predefined_dataset_w_df(ids, queries, train, val, test, id_column="id"):
    train_ids = list(train[id_column])
    test_ids = list(test[id_column])
    val_ids = list(val[id_column])
    train_q, val_q, test_q = [], [], []
    for i, q in zip(ids, queries):
        if i in train_ids:
            train_q.append(q)
        if i in test_ids:
            test_q.append(q)
        if i in val_ids:
            val_q.append(q)
    assert len(train_q) == len(train_ids)
    assert len(test_q) == len(test_ids)
    assert len(val_q) == len(val_ids)
    return train_q, train_ids, val_q, val_ids, test_q, test_ids


def get_indexed_data(data, index):
    new = []
    for i in index:
        new.append(data[i])
    return new


def get_id(query, queries, ids):
    for q, i in zip(queries, ids):
        if q == query:
            return i
    return None


def write_queries_from_ids(id_file, query_logs, output_path):
    ids = []
    with open(id_file, "r") as f:
        for t, line in enumerate(f.readlines()):
            if t > 0:
                ids.append(line.split(",")[0])
    df = pd.read_csv(query_logs)
    f = open(output_path, "w")
    f.write("id,query\n")
    for i in ids:
        row = df.loc[df["s"] == i]
        f.write("{},{}\n".format(row.iloc[0]["s"], row.iloc[0]["query"]))
    f.close()


def create_queries_only_file(fp, ofp):
    queries = []
    with open(fp, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.replace("\n", "")
            if i > 0:
                spl = line.split(",", 1)
                queries.append(spl[1])
    with open(ofp, "w") as f:
        for q in queries:
            f.write(q + "\n")


def create_algebra_only_file(fp, ofp):
    queries = []
    with open(fp, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            spl = line.split(",", 1)
            queries.append(spl[1])
    with open(ofp, "w") as f:
        for q in queries:
            f.write(q + "\n")


def create_execution_time_file(ifp, ofp, query_log):
    df = pd.read_csv(query_log)
    ids = []
    with open(ifp, "r") as f:
        for t, line in enumerate(f.readlines()):
            if t > 0:
                ids.append(line.split(",")[0])
    f = open(ofp, "w")
    f.write("ex_time\n")
    for i in ids:
        row = df.loc[df["s"] == i]
        f.write("{}\n".format(row.iloc[0]["runTimeMs"]))
    f.close()


def cluster_to_final_file(ifp, ofp):
    w = open(ofp, "w")
    with open(ifp, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i > 0:
                vec = line.replace("\n", "").split(",", 1)[1].strip("][").split(",")
                vec = [(1 / (1 + float(x))) for x in vec]
                if i == 1:
                    cluster_prefix = "pcs"
                    header = [cluster_prefix + str(i) for i, _ in enumerate(vec)]
                    temp = ""
                    for h in header:
                        temp += h + ","
                    temp = temp[:-1]
                    temp += "\n"
                    w.write(temp)
                temp = ""
                for v in vec:
                    temp += str(v) + ","
                temp = temp[:-1]
                temp += "\n"
                w.write(temp)
    w.close()


def separate_alg_splits(alg_feat, query_file, output_path):
    filter_ids = []
    with open(query_file, "r") as f:
        f.readline()
        for line in f.readlines():
            spl = line.split(",", 1)
            filter_ids.append(spl[0])

    df = pd.read_csv(alg_feat)
    df = df.loc[df["query_id"].isin(filter_ids)]
    df = df[
        [
            "query_id",
            "triple",
            "bgp",
            "join",
            "leftjoin",
            "union",
            "filter",
            "graph",
            "extend",
            "minus",
            "path*",
            "pathN*",
            "path+",
            "pathN+",
            "path?",
            "notoneof",
            "tolist",
            "order",
            "project",
            "distinct",
            "reduced",
            "multi",
            "top",
            "group",
            "assign",
            "sequence",
            "slice",
            "treesize",
            "execTime",
        ]
    ]
    assert len(df) == len(filter_ids)
    df.to_csv(output_path, index=False)


def prepare_svm_dat(path, querylog):
    create_queries_only_file(
        path + "/val_queries.csv", path + "/svm/" + "/val_queries.csv"
    )
    create_queries_only_file(
        path + "/test_queries.csv", path + "/svm/" + "/test_queries.csv"
    )
    create_queries_only_file(
        path + "/training_queries.csv", path + "/svm/" + "/train_queries.csv"
    )

    create_algebra_only_file(
        path + "/train_algebra.csv", path + "/svm/" + "/train_queries_algebra.csv"
    )
    create_algebra_only_file(
        path + "/val_algebra.csv", path + "/svm/" + "/val_queries_algebra.csv"
    )
    create_algebra_only_file(
        path + "/test_algebra.csv", path + "/svm/" + "/test_queries_algebra.csv"
    )

    create_execution_time_file(
        path + "/training_queries.csv", path + "/svm/" + "/train_exctimes.csv", querylog
    )
    create_execution_time_file(
        path + "/test_queries.csv", path + "/svm/" + "/test_exctimes.csv", querylog
    )
    create_execution_time_file(
        path + "/val_queries.csv", path + "/svm/" + "/val_exctimes.csv", querylog
    )

    cluster_to_final_file(
        path + "/val_cluster.csv", path + "/svm/" + "/val_cluster.csv"
    )
    cluster_to_final_file(
        path + "/traindata_cluster.csv", path + "/svm/" + "/train_cluster.csv"
    )
    cluster_to_final_file(
        path + "/test_cluster.csv", path + "/svm/" + "/test_cluster.csv"
    )


if __name__ == "__main__":
    # Test
    query = (
        "SELECT distinct ?value WHERE { <http://dbpedia.org/resource/Italy> "
        + "<http://dbpedia.org/property/pushpinMap> ?value . <http://dbpedia.org/resource/Ital> "
        + "?test ?value . OPTIONAL{ <http://dbpedia.org/resource/Ita> "
        + "<http://dbpedia.org/property/pushpinMap> ?value .} }"
    )

    query2 = (
        "SELECT distinct ?value WHERE { <http://dbpedia.org/resource/Italy> "
        + "<http://dbpedia.org/property/pushpinMap> ?value . <http://dbpedia.org/resource/Ital> "
        + "<http://dbpedia.org/property/pushpinMap> ?value . OPTIONAL{ <http://dbpedia.org/resource/Ita> "
        + "<http://dbpedia.org/property/pushpinMap> ?value .} }"
    )
    """dist_folder = '/Users/abirammohanaraj/Documents/GitHub/sparql-query2vec/data'
    query_log="/Users/abirammohanaraj/Documents/GitHub/qpp/data/datasetlsq_30000.csv"
    test_queries="/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/test_cluster.csv"
    train_queries="/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/traindata_cluster.csv"
    val_queries ="/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/val_cluster.csv""" ""
    query_log = "/Users/abirammohanaraj/Documents/GitHub/qpp/data/datasetlsq_30000.csv"
    # write_queries_from_ids(train_queries,query_log,"data/experiment2/train_queries.csv")
    # write_queries_from_ids(test_queries,query_log,"data/experiment2/test_queries.csv")
    # write_queries_from_ids(val_queries,query_log,"data/experiment2/val_queries.csv")
    # idx, dists, exc, id_mapper = create_sing_dist_file(dist_folder)
    # d = java_similarity_jar(query, query2)
    # print(d)

    # svm data creation
    create_queries_only_file(
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/val_queries.csv",
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/svm_data/val_queries.csv",
    )
    create_queries_only_file(
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/test_queries.csv",
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/svm_data/test_queries.csv",
    )
    create_queries_only_file(
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/train_queries.csv",
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/svm_data/train_queries.csv",
    )
    create_algebra_only_file(
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/train_queries_algebra.csv",
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/svm_data/train_queries_algebra.csv",
    )
    create_algebra_only_file(
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/val_queries_algebra.csv",
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/svm_data/val_queries_algebra.csv",
    )
    create_algebra_only_file(
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/test_queries_algebra.csv",
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/svm_data/test_queries_algebra.csv",
    )

    create_execution_time_file(
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/train_queries.csv",
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/svm_data/train_exctimes.csv",
        query_log,
    )
    create_execution_time_file(
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/test_queries.csv",
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/svm_data/test_exctimes.csv",
        query_log,
    )
    create_execution_time_file(
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/val_queries.csv",
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/svm_data/val_exctimes.csv",
        query_log,
    )
    cluster_to_final_file(
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/val_cluster.csv",
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/svm_data/val_cluster.csv",
    )
    cluster_to_final_file(
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/traindata_cluster.csv",
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/svm_data/train_cluster.csv",
    )
    cluster_to_final_file(
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/test_cluster.csv",
        "/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/svm_data/test_cluster.csv",
    )
