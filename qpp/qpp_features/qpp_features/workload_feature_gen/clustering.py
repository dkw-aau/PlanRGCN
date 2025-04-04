# Clustering class inspired by (original implementation of )https://github.com/rhasan/query-performance/blob/cf2230b66c3fd20480d9eb74e1d18da177bbd4c5/clustering/sparql_clustering.py#L196

import pickle

from utils import (
    java_similarity_jar,
    load_data,
    create_sing_dist_file,
    sort_queries_dist_matrx,
    split_dataset,
    recompute_distmatrix,
    write_queries_from_ids,
    predefined_dataset,
    predefined_dataset_w_df,
)
from kmediod import kMediod
import configparser
import os
import numpy as np
import time
import pandas as pd
import time, numpy as np


class QueryClusterer:
    def __init__(
        self,
        configfile,
        training_queries,
        train_ids,
        test_queries=None,
        test_ids=None,
        validation_queries=None,
        val_ids=None,
        distance=None,
    ):
        config = configparser.RawConfigParser()
        config.read(configfile)

        self.train_queries = training_queries
        self.train_ids = train_ids
        self.val_queries = validation_queries
        self.val_ids = val_ids
        self.test_queries = test_queries
        self.test_ids = test_ids

        self.distance = distance
        self.kmediod = kMediod(training_queries, distance)

        # Kmediod related data
        self.X = None

        self.K = config.getint("QueryClustering", "K")
        self.random_shuffel_max_iters = config.getint(
            "QueryClustering", "random_shuffel_max_iters"
        )
        self.kmediods_max_iters = config.getint("QueryClustering", "kmediods_max_iters")
        self.cluster_cach_file = config.get("QueryClustering", "cluster_cach_file")
        self.center_cache_file = config.get("QueryClustering", "center_cache_file")
        self.distance_matrix_file = config.get(
            "QueryClustering", "distance_matrix_file"
        )
        self.vectorization_path = config.get("QueryClustering", "vectorization_path")
        if self.vectorization_path[-1] != "/":
            self.vectorization_path += "/"

        self.center_ids = None
        self.id = None

    def compute_distance_matrix_real_time(self):
        self.distance_matrix = self.kmediod.compute_symmetric_distance(self.X)

        distance_filename = self.distance_cach_filename()
        np.save(self.vectorization_path + distance_filename, self.distance_matrix)

    def compute_distance_matrix_from_cach(self):
        distance_filename = self.distance_cach_filename()
        self.distance_matrix = np.load(
            self.vectorization_path + distance_filename + ".npy"
        )

    def distance_cach_filename(self):
        file_name = "distance_matrix"

        file_name = self.distance_function_name() + "_hungarian"
        file_name = file_name + "_cach"
        return file_name

    def distance_function_name(self):
        if self.distance == None:
            raise Exception("Distance is None, not defined")
        return self.distance.__name__

    def save_clusters(self):
        np.savetxt(self.vectorization_path + self.cluster_cach_file, self.id, fmt="%d")
        np.savetxt(
            self.vectorization_path + self.center_cache_file, self.center_ids, fmt="%d"
        )

    def predict_cluster(self, Xi, center_id_path=None):
        df_name = self.distance_function_name()
        if center_id_path != None:
            self.center_ids = np.load(center_id_path)
        elif self.center_ids == None:
            # self.center_ids = np.load(self.center_cache_file)
            self.center_ids = np.load(
                self.vectorization_path + self.center_cache_file + df_name + ".npy"
            )

        min_dist = np.inf
        min_k = -1

        for k in self.center_ids:
            k_Xi = self.X[k]
            if self.distance == None:
                raise Exception("Distance is None, not defined")
            d = self.distance(Xi, k_Xi)
            if min_dist > d:
                min_dist = d
                min_k = k

        return min_k

    def cluster_queries(self):
        # (min_center_idxs,min_cost) = k_mediods.initial_random_centers_cost_minimization(self.X ,self.K,self.distance_matrix,self.random_shuffel_max_iters,self.kmediods_max_iters)
        # print "min model cost: ", min_cost

        (initial_centers, min_center_idxs) = self.kmediod.initial_random_centers(
            self.X, self.K
        )

        (self.center_ids, self.id) = self.kmediod.k_mediods(
            self.X, min_center_idxs, self.kmediods_max_iters, self.distance_matrix
        )

        ##self.kmediod.print_clusters(self.X, self.id, self.center_ids)

        total_cost = self.kmediod.model_cost(
            self.X, self.id, self.center_ids, self.distance_matrix
        )
        print("model cost: ", total_cost)
        self.total_cost = total_cost

        self.save_clusters()
        return self.center_ids

    def load_distaince_hungarian_matrix(self):
        """
        Must be called after loading training queries
        """
        m = np.size(self.X, 0)
        self.distance_matrix = np.zeros((m, m), dtype=float)
        f = open(self.vectorization_path + self.distance_matrix_file)
        for line in f:
            row = line.split()
            i = int(row[0])
            j = int(row[1])
            d = float(row[2])
            self.distance_matrix[i, j] = d
            self.distance_matrix[j, i] = d

    def load_training_queries(self, limit=None):
        # if limit == None:
        #     limit = int(self.total_query*0.6)

        self.X = np.array(self.train_queries).transpose()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def print_clusters(self):
        self.kmediod.print_clusters(self.X, self.id, self.center_ids)

    def cluster_to_vec(self, query):
        vec = []
        for i in self.center_ids:
            center_query = self.X[i]
            if self.distance_matrix is not None:
                vec.append(self.distance_matrix[i][self.get_query_id(query)])
                # print("Dist matrix: {},  dist func: {}\n".format(self.distance_matrix[i][self.get_query_id( query)],self.distance(query,center_query)))
                # input()
            else:
                if self.distance == None:
                    raise Exception(
                        "Distance is None, not defined. Add dist matric manually"
                    )
                vec.append(self.distance(query, center_query))
        return vec

    def save_feature_representation(
        self,
        file=None,
        train_graph_ptn_path="traindata_cluster.csv",
        val_graph_ptn_path="val_cluster.csv",
        test_graph_ptn_path="test_cluster.csv",
        vec_path=True,
    ):
        if file is None:
            log = False
        else:
            log = True
        if log:
            start = time.time()
            self.save_feat_helper(
                train_graph_ptn_path,
                self.train_ids,
                self.train_queries,
                vec_path=vec_path,
            )
            end = time.time()
            file.write("train_cluster.csv" + " created in " + str(end - start) + "\n")
            print("train_cluster.csv" + " created in " + str(end - start) + "\n")
        else:
            self.save_feat_helper(
                train_graph_ptn_path,
                self.train_ids,
                self.train_queries,
                vec_path=vec_path,
            )
        if log:
            start = time.time()
            self.save_feat_helper(
                val_graph_ptn_path, self.val_ids, self.val_queries, vec_path=vec_path
            )
            end = time.time()
            file.write("val_cluster.csv" + " created in " + str(end - start) + "\n")
            print("val_cluster.csv" + " created in " + str(end - start) + "\n")
        else:
            self.save_feat_helper(
                val_graph_ptn_path, self.val_ids, self.val_queries, vec_path=vec_path
            )
        if log:
            start = time.time()
            self.save_feat_helper(
                test_graph_ptn_path, self.test_ids, self.test_queries, vec_path=vec_path
            )
            end = time.time()
            file.write("test_cluster.csv" + " created in " + str(end - start) + "\n")
            print("test_cluster.csv" + " created in " + str(end - start) + "\n")
        else:
            self.save_feat_helper(
                test_graph_ptn_path, self.test_ids, self.test_queries, vec_path=vec_path
            )

    def save_feat_helper(self, path, ids, queries, vec_path=True):
        if vec_path:
            f = open(self.vectorization_path + path, "w")
        else:
            f = open(path, "w")
        f.write("id,vec\n")
        temp = ""
        for id, query in zip(ids, queries):
            train_vec = self.cluster_to_vec(query)
            t = str(id) + "," + str(train_vec) + "\n"
            temp += t

        f.write(temp)
        f.close()

    # Should probably reorder test and val.
    def get_query_id(self, query):
        for idx, x in enumerate(self.train_queries):
            if query == x:
                return idx

        for idx, x in enumerate(self.val_queries):
            if query == x:
                return len(self.train_queries) + idx
        for idx, x in enumerate(self.test_queries):
            if query == x:
                return len(self.train_queries) + len(self.val_queries) + idx
        return None


def load(path):
    with open(path, "rb") as f:
        clusterer = pickle.load(f)
    return clusterer


def cluster_random_split(log_file, query_log, dist_folder, config_file, clusterer):
    f = open(log_file, "w")
    start = time.time()

    query_ids, queries, execution_times = load_data(query_log)

    idx, dists, exc, id_to_dist_index = create_sing_dist_file(dist_folder)
    queries_filtered = sort_queries_dist_matrx(query_ids, queries, idx)
    # query_ids, queries = query_ids[:2500], queries[:2500]
    load_time = time.time()
    f.write("Time to load data: " + str(load_time - start) + "\n")
    print("Time to load data: " + str(load_time - start) + "\n")

    # Distribution in original work dataset is 60/20/20
    """split = int(len(queries_filtered)*0.8)
    train_queries, train_ids = queries_filtered[:split],idx[:split]
    test_queries, test_ids = queries_filtered[split:],idx[split:]"""
    (
        train_queries,
        train_ids,
        val_queries,
        val_ids,
        test_queries,
        test_ids,
    ) = split_dataset(idx, queries_filtered)
    # new_dists= recompute_distmatrix(train_ids, val_ids, test_ids, id_to_dist_index, dists)

    print("Queries loaded")
    # config_file ='/data/experiment2/config.prop'
    i = QueryClusterer(
        config_file,
        java_similarity_jar,
        train_queries,
        train_ids,
        test_queries,
        test_ids,
        val_queries,
        val_ids,
    )
    clust_time_start = time.time()
    i.load_training_queries()
    i.distance_matrix = dists  # new_dists

    i.cluster_queries()
    f.write("Time to create clusters: " + str(time.time() - clust_time_start) + "\n")
    print("Time to create clusters: " + str(time.time() - clust_time_start) + "\n")
    i.save_feature_representation(file=f)
    f.close()
    # clusterer='k20.pickle'
    i.save(clusterer)

    datapath = os.environ["datapath"]
    exp = os.environ["experiment_name"]
    path = datapath + "/" + exp + "/data"
    write_queries_from_ids(
        i.vectorization_path + "traindata_cluster.csv",
        query_log,
        path + "/training_queries.csv",
    )
    write_queries_from_ids(
        i.vectorization_path + "val_cluster.csv", query_log, path + "/val_queries.csv"
    )
    write_queries_from_ids(
        i.vectorization_path + "test_cluster.csv", query_log, path + "/test_queries.csv"
    )


def cluster_specific_fit(
    log_file, query_log, dist_folder, config_file, clusterer, train_d, val_d, test_d
):
    f = open(log_file, "w")
    start = time.time()

    query_ids, queries, execution_times = load_data(query_log)
    idx, dists, exc, id_to_dist_index = create_sing_dist_file(dist_folder)

    queries_filtered = sort_queries_dist_matrx(query_ids, queries, idx)
    # query_ids, queries = query_ids[:2500], queries[:2500]
    load_time = time.time()
    f.write("Time to load data: " + str(load_time - start) + "\n")
    print("Time to load data: " + str(load_time - start) + "\n")

    # Distribution in original work dataset is 60/20/20
    """split = int(len(queries_filtered)*0.8)
    train_queries, train_ids = queries_filtered[:split],idx[:split]
    test_queries, test_ids = queries_filtered[split:],idx[split:]"""
    (
        train_queries,
        train_ids,
        val_queries,
        val_ids,
        test_queries,
        test_ids,
    ) = predefined_dataset(idx, queries_filtered, train_d, val_d, test_d)
    print("Queries split into train/val/test sets with:\n")
    print(
        "\t{} training queries\n\t{} validation queries\n\t{} test queries".format(
            len(train_queries), len(val_queries), len(test_queries)
        )
    )
    new_dists = recompute_distmatrix(
        train_ids, val_ids, test_ids, id_to_dist_index, dists
    )
    print("Distance matrix recomputed")

    i = QueryClusterer(
        config_file,
        java_similarity_jar,
        train_queries,
        train_ids,
        test_queries,
        test_ids,
        val_queries,
        val_ids,
    )
    clust_time_start = time.time()
    i.load_training_queries()
    i.distance_matrix = new_dists

    i.cluster_queries()
    f.write("Time to create clusters: " + str(time.time() - clust_time_start) + "\n")
    print("Time to create clusters: " + str(time.time() - clust_time_start) + "\n")
    i.save_feature_representation(file=f)
    f.flush()
    f.close()
    # clusterer='k20.pickle'
    i.save(clusterer)

    datapath = os.environ["datapath"]
    exp = os.environ["experiment_name"]
    path = datapath + "/" + exp + "/data"
    write_queries_from_ids(
        i.vectorization_path + "traindata_cluster.csv",
        query_log,
        path + "/training_queries.csv",
    )
    write_queries_from_ids(
        i.vectorization_path + "val_cluster.csv", query_log, path + "/val_queries.csv"
    )
    write_queries_from_ids(
        i.vectorization_path + "test_cluster.csv", query_log, path + "/test_queries.csv"
    )


def queries_with_no_dist(df: pd.DataFrame, id_to_dist_index):
    ids_to_remove = []
    for i in df["queryID"]:
        try:
            id_to_dist_index[i]
        except KeyError:
            ids_to_remove.append(i)
    df = df[~df["queryID"].isin(ids_to_remove)]
    return df


# fixed version of recomputation of distances
def recompute_lsq_dists_old(dir_path, train_ids, val_ids, test_ids):
    start = time.time()
    files = sorted(os.listdir(dir_path))
    all_ids = train_ids
    all_ids.extend(val_ids)
    all_ids.extend(test_ids)
    all_ids = set(all_ids)
    print(f"{time.time()-start}: Ids in train val test: {len(all_ids)}")
    dists = {}
    # missing_pairs = []
    lsq_ids_dict = {}
    lsq_ids_order = []
    for f_count, f in enumerate(files):
        with open(f"{dir_path}/{f}", "r") as fh:
            for l_count, line in enumerate(fh.readlines()):
                lsq_ids_dict[f_count + l_count] = line.split(",")[0]
                lsq_ids_order.append(line.split(",")[0])
    # temp code
    with open("lsq_order.pickle", "wb") as f:
        pickle.dump(lsq_ids_order, f)

    print(f"{time.time()-start}: {len(files)} enumerated!")
    for num_f, f in enumerate(files):
        print(f"{time.time()-start}: Beginning file no: {num_f+1}")
        with open(f"{dir_path}/{f}", "r") as fh:
            for line in fh.readlines():
                splits = line.split(",")
                current_id = splits[0]
                if not current_id in all_ids:
                    continue
                ds = splits[2:]
                # final_idx = None
                for idx, d in enumerate(ds):
                    if not lsq_ids_order[idx] in all_ids:
                        continue
                    dists[current_id, lsq_ids_order[idx]] = float(d)
                    # dists[lsq_ids_order[idx],current_id] = float(d)
                    # final_idx = idx
                # if final_idx < len(lsq_ids_order):
                #    for i in range(len(lsq_ids_order)-final_idx, len(lsq_ids_order)):
                #        missing_pairs.append((current_id,lsq_ids_order[i]))
    del all_ids
    new_order = train_ids
    new_order.extend(val_ids)
    new_order.extend(test_ids)
    queries_w_no_dists = set()
    t = []
    for x in new_order:
        if x in lsq_ids_order:
            t.append(x)
        else:
            queries_w_no_dists.add(x)
    new_order = t

    dist_matrix = []
    for ids in new_order:
        temp = []
        for ids2 in new_order:
            d = None
            try:
                d = dists[ids, ids2]
            except KeyError:
                d = dists[ids2, ids]
            temp.append(d)
        dist_matrix.append(temp)
    return np.array(dist_matrix), list(queries_w_no_dists)


# fixed version of recomputation of distances, efficient version
def recompute_lsq_dists(dir_path, train_ids, val_ids, test_ids):
    start = time.time()
    files = sorted(os.listdir(dir_path))
    files = sorted([x for x in files if x.endswith(".csv")])
    # temp code
    # with open('lsq_order.pickle', 'rb') as f:
    #    lsq_ids_order = pickle.load(f)
    # print(f'{time.time()-start}: lsq order loaded: {len(lsq_ids_order)}')
    missing_pairs = []
    lines = []
    lsq_ids_order = []
    for f_count, f in enumerate(files):
        print(f"{time.time()-start}: Beginning file no: {f_count+1}")
        with open(f"{dir_path}/{f}", "r") as fh:
            for l_count, line in enumerate(fh.readlines()):
                lsq_ids_order.append(line.split(",")[0])
                lines.append(
                    np.array(line.split(",")[2:]).astype(np.float32)
                )  # 1 in ged dist file is the runtime.

    with open("lines.pickle", "wb") as f:
        pickle.dump(lines, f)
    with open("lines.pickle", "rb") as f:
        lines = pickle.load(f)
    print("lines done")
    dists = np.stack(lines, axis=0)
    # with open('dists.pickle','rb') as f:
    #    dists = pickle.load(f)
    print("dists done")
    del lines
    new_order = train_ids
    new_order.extend(val_ids)
    new_order.extend(test_ids)
    queries_w_no_dists = set()
    dist_matrix = []
    lsq_mapping = {}
    for i in range(len(lsq_ids_order)):
        lsq_mapping[lsq_ids_order[i]] = i
    with open("lsqmapper.pickle", "wb") as f:
        pickle.dump(lsq_mapping, f)
    print("lsq mapper done")
    for no_ids, ids in enumerate(new_order):
        print(f"{time.time()-start}: Beginning at no: {no_ids+1}")
        temp = []
        try:
            ids_index = lsq_mapping[ids]
        except KeyError:
            queries_w_no_dists.add(ids)
            continue
        for ids2 in new_order:
            try:
                ids2_index = lsq_mapping[ids2]
                # ids2_index = lsq_ids_order.index(ids2)
            except KeyError:
                queries_w_no_dists.add(ids2)
                continue
            d = None
            try:
                d = dists[ids_index, ids2_index]
            except KeyError:
                d = dists[ids2_index, ids_index]
            temp.append(d)
        dist_matrix.append(temp)
    # temp code
    # for f_count,f in enumerate(files):
    #    with open(f"{dir_path}/{f}",'r') as fh:
    #        for l_count,line in enumerate(fh.readlines()):
    #            lsq_ids_dict[f_count+l_count] = line.split(',')[0]
    #            lsq_ids_order.append(line.split(',')[0])
    # with open('lsq_order.pickle','wb') as f:
    #    pickle.dump(lsq_ids_order,f)
    return np.array(dist_matrix), list(queries_w_no_dists)


def cluster_lsq_datasets(
    train_df,
    val_df,
    test_df,
    train_graph_path,
    val_graph_path,
    test_graph_path,
    dist_dir_path,
    config_file,
    save_clust_obj=None,
):
    # _, dists, _, id_to_dist_index = create_sing_dist_file(dist_dir_path)

    # new_dists, lsq_ids_dict = lsq_compute_dists(dist_dir_path)#recompute_distmatrix(train_ids, val_ids, test_ids, id_to_dist_index, dists)
    # print(new_dists[1,2912])
    # train_df = queries_with_no_dist(train_df, id_to_dist_index)
    train_ids = list(train_df["queryID"])
    # val_df = queries_with_no_dist(val_df, id_to_dist_index)
    val_ids = list(val_df["queryID"])
    # test_df = queries_with_no_dist(test_df, id_to_dist_index)
    test_ids = list(test_df["queryID"])
    # new_dists= recompute_distmatrix(train_ids, val_ids, test_ids, id_to_dist_index, dists)
    print("Begin dist matrix computation")
    new_dists, queries_to_remove = recompute_lsq_dists(
        dist_dir_path, train_ids, val_ids, test_ids
    )
    print("Finsihed dist matrix computation")

    train_len, val_len, test_len = len(train_df), len(val_df), len(test_df)
    train_df = train_df.set_index("queryID")
    val_df = val_df.set_index("queryID")
    test_df = test_df.set_index("queryID")
    train_df = train_df.drop(queries_to_remove, axis=0, errors="ignore")
    test_df = test_df.drop(queries_to_remove, axis=0, errors="ignore")
    val_df = val_df.drop(queries_to_remove, axis=0, errors="ignore")

    # to check for no filtering
    assert train_len == len(train_df)
    assert val_len == len(val_df)
    assert test_len == len(test_df)

    train_queries, val_queries, test_queries = (
        list(train_df["queryString"]),
        list(val_df["queryString"]),
        list(test_df["queryString"]),
    )

    i = QueryClusterer(
        config_file,
        train_queries,
        train_ids,
        test_queries=test_queries,
        test_ids=test_ids,
        validation_queries=val_queries,
        val_ids=val_ids,
    )
    clust_time_start = time.time()
    i.load_training_queries()
    i.distance_matrix = new_dists

    print("Begin Clustering")
    i.cluster_queries()
    # f.write("Time to create clusters: "+str(time.time()-clust_time_start)+"\n")
    print("Time to create clusters: " + str(time.time() - clust_time_start) + "\n")
    i.save_feature_representation(
        train_graph_ptn_path=train_graph_path,
        val_graph_ptn_path=val_graph_path,
        test_graph_ptn_path=test_graph_path,
        vec_path=False,
    )

    # f.flush()
    # f.close()
    # clusterer='k20.pickle'
    if not save_clust_obj == None:
        i.save(save_clust_obj)


def recompute_lsq_dists_train(dir_path, train_ids):
    start = time.time()
    files = sorted(os.listdir(dir_path))

    # temp code
    # with open('lsq_order.pickle', 'rb') as f:
    #    lsq_ids_order = pickle.load(f)
    lsq_ids_order = []
    for f_count, f in enumerate(files):
        with open(f"{dir_path}/{f}", "r") as fh:
            for line in fh.readlines():
                lsq_ids_order.append(line.split(",")[0])
    print(f"{time.time()-start}: lsq order loaded: {len(lsq_ids_order)}")
    # missing_pairs = []
    lines = []
    for f_count, f in enumerate(files):
        print(f"{time.time()-start}: Beginning file no: {f_count+1}")
        with open(f"{dir_path}/{f}", "r") as fh:
            for l_count, line in enumerate(fh.readlines()):
                lines.append(np.array(line.split(",")[1:]).astype(np.float32))

    # with open('lines.pickle','wb') as f:
    #    pickle.dump(lines,f)

    # with open('lines.pickle','rb') as f:
    #    lines = pickle.load(f)
    print("lines done")

    dists = np.stack(lines, axis=0)
    # with open('dists.pickle','rb') as f:
    #    dists = pickle.load(f)
    print("dists done")
    del lines
    new_order = train_ids
    queries_w_no_dists = set()
    dist_matrix = []
    lsq_mapping = {}
    for i in range(len(lsq_ids_order)):
        lsq_mapping[lsq_ids_order[i]] = i
    with open("lsqmapper.pickle", "wb") as f:
        pickle.dump(lsq_mapping, f)
    print("lsq mapper done")
    for no_ids, ids in enumerate(new_order):
        print(f"{time.time()-start}: Beginning at no: {no_ids+1}")
        temp = []
        try:
            ids_index = lsq_mapping[ids]
        except KeyError:
            queries_w_no_dists.add(ids)
            continue
        for ids2 in new_order:
            try:
                ids2_index = lsq_mapping[ids2]
                # ids2_index = lsq_ids_order.index(ids2)
            except KeyError:
                queries_w_no_dists.add(ids2)
                continue
            d = None
            try:
                d = dists[ids_index, ids2_index]
            except KeyError:
                d = dists[ids2_index, ids_index]
            temp.append(d)
        dist_matrix.append(temp)
    # temp code
    # for f_count,f in enumerate(files):
    #    with open(f"{dir_path}/{f}",'r') as fh:
    #        for l_count,line in enumerate(fh.readlines()):
    #            lsq_ids_dict[f_count+l_count] = line.split(',')[0]
    #            lsq_ids_order.append(line.split(',')[0])
    # with open('lsq_order.pickle','wb') as f:
    #    pickle.dump(lsq_ids_order,f)
    return np.array(dist_matrix), list(queries_w_no_dists)


def find_k_queries_lsq_datasets(
    train_df, dist_dir_path, config_file, save_clust_obj=None, save_train_k_queries=None
):
    # _, dists, _, id_to_dist_index = create_sing_dist_file(dist_dir_path)

    # new_dists, lsq_ids_dict = lsq_compute_dists(dist_dir_path)#recompute_distmatrix(train_ids, val_ids, test_ids, id_to_dist_index, dists)
    # print(new_dists[1,2912])
    # train_df = queries_with_no_dist(train_df, id_to_dist_index)
    train_ids = list(train_df["queryID"])

    # new_dists= recompute_distmatrix(train_ids, val_ids, test_ids, id_to_dist_index, dists)
    print("Begin dist matrix computation")
    new_dists, queries_to_remove = recompute_lsq_dists_train(dist_dir_path, train_ids)
    print("Finsihed dist matrix computation")

    train_df = train_df.set_index("queryID")
    train_df = train_df.drop(queries_to_remove, axis=0, errors="ignore")
    train_queries = list(train_df["queryString"])

    i = QueryClusterer(config_file, train_queries, train_ids)
    clust_time_start = time.time()
    i.load_training_queries()
    i.distance_matrix = new_dists

    print("Begin Clustering")
    center_idxs = i.cluster_queries()
    if save_train_k_queries != None:
        with open(save_train_k_queries, "a") as f:
            for center_idx in center_idxs:
                f.write(train_queries[center_idx] + "\n")
    # f.write("Time to create clusters: "+str(time.time()-clust_time_start)+"\n")
    print(
        "Time to select cluster queries: " + str(time.time() - clust_time_start) + "\n"
    )

    # f.flush()
    # f.close()
    # clusterer='k20.pickle'
    if not save_clust_obj == None:
        i.save(save_clust_obj)


import sys

if __name__ == "__main__":
    train_df = pd.read_csv(
        "/Users/abirammohanaraj/Documents/GitHub/dkw/qpp/dataset/knn/train_0.tsv",
        sep="\t",
    )
    dist_dir_path = (
        "/Users/abirammohanaraj/Documents/GitHub/dkw/qpp/dataset/knn/train_0"
    )
    config_file = "/Users/abirammohanaraj/Documents/GitHub/dkw/qpp/dataset/config.prop"
    save_train_k_queries = (
        "/Users/abirammohanaraj/Documents/GitHub/dkw/qpp/dataset/train_centers_0.txt"
    )
    find_k_queries_lsq_datasets(
        train_df, dist_dir_path, config_file, save_train_k_queries=save_train_k_queries
    )
    exit()
    train_df = pd.read_csv("/code/data/lsqrun/better_dist/train.tsv", sep="\t")
    val_df = pd.read_csv("/code/data/lsqrun/better_dist/test.tsv", sep="\t")
    test_df = pd.read_csv("/code/data/lsqrun/better_dist/val.tsv", sep="\t")
    dist_matrix, queries_to_remove = recompute_lsq_dists(
        sys.argv[1],
        list(train_df["queryID"]),
        list(val_df["queryID"]),
        list(test_df["queryID"]),
    )
    print(f"Amount of queries which should be dropped: {len(queries_to_remove)}")
    print(
        f"Before drop Train: {len(train_df)}, Val: {len(val_df)}, test: {len(test_df)}"
    )
    train_df = train_df.set_index("queryID")
    val_df = val_df.set_index("queryID")
    test_df = test_df.set_index("queryID")
    train_df = train_df.drop(queries_to_remove, axis=0, errors="ignore")
    test_df = test_df.drop(queries_to_remove, axis=0, errors="ignore")
    val_df = val_df.drop(queries_to_remove, axis=0, errors="ignore")
    print(
        f"After drop Train: {len(train_df)}, Val: {len(val_df)}, test: {len(test_df)}"
    )
    # lsq_compute_dists(sys.argv[1])
