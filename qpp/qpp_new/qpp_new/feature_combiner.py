import os
from pathlib import Path
import numpy as np
import time
import pandas as pd
import time, numpy as np
import pickle
import matplotlib.pyplot as pl
from qpp_features.database import GEDDict

def create_different_k_ged_dist_matrix(
    basedir="/qpp/dataset/DBpedia_2016_12k_sample",
    database_path = "/data/dbpedia_ged.db",
):
    dist_matrix = GEDDict(file_name=database_path,update_improve=True)
    np.random.seed(42)
    # K=25
    run(
        train_path=f"{basedir}/train_sampled.tsv",
        val_path=f"{basedir}/val_sampled.tsv",
        test_path=f"{basedir}/test_sampled.tsv",
        dist_matrix=dist_matrix,
        save_dist_info_path=f'{basedir}/baseline',
        K=25,
        random_shuffel_max_iters=100,
        kmediods_max_iters=100,
        distance_matrix_file=f"{basedir}/baseline/distance_matrix.npy",
        vectorization_path=f"{basedir}/knn25/",
        center_cache_file="center_cache_file",
        save_cluster_obj_file=f"{basedir}/knn25/clusterobj.pickle"
    )
    
    # K=10
    """run(
        train_path=f"{basedir}/train_sampled.tsv",
        val_path=f"{basedir}/val_sampled.tsv",
        test_path=f"{basedir}/test_sampled.tsv",
        dist_matrix=dist_matrix,
        save_dist_info_path=basedir,
        K=10,
        random_shuffel_max_iters=100,
        kmediods_max_iters=100,
        distance_matrix_file="/qpp/data/new_features/distance_matrix.npy",
        vectorization_path=f"{basedir}/knn10/",
    )"""
    
def create_different_k_ged(
    basedir="/qpp/dataset/DBpedia_2016_12k_sample",
    dist_dir="/SPARQLBench/dbpedia2015_16/ged_dir_ordered2015_2016",
):
    np.random.seed(1242)
    # K=25
    run(
        train_path=f"{basedir}/train_sampled.tsv",
        val_path=f"{basedir}/val_sampled.tsv",
        test_path=f"{basedir}/test_sampled.tsv",
        dist_dir=dist_dir,
        save_dist_info_path=basedir,
        K=25,
        random_shuffel_max_iters=100,
        kmediods_max_iters=100,
        distance_matrix_file="/qpp/data/new_features/distance_matrix.npy",
        vectorization_path=f"{basedir}/knn25/",
    )

    # K=10
    run(
        train_path=f"{basedir}/train_sampled.tsv",
        val_path=f"{basedir}/val_sampled.tsv",
        test_path=f"{basedir}/test_sampled.tsv",
        dist_dir=dist_dir,
        save_dist_info_path=basedir,
        K=10,
        random_shuffel_max_iters=100,
        kmediods_max_iters=100,
        distance_matrix_file=dist_dir,
        vectorization_path=f"{basedir}/knn10/",
    )
    # K=20
    run(
        train_path=f"{basedir}/train_sampled.tsv",
        val_path=f"{basedir}/val_sampled.tsv",
        test_path=f"{basedir}/test_sampled.tsv",
        dist_dir=dist_dir,
        save_dist_info_path=basedir,
        K=20,
        random_shuffel_max_iters=100,
        kmediods_max_iters=100,
        distance_matrix_file="/qpp/data/new_features/distance_matrix.npy",
        vectorization_path=f"{basedir}/knn20/",

    )


# runner function
def run(
    # query_path="/SPARQLBench/dbpedia2015_16/ordered_queries2015_2016_clean_w_stat.tsv",
    train_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
    val_path="/qpp/dataset/DBpedia_2016_12k_sample/val_sampled.tsv",
    test_path="/qpp/dataset/DBpedia_2016_12k_sample/test_sampled.tsv",
    dist_dir="/SPARQLBench/dbpedia2015_16/ged_dir_ordered2015_2016",
    save_dist_info_path="/qpp/dataset/DBpedia_2016_12k_sample",
    K=25,
    dist_matrix=None,
    random_shuffel_max_iters=100,
    kmediods_max_iters=100,
    cluster_cach_file="cluster_cache_file",
    center_cache_file="center_cache_file",
    distance_matrix_file="/qpp/data/new_features/distance_matrix.npy",
    vectorization_path="/qpp/data/new_features/knn25/",
    train_graph_path="train_ged.csv",
    val_graph_path="val_ged.csv",
    test_graph_path="test_ged.csv",
    save_cluster_obj_file = None
    #center_cache_file = None
):
    base_dir = Path(train_path).parent
    train_df = pd.read_csv(train_path, sep="\t")
    val_df = pd.read_csv(val_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")
    cluster_lsq_datasets(
        train_df,
        val_df,
        test_df,
        train_graph_path,
        val_graph_path,
        test_graph_path,
        dist_dir,
        K=K,
        random_shuffel_max_iters=random_shuffel_max_iters,
        kmediods_max_iters=kmediods_max_iters,
        cluster_cach_file=cluster_cach_file,
        center_cache_file=center_cache_file,
        distance_matrix_file=distance_matrix_file,
        save_dist_info_path=save_dist_info_path,
        vectorization_path=vectorization_path,
        base_dir=base_dir,
        dist_matrix=dist_matrix,
        save_cluster_obj_file=save_cluster_obj_file,
    )

def cluster_lsq_datasets(
    train_df,
    val_df,
    test_df,
    train_graph_path,
    val_graph_path,
    test_graph_path,
    dist_dir_path,
    K=25,
    random_shuffel_max_iters=100,
    kmediods_max_iters=100,
    cluster_cach_file="cluster_cache_file",
    center_cache_file="center_cache_file",
    distance_matrix_file="distance_matrix.npy",
    vectorization_path="/qpp/data/new_features/knn25/",
    save_dist_info_path="/qpp/data/new_features",
    save_clust_obj=None,
    base_dir="",
    dist_matrix=None,
    save_cluster_obj_file = None
):
    train_ids = list(train_df["queryID"])
    val_ids = list(val_df["queryID"])
    test_ids = list(test_df["queryID"])

    dists_info_path = f"{base_dir}/dists_data.pickle"
    if dist_matrix is not None:
        new_dists = dist_matrix
        queries_to_remove = []
    else:
        if not os.path.exists(dists_info_path):
            print("Begin dist matrix computation")
            new_dists, queries_to_remove = recompute_lsq_dists(
                dist_dir_path, train_ids, val_ids, test_ids
            )
            print(f"Queries Removed from original {len(queries_to_remove)}")
            print("Finsihed dist matrix computation")
            with open(dists_info_path, "wb") as f:
                pickle.dump((new_dists, queries_to_remove), f)
        else:
            new_dists, queries_to_remove = pickle.load(open(dists_info_path, "rb"))
            print(f"Queries Removed from original {len(queries_to_remove)}")

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
        train_queries,
        train_ids,
        test_queries=test_queries,
        test_ids=test_ids,
        validation_queries=val_queries,
        val_ids=val_ids,
        K=K,
        random_shuffel_max_iters=random_shuffel_max_iters,
        kmediods_max_iters=kmediods_max_iters,
        cluster_cach_file=cluster_cach_file,
        center_cache_file=center_cache_file,
        distance_matrix_file=distance_matrix_file,
        vectorization_path=vectorization_path,
    )
    
    clust_time_start = time.time()
    i.load_training_queries()
    i.distance_matrix = new_dists

    print("Begin Clustering!")
    if not os.path.exists(center_cache_file):
        i.cluster_queries()
    # f.write("Time to create clusters: "+str(time.time()-clust_time_start)+"\n")
    print("Time to create clusters: " + str(time.time() - clust_time_start) + "\n")
    i.save_feature_representation(
        train_graph_ptn_path=train_graph_path,
        val_graph_ptn_path=val_graph_path,
        test_graph_ptn_path=test_graph_path,
        vec_path=True,
    )
    if not save_clust_obj == None:
        i.save(save_clust_obj)
    with open(save_cluster_obj_file, 'wb') as f:
        pickle.dump(i, f)


# fixed version of recomputation of distances, efficient version
def recompute_lsq_dists(dir_path, train_ids, val_ids, test_ids, save_path=""):
    start = time.time()
    files = sorted(os.listdir(dir_path))
    files = sorted([x for x in files if x.endswith(".csv")])
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

    with open(f"{save_path}lines.pickle", "wb") as f:
        pickle.dump(lines, f)
    # with open("lines.pickle", "rb") as f:
    #    lines = pickle.load(f)
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
            # to acoomodate queries where limit is removed from dbpedia 2016
            if "LIMIT" in ids:
                ids_index = lsq_mapping[ids.replace("LIMIT", "")]
            else:
                ids_index = lsq_mapping[ids]
        except KeyError:
            queries_w_no_dists.add(ids)
            continue
        for ids2 in new_order:
            try:
                # to acoomodate queries where limit is removed from dbpedia 2016
                if "LIMIT" in ids2:
                    ids2_index = lsq_mapping[ids2.replace("LIMIT", "")]
                else:
                    ids2_index = lsq_mapping[ids2]
                # ids2_index = lsq_mapping[ids2]
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
    return np.array(dist_matrix), list(queries_w_no_dists)


def find_k_queries_lsq_datasets(
    train_df,
    dist_dir_path,
    save_clust_obj=None,
    save_train_k_queries=None,
    save_dist_matrix=None,
):
    train_ids = list(train_df["queryID"])

    # new_dists= recompute_distmatrix(train_ids, val_ids, test_ids, id_to_dist_index, dists)
    print("Begin dist matrix computation")
    new_dists, queries_to_remove = recompute_lsq_dists_train(dist_dir_path, train_ids)
    print(f"Finsihed dist matrix computation: {queries_to_remove}")
    if save_dist_matrix is not None:
        np.save(save_dist_matrix, new_dists)

    train_df = train_df.set_index("queryID")
    train_df = train_df.drop(queries_to_remove, axis=0, errors="ignore")
    train_queries = list(train_df["queryString"])

    i = QueryClusterer(train_queries, train_ids)
    clust_time_start = time.time()
    i.load_training_queries()
    i.distance_matrix = new_dists

    print("Begin Clustering")
    center_idxs = i.cluster_queries()
    if save_train_k_queries != None:
        with open(save_train_k_queries, "w") as f:
            for center_idx in center_idxs:
                f.write(train_queries[center_idx] + "\n")
    # f.write("Time to create clusters: "+str(time.time()-clust_time_start)+"\n")
    print(
        "Time to select cluster queries: " + str(time.time() - clust_time_start) + "\n"
    )
    i.save_feature_representation(
        train_graph_ptn_path="traindata_cluster.csv",
        val_graph_ptn_path="val_cluster.csv",
        test_graph_ptn_path="test_cluster.csv",
        vec_path=True,
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


class QueryClusterer:
    def __init__(
        self,
        training_queries,
        train_ids,
        test_queries=None,
        test_ids=None,
        validation_queries=None,
        val_ids=None,
        distance=None,
        K=25,
        random_shuffel_max_iters=100,
        kmediods_max_iters=100,
        cluster_cach_file="cluster_cache_file",
        center_cache_file="center_cache_file",
        distance_matrix_file="distance_matrix.npy",
        vectorization_path="/qpp/data/new_features/knn25",
    ):
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

        """self.K = config.getint('QueryClustering','K')
        self.random_shuffel_max_iters = config.getint('QueryClustering','random_shuffel_max_iters')
        self.kmediods_max_iters = config.getint('QueryClustering','kmediods_max_iters')
        self.cluster_cach_file = config.get('QueryClustering','cluster_cach_file')
        self.center_cache_file = config.get('QueryClustering','center_cache_file')
        self.distance_matrix_file = config.get('QueryClustering','distance_matrix_file')
        self.vectorization_path = config.get('QueryClustering','vectorization_path')"""
        self.K = K
        self.random_shuffel_max_iters = random_shuffel_max_iters
        self.kmediods_max_iters = kmediods_max_iters
        self.cluster_cach_file = cluster_cach_file
        self.center_cache_file = center_cache_file
        self.distance_matrix_file = distance_matrix_file
        self.vectorization_path = vectorization_path
        os.system(f"mkdir -p {self.vectorization_path}")
        if self.vectorization_path[-1] != "/":
            self.vectorization_path += "/"

        self.center_ids = None
        self.id = None
        
        if os.path.exists(os.path.join(self.vectorization_path,self.center_cache_file)):
            self.center_ids = np.loadtxt(os.path.join(self.vectorization_path,self.center_cache_file), dtype=np.int64)

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

    def cluster_to_vec(self, query, id=None):
        vec = []
        for i in self.center_ids:
            #center_query = self.X[i]
            center_lsq_id = self.train_ids[i]
            if self.distance_matrix is not None:
                vec.append(self.distance_matrix[center_lsq_id, id])
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
            train_vec = self.cluster_to_vec(query, id=id)
            t = str(id) + "," + str(train_vec) + "\n"
            temp += t

        f.write(temp)
        f.close()

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


# A version of code from https://github.com/rhasan/query-performance/blob/master/clustering/k_mediods.py
class kMediod:
    def __init__(self, queries, distance):
        self.queries = queries
        self.dist = distance

    def initial_random_centers(self, X, K):
        randidx = np.random.permutation(range(np.size(X, 0)))
        if len(X.shape) == 1:
            centers = X[randidx[0:K]]  # , :
        else:
            centers = X[randidx[0:K], :]
        return (centers, randidx[0:K])

    def find_closest_centers(self, X, center_idxs, distance_matrix):
        K = np.size(center_idxs, 0)
        m = np.size(X, 0)
        idx = np.zeros(m, dtype=int)

        for i in range(m):
            min_d = np.inf
            min_j = -1
            for j in center_idxs:
                if j < 0:
                    continue

                d = distance_matrix[i, j]
                if min_d > d:
                    min_d = d
                    min_j = j
            idx[i] = min_j
            if min_j == -1:
                raise Exception()

        for j in center_idxs:
            idx[j] = j
        return idx

    def compute_centers(self, X, idx, center_idxs, distance_matrix):
        K = np.size(center_idxs, 0)
        moved_centers = np.zeros(K, dtype=int)
        i = 0
        for k in center_idxs:
            (x_indxs,) = np.where(idx[:] == k)
            # print "center:", k,"=",x_indxs
            # print "vals", X[x_indxs,:]

            min_cost = np.inf
            min_c = -1
            for c_indx in x_indxs:
                cost = 0.0
                for y_indx in x_indxs:
                    cost += distance_matrix[c_indx, y_indx]
                    # print "c_indx:", c_indx, "y_indx:",y_indx
                    # print "dist:", distance_matrix[c_indx,y_indx]
                # pri    nt "cost:", cost
                if min_cost > cost:
                    min_cost = cost
                    min_c = c_indx

            if np.size(x_indxs, 0) > 0:
                moved_centers[i] = min_c
                i += 1

            # print "min_cost:", min_cost
        return moved_centers[0:i]

    def k_mediods(self, X, initial_center_idxs, max_iters, distance_matrix):
        m = np.size(X, 0)
        K = np.size(initial_center_idxs, 0)
        center_idxs = initial_center_idxs
        previous_center_idxs = center_idxs
        idx = np.zeros(m, dtype=int)

        for i in range(max_iters):
            idx = self.find_closest_centers(X, center_idxs, distance_matrix)
            previous_center_idxs = center_idxs
            center_idxs = self.compute_centers(X, idx, center_idxs, distance_matrix)

            if (
                np.size(previous_center_idxs, 0) == np.size(center_idxs, 0)
                and np.size(center_idxs, 0) > 1
            ):
                if (previous_center_idxs == center_idxs).all() == True:
                    break
            elif (
                np.size(previous_center_idxs, 0) == np.size(center_idxs, 0)
                and np.size(center_idxs, 0) == 1
            ):
                if previous_center_idxs == center_idxs:
                    break

        if (
            np.size(previous_center_idxs, 0) == np.size(center_idxs, 0)
            and np.size(center_idxs, 0) > 1
        ):
            if (previous_center_idxs == center_idxs).all() == False:
                idx = self.find_closest_centers(X, center_idxs, distance_matrix)
        elif (
            np.size(previous_center_idxs, 0) == np.size(center_idxs, 0)
            and np.size(center_idxs, 0) == 1
        ):
            if previous_center_idxs == center_idxs:
                idx = self.find_closest_centers(X, center_idxs, distance_matrix)

        return (center_idxs, idx)

    def model_cost(self, X, idx, center_idxs, distance_matrix):
        K = np.size(center_idxs, 0)
        total_cost = 0.0
        for k in center_idxs:
            (k_cluster_x_indxs,) = np.where(idx[:] == k)
            # print k_cluster_x_indxs
            cost = 0.0
            for x_indx in k_cluster_x_indxs:
                cost += distance_matrix[k, x_indx]
            total_cost += cost
        return total_cost

    def initial_random_centers_cost_minimization(
        self, X, K, distance_matrix, random_shuffel_max_iters, kmediods_max_iters
    ):
        min_cost = np.inf

        for i in range(random_shuffel_max_iters):
            (initial_centers, initial_center_idxs) = self.initial_random_centers(X, K)
            (center_idxs, idx) = self.k_mediods(
                X, initial_center_idxs, kmediods_max_iters, distance_matrix
            )
            total_cost = self.model_cost(X, idx, center_idxs, distance_matrix)
            if min_cost > total_cost:
                min_cost = total_cost
                min_center_idxs = center_idxs

        return (min_center_idxs, min_cost)

    def elbow_method_choose_k_with_random_init_cost_minimization(
        self, X, max_K, distance_matrix, random_shuffel_max_iters, kmediods_max_iters
    ):
        cost_array = np.zeros(max_K, dtype=float)
        for K in range(1, max_K + 1):
            (init_center_idxs, cost) = self.initial_random_centers_cost_minimization(
                X, K, distance_matrix, random_shuffel_max_iters, kmediods_max_iters
            )
            (center_idxs, idx) = self.k_mediods(
                X, init_center_idxs, kmediods_max_iters, distance_matrix
            )
            total_cost = self.model_cost(X, idx, center_idxs, distance_matrix)
            cost_array[K - 1] = total_cost

            # print_clusters(X,idx,center_idxs)
            print("cost:", total_cost, "K:", K)

        K_vals = np.linspace(1, max_K, max_K)
        pl.plot(K_vals, cost_array)
        pl.plot(K_vals, cost_array, "rx", label="distortion")
        pl.show()

    def compute_symmetric_distance(self, X):
        m = np.size(X, 0)
        dist = np.zeros((m, m), dtype=float)
        start = time.time()
        # path = 'data/experiment1/distance_matrix_concurrent.txt'
        # max_i,max_j = -1,-1
        """if os.path.exists(path):
            with open(path,'r') as f:
                for line in f.readlines():
                    spl = line.split(',')
                    try:
                        dist[int(spl[0]),int(spl[1])] = float(spl[2])
                        if max_i < int(spl[0]):
                            max_i = int(spl[0])
                        if max_j < int(spl[1]):
                            max_j = int(spl[1])
                    except:
                        pass
        f = open(path,'a')"""
        for i in range(m):
            for j in range(m):
                """if i < max_i:
                    continue
                if (i% 10)==0:
                    print("Query {} of {} calculated \n".format(i,m))
                for j in range(i + 1, m):
                    if j < max_j:
                        continue
                    #if (int(time.time()-start)%10) == 0:
                    print("{},{}: Time calculating distance matrix: {}\n".format(i,j,time.time()-start))
                """
                try:
                    dist[i, j] = self.dist(X[i], X[j])
                    dist[j, i] = dist[i, j]
                    # print "distance between", X[i], "and ", X[j], ":",dist[i,j]
                except:
                    dist[i, j] = np.inf
                    dist[j, i] = np.inf
        return dist

    def print_clusters(self, X, idx, center_idxs):
        for k in center_idxs:
            print("Cluster:", X[k])
            print(X[idx[:] == k])


import sys

if __name__ == "__main__":
    create_different_k_ged_dist_matrix(basedir=sys.argv[1], database_path=sys.argv[2] )
    #previous ged calculation version
    #create_different_k_ged(basedir=sys.argv[1], dist_dir=sys.argv[2])
