import os
from qpp_new.pred_n_qs import SPARQLDistCalcular


def extract_ged_queries(folder, train_sampled_file):
    import pandas as pd
    def load_cluster_file( fp):
        data = {}
        with open(fp, "r") as f:
            columns = f.readline().replace("\n", "").split(",")
            data[columns[0]] = []
            for l_i, line in enumerate(f.readlines()):
                line = line.replace("\n", "").replace("[", "").replace("]", "")
                spl = line.split(",", 1)
                data[columns[0]].append(spl[0])
                spl = spl[1].split(",")
                for i, x in enumerate(spl):
                    if l_i == 0:
                        data["cls_{}".format(i)] = []
                    #data["cls_{}".format(i)].append((1 / (1 +  float(x))))
                    data["cls_{}".format(i)].append(float(x))
        df = pd.DataFrame.from_dict(data)
        return df

    center_cache = os.path.join(folder, 'center_cache_file')
    train_ged_file = os.path.join(folder, 'train_ged.csv')
    train_g_df = pd.read_csv(train_ged_file)
    dist_calculator = SPARQLDistCalcular()

    #Train query data
    train_df = pd.read_csv(train_sampled_file, sep='\t')

    center_idxs = []
    with open(center_cache, 'r') as f:
        for l in f.readlines():
            center_idxs.append(int(l))
    ged_map = {}
    for row_no, (idx, row) in enumerate(train_g_df.iterrows()):
        if row_no in center_idxs:
            ged_map[row_no] = idx[0]

    # Extract the query text in order for the calculation of ged distance features.
    ged_queries = []
    for x in center_idxs:
        ged_queries.append(train_df[train_df['queryID'] == ged_map[x]]['queryString'].iloc[0])
    ged_query_file = os.path.join(folder, 'ged_queries.txt')

    with open(ged_query_file, 'w') as f:
        for q in ged_queries:
            f.write(q+'\n')

    train_g_df = load_cluster_file(train_ged_file)
    def test_ged_queries(test_idx=0):
        test_id = train_df['queryID'].iloc[test_idx]
        test_query_txt = train_df[train_df['queryID'] == test_id]['queryString'].iloc[0]
        geds = []
        for ged_q in ged_queries:
            geds.append(dist_calculator.distance_ged(ged_q, test_query_txt))

        existing_ged = train_g_df[train_g_df['id'] == test_id]
        ex_ged = []
        for i in range(0,25):
            ex_ged.append(existing_ged[f'cls_{i}'].iloc[0])
        existing_ged = ex_ged
        same_values = 0
        for i in range(0, 25):
            if existing_ged[i] == geds[i]:
                same_values += 1
        return same_values
    def is_ged_same(test_idx):
        v = test_ged_queries(test_idx=test_idx)
        if v == 25:
            return True
        return False

    not_same_ged = []
    for i in range(10000):
        if not is_ged_same(i):
            not_same_ged.append(i)


if __name__ == "__main__":
    extract_ged_queries('/data/DBpedia_3_class_full/baseline/knn25', '/data/DBpedia_3_class_full/train_sampled.tsv')
    extract_ged_queries('/data/wikidata_3_class_full/baseline/knn25', '/data/wikidata_3_class_full/train_sampled.tsv')
