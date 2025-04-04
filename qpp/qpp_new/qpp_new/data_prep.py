import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import StandardScaler


class BaseDataPrepper:
    def get_used_algebra_cols(self):
        return [
            "triple",
            "bgp",
            "leftjoin",
            "union",
            "filter",
            "graph",
            "extend",
            "minus",
            "order",
            "project",
            "distinct",
            "group",
            "slice",
            "treesize",
        ]

    def get_all_extracted_algebra_cols(self):
        return [
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
        ]


class SVMDataPrepper(BaseDataPrepper):
    def __init__(
        self,
        train_ged_path=None,
        val_ged_path=None,
        test_ged_path=None,
        train_algebra_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
        val_algebra_path="/qpp/dataset/DBpedia_2016_12k_sample/val_sampled.tsv",
        test_algebra_path="/qpp/dataset/DBpedia_2016_12k_sample/test_sampled.tsv",
    ) -> None:
        self.train_ged_path = train_ged_path
        self.val_ged_path = val_ged_path
        self.test_ged_path = test_ged_path
        self.train_algebra_path = train_algebra_path
        self.val_algebra_path = val_algebra_path
        self.test_algebra_path = test_algebra_path

    # to get feature data
    def prepare(self):
        train, val, test = self.process_svm_data()
        # train, val,test = svm_pre_process(train,val,test)
        train, val, test = (
            self.rename_time(train, version=3),
            self.rename_time(val, version=3),
            self.rename_time(test, version=3),
        )
        # This is not original code
        # train = train.dropna()
        # val = val.dropna()
        # test = test.dropna()
        self.train = train
        self.val = val
        self.test = test
        return train, val, test

    def rename_time(self, df: pd.DataFrame, version=3):
        if version == 1:
            df["time"] = df["duration"]
            df = df.drop(columns=["duration"])
            return df
        elif version == 3:
            if "execTime" in df.columns:
                df["time"] = df["execTime"]
                df = df.drop(columns=["execTime"])
                self.latency_col = "time"
                return df
        elif version == 2:
            df["time"] = df["mean_latency"]
            df = df.drop(columns=["mean_latency"])
            return df
        return df

    def process_svm_data(
        self,
    ):
        # TODO:Not all queries have these features
        train_graph = self.load_cluster_file(self.train_ged_path)
        val_graph = self.load_cluster_file(self.val_ged_path)
        test_graph = self.load_cluster_file(self.test_ged_path)
        # l_tr,l_v,_lte = len(train_graph), len(test_graph), len(val_graph)
        # print(len(train_graph))
        print(f"Unique {len(train_graph['id'].unique())} of {len(train_graph)}")
        print(f"Unique {len(test_graph['id'].unique())} of {len(test_graph)}")
        print(f"Unique {len(val_graph['id'].unique())} of {len(val_graph)}")
        # RThis should be handled by sampler now on new queries.
        # train_graph = train_graph.drop_duplicates()
        # test_graph = test_graph.drop_duplicates()
        # val_graph = val_graph.drop_duplicates()
        # print(len(train_graph))

        # print(f"Dublicartes in graph train: {len(train_graph['id'].unique())-len(train_graph['id'])}")

        # temporily renaming these files
        # train_df = pd.read_csv(args.data_path+'train.tsv',sep='\t').drop_duplicates()
        # train_df = pd.read_csv(self.train_algebra_path, sep="\t")#.drop_duplicates()
        train_df = pd.read_csv(self.train_algebra_path)  # .drop_duplicates()
        train_df = train_df.drop(columns=[c for c in train_df.columns if 'Unnamed' in c])
        
        # val_df = pd.read_csv(args.data_path+'val.tsv',sep='\t').drop_duplicates()
        # val_df = pd.read_csv(self.val_algebra_path, sep="\t")#.drop_duplicates()
        val_df = pd.read_csv(self.val_algebra_path)  # .drop_duplicates()
        val_df = val_df.drop(columns=[c  for c in val_df.columns if 'Unnamed' in c])
        # test_df = pd.read_csv(args.data_path+'test.tsv',sep='\t').drop_duplicates()
        # test_df = pd.read_csv(self.test_algebra_path, sep="\t")#.drop_duplicates()
        test_df = pd.read_csv(self.test_algebra_path)  # .drop_duplicates()
        test_df = test_df.drop(columns=[c  for c in test_df.columns if 'Unnamed' in c])
        len_train = len(train_df)
              
        print(len(train_df), len(test_df), len(val_df))
        # print(train_graph.columns)
        # print(train_df.columns)

        train_df = train_df.merge(
            train_graph, left_on="queryID", right_on="id", how="left"
        )

        print(f"Differnece is {len_train-len(train_df)}")

        val_df = val_df.merge(val_graph, left_on="queryID", right_on="id", how="left")
        test_df = test_df.merge(
            test_graph, left_on="queryID", right_on="id", how="left"
        )
        print(len(train_df), len(test_df), len(val_df))
        return train_df, val_df, test_df

    def load_cluster_file(self, fp):
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
                    data["cls_{}".format(i)].append((1 / (1 + float(x))))
                # for value,col_name in zip(spl,columns):
                #    data[col_name].append(value)
        df = pd.DataFrame.from_dict(data)
        return df


class NNDataPrepper(SVMDataPrepper):
    def __init__(
        self,
        train_ged_path=None,
        val_ged_path=None,
        test_ged_path=None,
        train_algebra_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
        val_algebra_path="/qpp/dataset/DBpedia_2016_12k_sample/val_sampled.tsv",
        test_algebra_path="/qpp/dataset/DBpedia_2016_12k_sample/test_sampled.tsv",
        latency_col="mean_latency",
        filter_join_data_path="/qpp/dataset/DBpedia_2016_extra/extra",
    ) -> None:
        super().__init__(
            train_ged_path,
            val_ged_path,
            test_ged_path,
            train_algebra_path,
            val_algebra_path,
            test_algebra_path,
        )
        self.latency_col = latency_col
        self.filter_join_data_path = filter_join_data_path

    def prepare(self):
        self.train_df, self.val_df, self.test_df = super().prepare()
        self.train_df, self.val_df, self.test_df = self.process_nn_data()
        train, val, test = (
            self.rename_time(self.train_df, version=3),
            self.rename_time(self.val_df, version=3),
            self.rename_time(self.test_df, version=3),
        )
        # time out resultset can have nan value for either bad queries/ time outs/ outof memory
        # train = train.dropna()
        # val = val.dropna()
        # test = test.dropna()
        self.train_df, self.val_df, self.test_df = train, val, test
        for c in self.train_df.columns:
            self.train_df[c] = self.train_df[c].fillna(0)
        for c in self.val_df.columns:
            self.val_df[c] = self.val_df[c].fillna(0)
        for c in self.test_df.columns:
            self.test_df[c] = self.test_df[c].fillna(0)
                
        return self.train_df, self.val_df, self.test_df

    def process_nn_data(self):
        data_tpf = pd.read_csv(self.filter_join_data_path, sep="\t")
        col_to_drop = []
        for col in ["Unnamed: 4", "execTime", "duration"]:
            if col in data_tpf.columns:
                col_to_drop.append(col)

        data_tpf = data_tpf.drop(columns=col_to_drop)
        data_tpf_clean = NNDataPrepperUtils.process_extra(data_tpf)
        data_tpf_clean = data_tpf_clean.drop_duplicates()

        X_train_extended = self.train_df.merge(
            data_tpf_clean, left_on="queryID", right_on="id", how="left"
        )
        X_val_extended = self.val_df.merge(
            data_tpf_clean, left_on="queryID", right_on="id", how="left"
        )
        X_test_extended = self.test_df.merge(
            data_tpf_clean, left_on="queryID", right_on="id", how="left"
        )

        X_train_extended["id"] = X_train_extended["queryID"]
        X_val_extended["id"] = X_val_extended["queryID"]
        X_test_extended["id"] = X_test_extended["queryID"]

        X_train_extended = X_train_extended.drop(columns=["id_y", "id_x", "queryID"])
        X_val_extended = X_val_extended.drop(columns=["id_y", "id_x", "queryID"])
        X_test_extended = X_test_extended.drop(columns=["id_y", "id_x", "queryID"])

        X_train_extended = X_train_extended.set_index("id")
        X_val_extended = X_val_extended.set_index("id")
        X_test_extended = X_test_extended.set_index("id")

        X_train_extended = X_train_extended.drop(columns=["join"])
        X_val_extended = X_val_extended.drop(columns=["join"])
        X_test_extended = X_test_extended.drop(columns=["join"])
        (
            scaled_df_train,
            scaled_df_val,
            scaled_df_test,
            scaler,
        ) = NNDataPrepperUtils.normalizaAlgebra(
            X_train_extended, X_val_extended, X_test_extended, returnScaler=True
        )
        col_gpm = [x for x in X_train_extended if x.startswith("cls_")]
        col_gpm.append(self.latency_col)
        x_train, x_val, x_test = NNDataPrepperUtils.joinAlgebraGPM(
            scaled_df_train,
            scaled_df_val,
            scaled_df_test,
            X_train_extended[col_gpm],
            X_val_extended[col_gpm],
            X_test_extended[col_gpm],
        )

        return x_train, x_val, x_test


# From existing works
class NNDataPrepperUtils:
    def process_extra(data_tpf):
        data_tpf["predicates"] = data_tpf["predicates"].apply(
            lambda x: NNDataPrepperUtils.filter_only_string_non_empty(x)
        )
        data_tpf["joinsv1"] = data_tpf["joins"].apply(
            lambda x: NNDataPrepperUtils.get_joins(x)
        )
        data_tpf["joins_count"] = data_tpf["joinsv1"].apply(lambda x: len(x))
        
        data_tpf["predicates_select"] = data_tpf["predicates"].apply(
            lambda x: NNDataPrepperUtils.pred_2_hist(x)
        )
        
        
        data_tpf["filter_uri"] = data_tpf["predicates_select"].apply(
            lambda x: NNDataPrepperUtils.get_filter_by_type(x, "uri")
        )
        
        
        data_tpf["filter_num"] = data_tpf["predicates_select"].apply(
            lambda x: NNDataPrepperUtils.get_filter_by_type(x, "num")
        )
        
        
        data_tpf["filter_literal"] = data_tpf["predicates_select"].apply(
            lambda x: NNDataPrepperUtils.get_filter_by_type(x, "literal")
        )
        
        data_tpf_clean = data_tpf[
            ["id", "joins_count", "filter_uri", "filter_num", "filter_literal"]
        ]
        return data_tpf_clean

    def filter_only_string_non_empty(x):
        if type(x) == float:
            return False
        return x.replace("EMPTY_VALUE", "")

    def uri_2_index_seq(x, uri2Index):
        """transform uris to sequences"""
        lista = [uri2Index[a] for a in x]
        #     print(lista)
        return lista

    def get_joins(x):
        if type(x) != str:
            return []
        lista = [a for a in x.split("??44??") if a != ""]
        if len(lista) == 1:
            return []
        valsjoins = []
        for i in range(len(lista))[::2]:
            valsjoins.append([lista[i], lista[i + 1]])
        return valsjoins

    def convert_string2dict(x):
        """Convertir string en un dict donde las keys son los pareces y los valores los impares, ignora llaves sin valores y llaves empty"""
        if type(x) is float and math.isnan(x):
            return {}
        if x is None:
            return {}
        values = x.split("B?B")
        rango = range(len(values))
        resp = {}
        for i in rango:
            if i % 2 == 0 and values[i] != "":
                try:
                    resp[values[i]] = values[i + 1]
                except:
                    print("error con ", values[i], x)
        return resp

    def get_sum_values(x):
        total = 0
        for val in list(x.values()):
            total += float(val)
        return total

    # select variable is not returned in this implementation originally
    def selectivity(val, low, high, distinct, operator):
        """
        Calculate Select Estimation  sel_A<=c"""
        select = 1
        if operator == "<=":
            select = (val - low) / (high - low)
        elif operator == ">=":
            select = 1 - ((val - low) / (high - low))
        elif operator == "=":
            select = 1 / distinct
        else:
            select = (distinct - 1) / (distinct)
        return select

    def get_hist_value(data_tpf_histdata, predicate, operator, on, value):
        """
        predicate: predicado uri para extraer la información del histograma.
        operator:
        value: valor para filtrar, si es
        return el valor.
        """
        if value == "ALL":
            # Si es ALL la selectividad es 1
            return 1

        data = data_tpf_histdata[data_tpf_histdata["predicate"] == predicate]
        data = data[data["on"] == on]
        if data.shape[0] > 0:
            hist_data = data["hist_array"].values[0]
            distinct = len(hist_data)
            if distinct == 0:
                # Todo, revisar que pasa en caso de que distinct es cero
                return 0
            type_row = data["type"].values[0]
            if type_row == "uri":
                # Todo ver que se hace con la selectividad, aqui, si es el total de duplicados sobre el total de elementos o total
                if value in hist_data:
                    # return float(hist_data[value])/distinct
                    return float(1) / distinct
                else:
                    # Suponemos que el valor es 1 si no se muestreo en el hist.
                    return float(1) / distinct
            elif type_row == "numeric":
                min_v = float(hist_data["min"])
                max_v = float(hist_data["max"])
                distinct_v = float(hist_data["distinct"])
                print(hist_data)
                try:
                    value = float(value)
                except ValueError:
                    print("[{}] input is not a number. It's a string".format(value))
                    # Todo verificar que hacer cuando el supuesto número es un str no numerico, de momento devolvemos  1/distinct
                    return float(1) / distinct_v
                return NNDataPrepperUtils.selectivity(
                    value, min_v, max_v, distinct_v, operator
                )
        #      If not return max selectivity
        return 1

    def pred_2_hist(x):
        resp = {"uri": 0, "num": 0, "literal": 0}
        if type(x) != str:
            return resp
        data = [el for el in x.split("??44??") if el != ""]
        
        #     print(data)
        #     get_hist_value(predicate, operator, value)

        for i in range(len(data)):
            if i % 4 == 0:
                uri = data[i]
                operator = data[i + 1]
                on = data[i + 2]
                try:
                    val = data[i + 3]
                    #             select = get_hist_value(uri, operator, on, val)
                    if val.startswith("http"):
                            resp["uri"] += 1
                    elif val.isnumeric():
                            resp["num"] += 1
                    elif val != "ALL":
                            resp["literal"] += 1
                except IndexError:
                    pass

        return resp

    def get_filter_by_type(x, typeOf):
        try:
            return x[typeOf]
        except:
            print(x)

    def get_std_data_cols(train_df):
        x_columns_to_norm = [
            "triple",
            "bgp",
            "leftjoin",
            "union",
            "filter",
            "graph",
            "extend",
            "minus",
            "order",
            "project",
            "distinct",
            "group",
            "slice",
            "treesize",
        ]  # extra data , 'filter_uri', 'filter_num', 'filter_literal','joins_count'
        for x in train_df.columns:
            if "cls" in x:
                x_columns_to_norm.append(x)

        cols = x_columns_to_norm
        print(cols)
        return cols
        # Standarizar

    def normalizaAlgebra(
        X_train_extended, X_val_extended, X_test_extended, returnScaler=False
    ):
        # no reduced in my data
        x_columns_to_norm = [
            "triple",
            "bgp",
            "leftjoin",
            "union",
            "filter",
            "graph",
            "extend",
            "minus",
            "order",
            "project",
            "distinct",
            "group",
            "slice",
            "treesize",
            "joins_count",
            "filter_uri",
            "filter_num",
            "filter_literal",
        ]  #'reduced'
        scalerx = StandardScaler()
        x_train_scaled = scalerx.fit_transform(X_train_extended[x_columns_to_norm])
        x_val_scaled = scalerx.transform(X_val_extended[x_columns_to_norm])
        # x_test_scaled = scalerx.fit_transform(X_test_extended[x_columns_to_norm]) //dobble check whether they do this in source, because it's wrong.
        x_test_scaled = scalerx.transform(X_test_extended[x_columns_to_norm])

        scaled_df_train = pd.DataFrame(
            x_train_scaled, index=X_train_extended.index, columns=x_columns_to_norm
        )
        scaled_df_val = pd.DataFrame(
            x_val_scaled, index=X_val_extended.index, columns=x_columns_to_norm
        )
        scaled_df_test = pd.DataFrame(
            x_test_scaled, index=X_test_extended.index, columns=x_columns_to_norm
        )
        if returnScaler:
            return scaled_df_train, scaled_df_val, scaled_df_test, scalerx
        return scaled_df_train, scaled_df_val, scaled_df_test

    def joinAlgebraGPM(Train_alg, Val_alg, Test_alg, Train_gpm, Val_gpm, Test_gpm):
        mergedTrain = Train_alg.merge(Train_gpm, left_index=True, right_index=True)
        mergedVal = Val_alg.merge(Val_gpm, left_index=True, right_index=True)
        mergedTest = Test_alg.merge(Test_gpm, left_index=True, right_index=True)
        return mergedTrain, mergedVal, mergedTest

    def scale_log_data_targets(df_train, df_val, df_test):
        y_train = df_train["time"].values.reshape(-1, 1)
        y_val = df_val["time"].values.reshape(-1, 1)
        y_test = df_test["time"].values.reshape(-1, 1)

        y_val_log = np.log(y_val)
        y_train_log = np.log(y_train)
        y_test_log = np.log(y_test)

        y_train_log_min = np.min(y_train_log)
        y_train_min = np.min(y_train)

        y_train_log_max = np.max(y_train_log)
        y_train_max = np.max(y_train)

        print("targets min:{} max: {}".format(y_train_min, y_train_max))
        print(
            "targets in log scale min:{} max: {}".format(
                y_train_log_min, y_train_log_max
            )
        )

        return (
            df_train.drop(columns=["time"]),
            df_val.drop(columns=["time"]),
            df_test.drop(columns=["time"]),
            y_train,
            y_val,
            y_test,
            y_train_log,
            y_val_log,
            y_test_log,
        )


def check_missing_ged_fet(df):
    cls_cols = [x for x in df.columns if x.startswith("cls_")]
    for c in cls_cols:
        df = df[~df[c].isna()]
    return df


class OriginalDataPrepper:
    def __init__(self) -> None:
        train_algebra_feat = "/query-performance/dbpsb/x_features.txt"
        val_algebra_feat = "/query-performance/dbpsb/xval_features.txt"
        test_algebra_feat = "/query-performance/dbpsb/xtest_features.txt"

        train_executions = "/query-performance/dbpsb/y_time.txt"
        val_executions = "/query-performance/dbpsb/yval_time.txt"
        test_executions = "/query-performance/dbpsb/ytest_time.txt"

        train_ged = "/query-performance/dbpsb-K20/x_features_simvec.txt"
        val_ged = "/query-performance/dbpsb-K20/xval_features_simvec.txt"
        test_ged = "/query-performance/dbpsb-K20/xtest_features_simvec.txt"

        self.train = self.prepare_features(
            train_algebra_feat, train_ged, train_executions
        )
        self.val = self.prepare_features(val_algebra_feat, val_ged, val_executions)
        self.test = self.prepare_features(test_algebra_feat, test_ged, test_executions)

    def prepare_features(self, algebra, ged, executions):
        alg = pd.read_csv(algebra)
        ged = pd.read_csv(ged)
        executions = pd.read_csv(executions)
        df = pd.concat([alg, ged, executions], axis=1)
        df["id"] = df.index
        df["time"] = df["ex_time"]
        del df["ex_time"]
        for c in [x for x in df.columns if x.startswith("pcs")]:
            label = f"cls_{c[3:]}"
            df[label] = df[c]
            del df[c]
        # filt = df['path+']!=0.0
        # print(df[filt])
        return df

    def prepare(self):
        return self.train, self.val, self.test


if __name__ == "__main__":
    d = OriginalDataPrepper()
    exit()

    d = SVMDataPrepper(
        train_ged_path="/qpp/dataset/DBpedia_2016_12k_sample/knn10/train_ged.csv",
        val_ged_path="/qpp/dataset/DBpedia_2016_12k_sample/knn10/val_ged.csv",
        test_ged_path="/qpp/dataset/DBpedia_2016_12k_sample/knn10/test_ged.csv",
    )

    train, val, test = d.prepare()
    print(train.columns)
    # print(len(check_missing_ged_fet(train)))
    # print(len(check_missing_ged_fet(val)))
    # print(len(check_missing_ged_fet(test)))
