import os
import pandas as pd
from qpp_new.data_prep import SVMDataPrepper
from qpp_new.models.svm import SVMTrainer
from sklearn import svm
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

np.warnings = warnings


class TwoStepSVM:
    def __init__(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        resultDir: str,
        prepper: SVMDataPrepper,
    ) -> None:
        self.train = train
        self.val = val
        self.test = test
        self.resultDir = resultDir
        self.resultDirCls = f"{resultDir}/cls"
        self.prepper = prepper
        os.system(f"mkdir -p {self.resultDirCls}")

    def trainer(self, classes=10):
        print(f"Beginning the training of the Time classifier")
        """self.cluster = XMeansCluster(
            np.reshape(self.train["time"].to_numpy(), (-1, 1)), classes
        )"""
        """self.cluster = KMeansCluster(
            self.train["time"].to_numpy().reshape(-1, 1), classes
        )"""
        self.cluster = HeuristicCluster()
        train = self.cluster.assign_class(self.train, time_col="time")
        val = self.cluster.assign_class(self.val, time_col="time")
        test = self.cluster.assign_class(self.test, time_col="time")
        train = self.cluster.assign_class(train, time_col="time")
        print(train["class_labels"].value_counts())
        # TODO: remove test code when not necessary anymore
        """train = train.iloc[:100]
        for i in range(20):
            train["class_labels"].iloc[i] = 1"""

        cls = TimeClassifer(train, val, test)
        (
            train["pred_cls"],
            val["pred_cls"],
            test["pred_cls"],
        ) = cls.trainer()
        train.to_csv(f"{self.resultDirCls}/train_cls.csv", index=False)
        val.to_csv(f"{self.resultDirCls}/val_cls.csv", index=False)
        test.to_csv(f"{self.resultDirCls}/test_cls.csv", index=False)
        print(train["pred_cls"].value_counts())
        classes = []
        for lst in [train, val, test]:
            classes.extend(np.unique(lst["pred_cls"]))
        classes = list(set(classes))
        print(f"Starting k SVM training")
        class_data = {}
        for c in classes:
            train_sub = train[train["pred_cls"] == c]
            train_sub: pd.DataFrame = train_sub.drop(
                columns=["class_labels", "pred_cls"]
            )
            val_sub = val[val["pred_cls"] == c]
            val_sub: pd.DataFrame = val_sub.drop(columns=["class_labels", "pred_cls"])
            test_sub = test[test["pred_cls"] == c]
            test_sub: pd.DataFrame = test_sub.drop(columns=["class_labels", "pred_cls"])
            class_data[c] = (train_sub, val_sub, test_sub)
        # first divide train val test to each class:
        # print(cls.predict_df(train))
        for c in class_data.keys():
            train_sub, val_sub, test_sub = class_data[c]
            resultDir = f"{self.resultDir}/{c}/"
            os.system(f"mkdir -p {resultDir}")
            trainer = SVMTrainer(train, val, test, resultDir)
            trainer.trainer()


class BaseSVM:
    def scale_log_data_targets(self, df_train, df_val, df_test, target_col="time"):
        if target_col is not None:
            y_train = df_train[target_col].values.reshape(-1, 1)
            y_val = df_val[target_col].values.reshape(-1, 1)
            y_test = df_test[target_col].values.reshape(-1, 1)

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
        else:
            y_train = df_train["class_labels"].values.reshape(-1, 1)
            y_val = df_val["class_labels"].values.reshape(-1, 1)
            y_test = df_test["class_labels"].values.reshape(-1, 1)

            y_val_log = y_val
            y_train_log = y_train
            y_test_log = y_test
        retain = [
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
        # adds ged features
        for x in df_train.columns:
            if x.startswith("cls_"):
                retain.append(x)
        return (
            df_train[retain],
            df_val[retain],
            df_test[retain],
            y_train,
            y_val,
            y_test,
            y_train_log,
            y_val_log,
            y_test_log,
        )

    """scalery, y_train_log_std, y_val_log_std, y_test_log_std = self.normalize_target(
        y_train_log, y_val_log, y_test_log
    )
    """

    def normalize_target(self, y_train_log, y_val_log, y_test_log):
        """
        Normalize data using StandardScaler.

        return scaler object; values of train,val and test sets standarized.
        """
        # Standarización del target
        scaler = StandardScaler()
        y_train_log_std = scaler.fit_transform(y_train_log)
        y_val_log_std = scaler.transform(y_val_log)
        y_test_log_std = scaler.transform(y_test_log)
        return scaler, y_train_log_std, y_val_log_std, y_test_log_std

    def get_prediction(self, sv, scalery, x_train):
        y_train_hat_svr = scalery.inverse_transform(sv.predict(x_train).reshape(-1, 1))
        return y_train_hat_svr


class TimeClassifer(BaseSVM):
    def __init__(self, train, val, test, kernel="rbf", nu=0.5):
        self.train = train
        self.val = val
        self.test = test

        # self.svm = svm.SVC(kernel=kernel)  # ,nu=nu
        self.svm = svm.NuSVC(kernel=kernel, nu=nu)  # ,nu=nu

    # def predict(self, x):
    #    return self.svm.predict([x])
    def trainer(self):
        (
            self.x_train,
            self.x_val,
            self.x_test,
            _,
            _,
            _,
            self.y_train_log,
            self.y_val_log,
            self.y_test_log,
        ) = self.scale_log_data_targets(
            self.train, self.val, self.test, target_col=None
        )
        """(
            self.scalery,
            self.y_train_log_std,
            self.y_val_log_std,
            self.y_test_log_std,
        ) = self.normalize_target(y_train_log, y_val_log, y_test_log)"""
        """X = train_df.drop(columns=["class_labels"])  # Features
        X = X.drop(columns=["time"])  # Features
        y = train_df["class_labels"]  # Target"""

        self.svm.fit(self.x_train.values, self.y_train_log)
        train_pred_cls = self.predict(self.x_train)
        val_pred_cls = self.predict(self.x_val)
        test_pred_cls = self.predict(self.x_test)
        return train_pred_cls, val_pred_cls, test_pred_cls

    def predict(self, X):
        # X = df.drop(columns=["class_labels"])  # Features
        # X = X.drop(columns=["time"])
        y_train_hat_svr = self.svm.predict(X.values)
        return y_train_hat_svr
        return self.get_prediction(self.svm, self.scalery, X.values)


class BaseCluster:
    def assign_class(self, df: pd.DataFrame, time_col="mean_latency"):
        classes = []
        for i, row in df.iterrows():
            classes.append(self.get_cluster(row[time_col]))
        df["class_labels"] = classes
        return df


class XMeansCluster(BaseCluster):
    def __init__(self, train_run_times, initial_k, seed=25):
        initial_centers = kmeans_plusplus_initializer(
            train_run_times, initial_k, random_state=seed
        ).initialize()
        # self.kmean = xmeans(train_run_times,initial_centers,initial_k)
        try:
            self.kmean = xmeans(train_run_times, initial_centers, initial_k)
        except OSError:
            self.kmean = xmeans(train_run_times, initial_centers, initial_k)

        self.kmean.process()
        self.clusters = self.kmean.get_clusters()
        self.centers = self.kmean.get_centers()
        # self.centers = self.centers[0]

    def get_cluster(self, latency):
        clust = -1
        closest = 10000000000000
        clust_no = -1
        for s in self.centers:
            clust_no += 1
            if abs(s[0] - latency) < closest:
                clust = clust_no
                closest = abs(s[0] - latency)
        return clust


class KMeansCluster(BaseCluster):
    def __init__(self, train_run_times, initial_k, seed=25):
        self.kmeans = KMeans(n_clusters=initial_k, random_state=seed)
        self.kmeans.fit(train_run_times)
        # self.centers = self.centers[0]

    def get_cluster(self, latency):
        latency = np.array([latency]).reshape(-1, 1)
        return self.kmeans.predict(latency)


class HeuristicCluster(BaseCluster):
    def __init__(self) -> None:
        self.ranges = {}
        self.ranges[(-1, 0.01)] = 0
        self.ranges[(0.01, 0.1)] = 1
        self.ranges[(0.1, 1)] = 2
        self.ranges[(1, 10)] = 3
        self.ranges[(10, 18000)] = 4  # 1800 is the time out limit

    def get_cluster(self, latency):
        for k in self.ranges.keys():
            t_start, t_end = k
            if t_start < latency and latency <= t_end:
                return self.ranges[k]
        raise Exception("Should not be possible to iterate all the way through")

    def get_labels(self):
        return [0, 1, 2, 3, 4]
