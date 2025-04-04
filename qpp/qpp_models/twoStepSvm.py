from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import pandas as pd, numpy as np
from sklearn import svm

seed = 25
np.random.seed(seed)


class TwoStepClassifier:
    def __init__(
        self, k, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        self.cluster = XMeansCluster()
        self.K = k  # or from xmeans
        self.timeClassifier = TimeClassifer()
        self.timeRegressions = [SVMregression() for x in range(self.K)]


class XMeansCluster:
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

    def assign_class(self, df: pd.DataFrame, time_col="mean_latency"):
        classes = []
        for i, row in df.iterrows():
            classes.append(self.get_cluster(row[time_col]))
        df["class_labels"] = classes
        return df


class TimeClassifer:
    def __init__(self, data, labels, kernel, nu=0.2):
        self.svm = svm.SVC(kernel=kernel)  # ,nu=nu
        # self.svm = svm.NuSVC(kernel=kernel, nu=nu) #,nu=nu
        self.svm.fit(data, labels)

    def predict(self, x):
        return self.svm.predict([x])


class SVMregression:
    def __init__(self):
        self.svr = svm.SVR()

    def train(self, Xs, Ys):
        self.svr.fit(Xs, Ys)


if __name__ == "__main__":
    df = pd.read_csv("/qpp/dataset/DBpedia_2016_sampled/train_sampled.tsv", sep="\t")
    run_times = df["mean_latency"].to_numpy()
    run_times = np.reshape(run_times, (-1, 1))
    print(f"Runtimes - Min: {run_times.min()}, Max: {run_times.max()}")
    cluster = XMeansCluster(run_times, 5)
    print(cluster.centers)
    df = cluster.assign_class(df)
    print(df.columns)
    # print(f"Cluster for 10s : {cluster.get_cluster(10)}")
