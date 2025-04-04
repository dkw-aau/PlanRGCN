import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import math
import warnings
from qpp_new.preprocessing.query_cls import QueryCls

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.simplefilter("ignore")


class Plotter:
    def __init__(
        self,
        plan_rgcn_path="/PlanRGCN/dbpedia2016/results",
        dataset_path="/qpp/dataset/DBpedia_2016_12k_simple_opt_filt",
        is_lsq=True,
        interval=[
            "0 - 1",
            " > 1",
        ],
        average="macro",
    ) -> None:
        self.is_lsq = is_lsq
        self.plan_path = plan_rgcn_path

        # self.ql = pd.read_csv(query_log, sep="\t")
        # self.ql["id"] = self.ql["queryID"]
        # del self.ql["queryID"]

        self.plantrain = pd.read_csv(f"{plan_rgcn_path}/train_pred.csv")
        self.planval = pd.read_csv(f"{plan_rgcn_path}/val_pred.csv")
        self.plantest = pd.read_csv(f"{plan_rgcn_path}/test_pred.csv")
        self.train = self.plantrain
        self.val = self.planval
        self.test = self.plantest
        self.train, self.val, self.test = self.add_query_log_info(dataset_path)
        # self.val = self.add_query_log_info(self.val)
        # self.test = self.add_query_log_info(self.test)
        tables = self.make_operator_table_f1()
        print("Training")
        print(tables[0].to_latex(caption="Training Performance on Operators"))
        print("Validation")
        print(tables[1].to_latex(caption="Validation Performance on Operators"))
        print("Test")
        print(tables[2].to_latex(caption="Test Performance on Operators"))
        # self.train_cls = self.make_cls_dfs(self.train)
        # self.val_cls = self.make_cls_dfs(self.val)
        # self.test_cls = self.make_cls_dfs(self.test)
        _, _, _, _, train_confusion = self.print_metrics_rgcn(
            self.train, average=average
        )
        _, _, _, _, val_confusion = self.print_metrics_rgcn(self.val, average=average)
        _, _, _, _, test_confusion = self.print_metrics_rgcn(self.test, average=average)
        for i, cap in zip(
            [train_confusion, val_confusion, test_confusion],
            [
                "Training Confusion Matrix",
                "Validation Confusion Matrix",
                "Test Confusion Matrix",
            ],
        ):
            df = pd.DataFrame(i, columns=interval, index=interval)
            print(df.to_latex(escape="\\", caption=cap))

    def make_operator_table_f1(self, algorithm="planrgcn_prediction", decimals=4):
        lst = []
        for df in [self.train, self.val, self.test]:
            cols = ["simple", "opt", "filt", "single", "multiple"]
            res = {}
            for c in cols:
                res_item = {}
                df1 = df[
                    QueryCls.get_features_filter(df, query_type=c)
                ]  # df[~((df[c] >= 1))]
                res_item["Queries"] = len(df1)
                accuracy, precision, recall, f1, confusion = self.get_metrics_cls(
                    df1, col=algorithm
                )
                accuracy, precision, recall, f1 = (
                    round(accuracy, decimals),
                    round(precision, decimals),
                    round(recall, decimals),
                    round(f1, decimals),
                )
                res_item["acc"] = accuracy
                res_item["prec"] = precision
                res_item["recall"] = recall
                res_item["f1"] = f1
                res[c] = res_item
            """df1 = df
            accuracy, precision, recall, f1, confusion = self.get_metrics_cls(
                df1, col=algorithm
            )
            res_item = {}
            res_item["acc"] = accuracy
            res_item["prec"] = precision
            res_item["recall"] = recall
            res_item["Queries"] = len(df1)
            res_item["f1"] = f1
            res["path"] = res_item"""
            lst.append(pd.DataFrame.from_dict(res, orient="index"))
        return lst

    def add_query_log_info(self, datasetpath):
        train = pd.read_csv(f"{datasetpath}/train_sampled.tsv", sep="\t")
        train["id"] = train["queryID"]
        if not self.is_lsq:
            train["id"] = train["id"].astype(np.int64)
            self.train["id"] = self.train["id"].apply(lambda x: x[20:])
            self.train["id"] = self.train["id"].astype(np.int64)

        del train["queryID"]
        self.train = self.train.merge(train, how="left", on="id")

        val = pd.read_csv(f"{datasetpath}/val_sampled.tsv", sep="\t")
        val["id"] = val["queryID"]
        if not self.is_lsq:
            self.val["id"] = self.val["id"].apply(lambda x: x[20:])
            val["id"] = val["id"].astype(np.int64)
            self.val["id"] = self.val["id"].astype(np.int64)
        del val["queryID"]
        self.val = self.val.merge(val, how="left", on="id")

        test = pd.read_csv(f"{datasetpath}/test_sampled.tsv", sep="\t")
        test["id"] = test["queryID"]
        if not self.is_lsq:
            self.test["id"] = self.test["id"].apply(lambda x: x[20:])
            test["id"] = test["id"].astype(np.int64)
            self.test["id"] = self.test["id"].astype(np.int64)
        del test["queryID"]
        self.test = self.test.merge(test, how="left", on="id")
        return self.train, self.val, self.test

    def print_metrics_rgcn(self, df, decimals=4, average="macro"):
        accuracy, precision, recall, f1, confusion = self.get_metrics_cls(
            df, col=f"planrgcn_prediction", v=2, average=average
        )
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro): {precision:.4f}")
        print(f"Recall (Macro): {recall:.4f}")
        print(f"F1-Score (Macro): {f1:.4f}")
        print("Confusion Matrix:")
        print(confusion)
        return (
            round(accuracy, decimals),
            round(precision, decimals),
            round(recall, decimals),
            round(f1, decimals),
            confusion,
        )

    def print_metrics(self, df, col="svm_prediction"):
        accuracy, precision, recall, f1, confusion = self.get_metrics_cls(
            df, col=f"{col}"
        )
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro): {precision:.4f}")
        print(f"Recall (Macro): {recall:.4f}")
        print(f"F1-Score (Macro): {f1:.4f}")
        print("Confusion Matrix:")
        print(confusion)
        r_squared, rmse = self.get_metrics_regression(df, col=col)
        print(f"R-squared (R²): {r_squared:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    def make_cls_dfs(self, df1: pd.DataFrame):
        df = df1.copy()
        for column in ["time", "nn_prediction", "svm_prediction"]:
            filt_df_001 = df[column] < 0.01
            filt_df_01 = (0.01 < df[column]) & (df[column] < 0.1)
            filt_df_1 = (0.1 < df[column]) & (df[column] < 1)
            filt_df_10 = (1 < df[column]) & (df[column] < 10)
            filt_df_100 = (10 < df[column]) & (df[column] < 100)
            filt_df_above = df[column] > 100
            label = f"{column}_cls"
            df[label] = -1
            df[label][filt_df_001] = 0
            df[label][filt_df_01] = 1
            df[label][filt_df_1] = 2
            df[label][filt_df_10] = 3
            df[label][filt_df_100] = 4
            df[label][filt_df_above] = 5
            del df[column]
        return df

    def merge_dfs(self, df1, df2, df3):
        merged = pd.merge(
            df1,
            df2,
            how="inner",
            left_on=["id"],
            right_on=["id"],
            suffixes=("", "_remove"),
        )
        merged.drop(
            [i for i in merged.columns if "remove" in i or "Unnamed" in i],
            axis=1,
            inplace=True,
        )
        merged = pd.merge(
            merged,
            df3,
            how="inner",
            left_on=["id"],
            right_on="id",
            suffixes=("", "_remove"),
        )
        merged.drop(
            [i for i in merged.columns if "remove" in i or "Unnamed" in i],
            axis=1,
            inplace=True,
        )
        return merged

    # todo bin latencies before doing this. #For one df
    def plot_scatterplot(self, df, pdf=False):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.scatterplot(
            x=df["time"],
            y=df["nn_prediction"],
            label="Neural Network",
            ax=axes[0],
        )
        axes[0].set_xlabel("Actual Runtime (s)")
        axes[0].set_ylabel("Predicted runtime (s)")
        axes[0].set_title("Neural Network w. Autoencoder")

        sns.scatterplot(
            x=df["time"],
            y=df["svm_prediction"],
            label="Support Vector Machine",
            ax=axes[1],
        )
        axes[1].set_xlabel("Actual Runtime (s)")
        axes[1].set_ylabel("Predicted runtime (s)")
        axes[1].set_title("Support Vector Machine")
        axes[1].legend()
        plt.tight_layout()
        if pdf:
            plt.savefig("scatterplot.pdf")
        else:
            plt.savefig("scatterplot.png")

    def scatterplot_train_val_test(self, pdf=False):
        # Create a figure with 6 scatter plots (train, validation, and test for two approaches)
        plt.figure(figsize=(15, 10))

        # Approach Neural Network
        name = "Neural Network w. Autoencoder"
        plt.subplot(2, 3, 1)
        sns.scatterplot(
            x=self.train["time"],
            y=self.train["nn_prediction"],
            label=f"Train Data ({name})",
        )
        plt.xlabel("Actual Runtime (s)")
        plt.ylabel("Predicted runtime (s)")
        plt.legend(loc="upper right")
        plt.title(f"Train Data - {name}")

        plt.subplot(2, 3, 2)
        sns.scatterplot(
            x=self.val["time"],
            y=self.val["nn_prediction"],
            label=f"Validation Data ({name})",
        )
        plt.xlabel("Actual Runtime (s)")
        plt.ylabel("Predicted runtime (s)")
        plt.legend(loc="upper right")
        plt.title(f"Validation Data - {name}")

        plt.subplot(2, 3, 3)
        sns.scatterplot(
            x=self.test["time"],
            y=self.test["nn_prediction"],
            label=f"Test Data ({name})",
        )
        plt.xlabel("Actual Runtime (s)")
        plt.ylabel("Predicted runtime (s)")
        plt.legend(loc="upper right")
        plt.title(f"Test Data - {name}")

        # Approach Support Vector Machine
        name = "Support Vector Machine"
        plt.subplot(2, 3, 4)
        sns.scatterplot(
            x=self.train["time"],
            y=self.train["svm_prediction"],
            label=f"Train Data ({name})",
        )
        plt.xlabel("Actual Runtime (s)")
        plt.ylabel("Predicted runtime (s)")
        plt.legend(loc="upper right")
        plt.title(f"Train Data - {name}")

        plt.subplot(2, 3, 5)
        sns.scatterplot(
            x=self.val["time"],
            y=self.val["svm_prediction"],
            label=f"Validation Data ({name})",
        )
        plt.xlabel("Actual Runtime (s)")
        plt.ylabel("Predicted runtime (s)")
        plt.legend(loc="upper right")
        plt.title(f"Validation Data - {name}")

        plt.subplot(2, 3, 6)
        sns.scatterplot(
            x=self.test["time"],
            y=self.test["svm_prediction"],
            label=f"Test Data ({name})",
        )
        plt.xlabel("Actual Runtime (s)")
        plt.ylabel("Predicted runtime (s)")
        plt.legend(loc="upper right")
        plt.title(f"Test Data - {name}")

        plt.tight_layout()
        if pdf:
            plt.savefig("scatterplot.pdf")
        else:
            plt.savefig("scatterplot.png")

    def get_metrics_regression(self, df: pd.DataFrame, col="nn_prediction"):
        actual_values = df["time"]
        predicted_values = df[col]
        # Calculate R-squared (R²)
        r_squared = r2_score(actual_values, predicted_values)

        # Calculate Root Mean Squared Error (RMSE)
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = math.sqrt(mse)
        return r_squared, rmse

    def get_metrics_cls(
        self, df: pd.DataFrame, col="nn_prediction", v=0, average="macro"
    ):
        """
        average (default "macro"): should either be binary or macro (macro is classwise average)
        """

        actual_values = list(df["time_cls"])
        predicted_values = list(df["planrgcn_prediction"])

        # Calculate accuracy
        accuracy = accuracy_score(actual_values, predicted_values)

        # Calculate precision for each class (macro-average)
        precision = precision_score(actual_values, predicted_values, average=average)

        # Calculate recall for each class (macro-average)
        recall = recall_score(actual_values, predicted_values, average=average)

        # Calculate F1-score for each class (macro-average)
        f1 = f1_score(actual_values, predicted_values, average=average)

        # Calculate the confusion matrix
        confusion = confusion_matrix(actual_values, predicted_values)
        return (
            accuracy,
            precision,
            recall,
            f1,
            confusion,
        )


# to compare against PlanRGCN
def snap_lat2onehot(lat):
    vec = np.zeros(6)
    if lat < 0.01:
        vec[0] = 1
    elif (0.01 < lat) and (lat < 0.1):
        vec[1] = 1
    elif (0.1 < lat) and (lat < 1):
        vec[2] = 1
    elif (1 < lat) and (lat < 10):
        vec[3] = 1
    elif 10 < lat and lat < 100:
        vec[4] = 1
    elif lat > 100:
        vec[5] = 1

    return vec


def snap_lat2onehotv2(lat):
    vec = np.zeros(3)
    if lat < 1:
        vec[0] = 1
    elif (1 < lat) and (lat < 10):
        vec[1] = 1
    elif 10 < lat:
        vec[2] = 1
    return vec


def snap_lat2onehot_binary(lat):
    vec = np.zeros(2)
    if lat < 1:
        vec[0] = 1
    else:
        vec[1] = 1
    return vec


def lat2cat(lat):
    return np.argmax(snap_lat2onehot_binary(lat))


class DatasetStats(Plotter):
    def __init__(
        self,
        NN_path="/qpp/dataset/DBpedia_2016_12k_sample/results/nn/k10",
        svm_path="/qpp/dataset/DBpedia_2016_12k_sample/results/svm",
    ) -> None:
        super().__init__(NN_path, svm_path)

    def get_distribution(self, df, column="time"):
        filt_df_001 = df[column] < 0.01
        filt_df_01 = (0.01 < df[column]) & (df[column] < 0.1)
        filt_df_1 = (0.1 < df[column]) & (df[column] < 1)
        filt_df_10 = (1 < df[column]) & (df[column] < 10)
        filt_df_100 = (10 < df[column]) & (df[column] < 100)
        filt_df_above = df[column] > 100

        return (
            df[filt_df_001],
            df[filt_df_01],
            df[filt_df_1],
            df[filt_df_10],
            df[filt_df_100],
            df[filt_df_above],
        )

    def print_dist(self):
        train_d = [len(x) for x in self.get_distribution(self.train)]
        val_d = [len(x) for x in self.get_distribution(self.val)]
        test_d = [len(x) for x in self.get_distribution(self.test)]
        print(train_d)
        print(val_d)
        print(test_d)


if __name__ == "__main__":
    # d = DatasetStats()
    # d.print_dist()
    p = Plotter(
        plan_rgcn_path="/PlanRGCN/dbpedia2016/results3",
        dataset_path="/qpp/dataset/DBpedia_2016_12k_simple_opt_filt",
        average="binary",
    )
    """p = Plotter(
        plan_rgcn_path="/PlanRGCN/wdbench/results",
        dataset_path="/qpp/dataset/wdbench",
        is_lsq=False,
        interval=["0 - 1", "1 -10", "> 10"],
        average="macro",
    )"""
    # p.scatterplot_train_val_test(pdf=True)
