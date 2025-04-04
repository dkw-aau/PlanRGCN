from urllib.parse import unquote
import pandas as pd


class DataPreparations:
    def __init__(
        self,
        train_f="/qpp/dataset/original_data/queries/x_query.txt",
        val_f="/qpp/dataset/original_data/queries/xval_query.txt",
        test_f="/qpp/dataset/original_data/queries/xtest_query.txt",
        ex_time={
            "train": "/qpp/dataset/original_data/execution_time/y_time.txt",
            "val": "/qpp/dataset/original_data/execution_time/yval_time.txt",
            "test": "/qpp/dataset/original_data/execution_time/ytest_time.txt",
        },
        save_path={
            "train": "/qpp/dataset/original_data/train.tsv",
            "val": "/qpp/dataset/original_data/val.tsv",
            "test": "/qpp/dataset/original_data/test.tesv",
            "all": "/qpp/dataset/original_data/all.tsv",
        },
    ) -> None:
        self.save_path = save_path

        self.train_queries = self.load_queries(train_f)
        self.train_queries = pd.Series(self.train_queries)
        self.val_queries = self.load_queries(val_f)
        self.val_queries = pd.Series(self.val_queries)
        self.test_queries = self.load_queries(test_f)
        self.test_queries = pd.Series(self.test_queries)

        self.train_ex = self.load_ex_time(ex_time["train"])
        self.train_ex = pd.Series(self.train_ex)
        self.val_ex = self.load_ex_time(ex_time["val"])
        self.val_ex = pd.Series(self.val_ex)
        self.test_ex = self.load_ex_time(ex_time["test"])
        self.test_ex = pd.Series(self.test_ex)
        self.merge_to_dfs()

    def load_queries(self, path: str) -> list[str]:
        qs = list()
        with open(path, "r") as f:
            for l in f.readlines():
                q = unquote(l[6:]).replace("+", " ").replace("\n", " ")
                qs.append(q)
        return qs

    def load_ex_time(self, path: str) -> list[str]:
        qs = list()
        with open(path, "r") as f:
            for l in f.readlines():
                if l.startswith("ex_time"):
                    continue
                qs.append(float(l))
        return qs

    def save_to_path(self):
        self.train.to_csv(self.save_path["train"], index=False, sep="\t")
        self.train.to_csv(self.save_path["val"], index=False, sep="\t")
        self.all.to_csv(self.save_path["all"], index=False, sep="\t")

    def merge_to_dfs(self):
        self.train = pd.DataFrame()
        self.train["queryString"] = self.train_queries
        self.train["queryID"] = self.train.index
        self.train["duration"] = self.train_ex
        self.train["resultCount"] = None

        self.val = pd.DataFrame()
        self.val["queryString"] = self.val_queries
        self.val["queryID"] = self.val.index
        self.val["duration"] = self.val_ex
        self.val["resultCount"] = None

        self.test = pd.DataFrame()
        self.test["queryString"] = self.test_queries
        self.test["queryID"] = self.test.index
        self.test["duration"] = self.test_ex
        self.test["resultCount"] = None
        self.test.to_csv(self.save_path["test"], index=False, sep="\t")
        self.all = pd.concat([self.train, self.val, self.test])
        self.all.reset_index()


d = DataPreparations()
print(d.train)
d.save_to_path()
