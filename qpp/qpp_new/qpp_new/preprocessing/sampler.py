import numpy as np
import math
import pandas as pd
import os


class Sampler:
    def __init__(
        self,
        dataset_path="/qpp/dataset/DBpedia_2016_sampled/benchmark.tsv",
        output_dir="/qpp/dataset/DBpedia_2016_12k_sample/",
        random_seed=25,
        train_distribution={
            0: 883,  # '[0:0.01]'    #11.2%
            1: 2000,  # '[0.01:0.1]' #25.5%
            2: 2000,  # '[0.1:1]'    #25.5%
            3: 1787,  # '[1:10]'     #22.8%
            4: 931,  # '[10:100]'   #11.9%
            5: 227  # '[100:]'     #2.9%
            # total 7828
        },
        querylog="/SPARQLBench/dbpedia2015_16/ordered_queries2015_2016_clean_w_stat.tsv",
    ) -> None:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f'The file path "{dataset_path}" does not exist')
        self.dataset_path = dataset_path

        if not os.path.exists(querylog):
            raise FileNotFoundError(
                f'The file path "{querylog}" does not exist for query log'
            )
        self.querylog = querylog

        os.system(f"mkdir -p {output_dir}")
        self.output_dir = output_dir
        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.train_distribution = train_distribution
        """self.train_distribution = {0: 100, # '[0:0.01]'
                    1: 100, # '[0.01:0.1]'
                    2:100,  # '[0.1:1]'
                    3:100,  # '[1:10]'
                    4: 50,  # '[10:100]'
                    5: 25   # '[100:]'
                    } # 475 in total"""

    # After collecting the latencies in Virtoso, a new sample is required. (for entire dataset instead of just 20,000 sample)
    def post_sample(self, column="mean_latency"):
        # for val
        other_dct = {}
        for k in self.train_distribution.keys():
            # 20% of queries are chosen for validation and test set. (We could consider having all of the remaining queries for test set.)
            other_dct[k] = Sampler.queries_for_test_and_val(
                self.train_distribution[k], train_percentage=60
            )
            self.train_distribution[k] = self.train_distribution[k] - other_dct[k] * 2
        # test contains remaining queries in ideal situation.
        df = pd.read_csv(self.dataset_path, sep="\t")
        train, val, test = [], [], []
        df_001 = df[df[column] < 0.01]
        df_01 = df[(0.01 < df[column]) & (df[column] < 0.1)]
        df_1 = df[(0.1 < df[column]) & (df[column] < 1)]
        df_10 = df[(1 < df[column]) & (df[column] < 10)]
        df_100 = df[(10 < df[column]) & (df[column] < 100)]
        df_above = df[df[column] > 100]
        for z, d in enumerate([df_001, df_01, df_1, df_10, df_100, df_above]):
            order = [x for x in range(len(d))]
            np.random.shuffle(order)
            try:
                train.append(d.iloc[order[: self.train_distribution[z]]])
            except TypeError:
                print(f"Type error on indexing: {self.train_distribution[z]}")
            order = order[self.train_distribution[z] :]
            split_val = other_dct[z]
            if len(order) < (other_dct[z] * 2):
                split_val = math.floor((len(order) / 2))
            val.append(d.iloc[order[:split_val]])
            order = order[split_val:]
            test.append(d.iloc[order[:split_val]])
        train = pd.concat(train)
        train = Sampler.merge_algebra_features(train, querylog=self.querylog)
        train.to_csv(f"{self.output_dir}train_sampled.tsv", sep="\t", index=False)
        val = pd.concat(val)
        val = Sampler.merge_algebra_features(val, querylog=self.querylog)
        val.to_csv(f"{self.output_dir}val_sampled.tsv", sep="\t", index=False)
        test = pd.concat(test)
        test = Sampler.merge_algebra_features(test, querylog=self.querylog)
        test.to_csv(f"{self.output_dir}test_sampled.tsv", sep="\t", index=False)

    def queries_for_test_and_val(train_amount, train_percentage=60):
        val_amount = train_amount * ((100 - train_percentage) / 100)
        test_amount = math.floor(val_amount / 2)
        # TODO make adjustment to code to consider entire sample/dataset
        # val_amount = val_amount-test_amount

        # x = (train_amount * (100/train_percentage))* ((100-train_percentage)/100)
        # x = math.floor(x/2)
        return test_amount

    def merge_algebra_features(
        df: pd.DataFrame,
        querylog="/SPARQLBench/dbpedia2015_16/ordered_queries2015_2016_clean_w_stat.tsv",
    ):
        df.set_index("id", inplace=True)
        q_df = pd.read_csv(querylog, sep="\t")
        q_df.set_index("queryID", inplace=True)
        joined = df.join(q_df, how="inner")
        joined = joined.reset_index()
        joined["queryID"] = joined["index"]
        del joined["index"]
        return joined


if __name__ == "__main__":
    dataset_path = "/SPARQLBench/plots/0_12400/latency_log.tsv"
    sampler = Sampler(dataset_path=dataset_path)
    sampler.post_sample()
    pass
