import json
import pickle
from typing import Tuple, Any

import jpype
from sklearn.model_selection import StratifiedShuffleSplit

from graph_construction.jar_utils import get_query_graph
from query_log_importer import TSVFileQLImporter
from time_interval import PercentileTimeIntervalExtractor
import numpy as np
import pandas as pd
from collections import Counter

class DataSplitter:

    def __init__(self, tsv_file=None, interval_type='percentile', percentiles=[0, 50, 92, 100]):
        self.column = "mean_latency"
        self.strat_col = "strat_col"
        self.test_size = 0.2
        self.seed = 25

        # other fields of note:
        # self.interval_extractor.intervals : will contain a list of tuples reprensting workload dependant runtime intervals
        self.query_df = TSVFileQLImporter(tsv_file).get_query_log() if tsv_file != None else None

        np.random.seed(self.seed)

        match interval_type:
            case 'percentile':
                # Consider using [0,50,90,100] percentile after removing some queries from the wikidata query log, such that 90% is reasonable
                self.interval_extractor = PercentileTimeIntervalExtractor(self.query_df['mean_latency'],
                                                                          percentiles)  # defines intervals for quantiles
                print(self.interval_extractor.print_intervals())
                print(self.interval_extractor.thresholds)



    def make_splits_files(self, train_file=None, val_file = None, test_file=None, splt_info_file=None,sep='\t', save_obj_file = None, intervals_file = None):

        def strfy_info(dct):
            for ko in dct.keys():
                for i in ['train', 'val', 'test']:
                    t_dct = {}
                    for k in dct[ko][i].keys():
                        t_dct[str(k)] = dct[ko][i][k]
                    dct[ko][i] = t_dct
            return dct

        train, val, test, splt_info = self.stratify()

        train: pd.DataFrame
        val: pd.DataFrame
        test: pd.DataFrame

        train.to_csv(train_file, sep=sep, index=False)
        val.to_csv(val_file, sep=sep, index=False)
        test.to_csv(test_file, sep=sep, index=False)
        with open(intervals_file, 'w') as f:
            f.write(str(self.interval_extractor.intervals))
            f.flush()

        splt_info = strfy_info(splt_info)
        with open(splt_info_file, 'w') as f:
            json.dump(splt_info, f)
        print(f'Successfully created dataset split with following statistcs')
        print(json.dumps(splt_info))
        with open(save_obj_file, 'wb') as f:
            pickle.dump(self, f)


    def stratify(
            self,
    ) -> tuple[Any, Any, Any, Any]:
        self.query_df[self.strat_col] = self.query_df[self.column].apply(
            lambda x: self.interval_extractor.find_query_intervals(x))

        # Get optional queries
        opt_ids = []
        pp_ids = []
        remaining = []
        filter_ids = []
        for idx, row in self.query_df.iterrows():
            try:
                qg = get_query_graph(row['queryString'])
            except jpype.JException:
                continue
            if np.sum([1 for x in qg['nodes'] if x['nodeType'] == 'PP']) > 0:
                pp_ids.append(idx)
            elif np.sum([1 for x in qg['edges'] if x[2] == 'Optional']) > 0: #for optional
                opt_ids.append(idx)
            elif np.sum([1 for x in qg['nodes'] if x['nodeType'] == 'FILTER']) > 0:
                filter_ids.append(idx)
            else:
                remaining.append(idx)

        trains = []
        vals = []
        tests = []
        splt_info = {}
        for q_type, ids in zip(['PP', 'Optionals', 'FILTERS', 'Regular'],[pp_ids, opt_ids, filter_ids, remaining]):
            df = self.query_df.loc[ids].reset_index()
            train, val, test, spl_info = self.stratify_df(df, query_type=q_type)
            trains.append(train)
            vals.append(val)
            tests.append(test)
            splt_info[q_type] = spl_info
        trains = pd.concat(trains).reset_index()
        vals = pd.concat(vals).reset_index()
        tests = pd.concat(tests).reset_index()
        return trains, vals, tests, splt_info

    def stratify_df(self, df: pd.DataFrame, query_type='none'):
        train_set = None
        validation_set = None
        test_set = None
        # Calculate the size of the validation set as a proportion of the whole dataset
        val_size = self.test_size / (1 - self.test_size)

        stratified_split = StratifiedShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.seed
        )
        for train_index, test_index in stratified_split.split(df, df[self.strat_col]):
            train_set = df.loc[train_index]
            test_set = df.loc[test_index]

        train_set = train_set.reset_index(drop=True)

        # Instantiate another StratifiedShuffleSplit for the second split
        stratified_split = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=self.seed
        )
        # Perform the stratified split for the second split
        for train_index, validation_index in stratified_split.split(
                train_set, train_set[self.strat_col]
        ):
            train_set_temp = train_set.loc[train_index]
            validation_set = train_set.loc[validation_index]
        train_set = train_set_temp

        splt_info = {'query_type': query_type, 'train':dict(Counter(train_set[self.strat_col])), 'val':dict(Counter(validation_set[self.strat_col])), 'test':dict(Counter(test_set[self.strat_col]))}
        del train_set[self.strat_col]
        del validation_set[self.strat_col]
        del test_set[self.strat_col]

        return train_set, validation_set, test_set, splt_info